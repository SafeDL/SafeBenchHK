"""
通用端到端自动驾驶模型 - CarlaEnv 基类模板
使用说明:
1. 修改 【模型专属配置区】 的参数，适配你的模型需求
2. 在 【模型专属逻辑】 区域实现定制化功能（相机、观测、奖励等）
3. 其余通用逻辑无需修改
"""
import random
import numpy as np
import pygame
from skimage.transform import resize
import gym
from gym import spaces
import carla

# ======================== 【模型专属配置区 - 开始】 ========================
# 1. 模型专用传感器配置（示例：BEV模型/激光雷达模型）
MODEL_CAMERA_WIDTH = 1280  # 模型输入图像宽度
MODEL_CAMERA_HEIGHT = 720  # 模型输入图像高度
MODEL_CAMERA_FOV = 90  # 相机视场角
MODEL_CAMERA_POS = (0.8, 1.7) # 相机安装位置 (x, z)
# 2. 模型专用观测键值（如需要添加 BEV 特征/LiDAR 点云）
MODEL_OBS_KEYS = ['camera_raw']  # 新增观测字段
# 3. 导入模型依赖的场景/工具类
# from safebench.scenario.scenario_definition.your_scenario import YourScenario
# from safebench.gym_carla.envs.your_tool import YourTool
# ======================== 【模型专属配置区 - 结束】 ========================

from safebench.gym_carla.envs.route_planner import RoutePlanner
from safebench.gym_carla.envs.misc import (
    display_to_rgb,
    rgb_to_display_surface,
    get_lane_dis,
    get_pos,
    get_preview_lane_dis
)
from safebench.scenario.scenario_definition.route_scenario import RouteScenario
from safebench.scenario.scenario_manager.scenario_manager import ScenarioManager
from safebench.carla_agents.navigation.global_route_planner import GlobalRoutePlanner


class CarlaEnvModel(gym.Env):
    """
    通用 CARLA 仿真环境基类（Gym 接口）
    适配任意端到端自动驾驶模型，需修改 【模型专属逻辑】 区域
    """

    def __init__(self, env_params, birdeye_render=None, display=None, world=None, logger=None):
        # 必检参数：CARLA world 不能为空
        assert world is not None, "the world passed into CarlaEnvModel is None"

        # ======================== 【通用逻辑】环境基础配置 ========================
        self.config = None
        self.world = world
        self.display = display
        self.logger = logger
        self.birdeye_render = birdeye_render

        # 步数记录
        self.reset_step = 0
        self.total_step = 0
        self.is_running = True
        self.env_id = None
        self.ego_vehicle = None
        self.auto_ego = env_params['auto_ego']

        # 传感器对象初始化
        self.collision_sensor = None
        self.lidar_sensor = None
        self.camera_sensor = None
        self.lidar_data = None
        self.lidar_height = 2.1

        # ======================== 【模型专属逻辑】添加模型专用传感器数据 ========================
        # 示例：TCP 添加 camera_img_raw 存储原始尺寸图像
        self.camera_img_raw = None  # 模型专用原始图像
        # self.bev_feature = None    # 如 BEV 模型需添加 BEV 特征存储

        # ======================== 【通用逻辑】场景管理器初始化 ========================
        self.scenario_manager = ScenarioManager(self.logger)

        # 可视化参数
        self.display_size = env_params['display_size']
        self.obs_range = env_params['obs_range']
        self.d_behind = env_params['d_behind']
        self.disable_lidar = env_params['disable_lidar']

        # 环境包装器参数
        self.max_past_step = env_params['max_past_step']
        self.max_episode_step = env_params['max_episode_step']
        self.max_waypt = env_params['max_waypt']
        self.lidar_bin = env_params['lidar_bin']
        self.out_lane_thres = env_params['out_lane_thres']
        self.desired_speed = env_params['desired_speed']
        self.acc_max = env_params['continuous_accel_range'][1]
        self.steering_max = env_params['continuous_steer_range'][1]

        # 场景参数
        self.ROOT_DIR = env_params['ROOT_DIR']
        self.scenario_category = env_params['scenario_category']
        self.warm_up_steps = env_params['warm_up_steps']
        self.running_results = {}

        # ======================== 【通用逻辑】观测空间定义 ========================
        if self.scenario_category in ['planning']:
            self.obs_size = int(self.obs_range / self.lidar_bin)
            observation_space_dict = {
                'camera': spaces.Box(low=0, high=255, shape=(self.obs_size, self.obs_size, 3), dtype=np.uint8),
                'lidar': spaces.Box(low=0, high=255, shape=(self.obs_size, self.obs_size, 3), dtype=np.uint8),
                'birdeye': spaces.Box(low=0, high=255, shape=(self.obs_size, self.obs_size, 3), dtype=np.uint8),
                'state': spaces.Box(np.array([-2, -1, -5, 0], dtype=np.float32),
                                    np.array([2, 1, 30, 1], dtype=np.float32), dtype=np.float32)
            }
        elif self.scenario_category == 'perception':
            self.obs_size = env_params['image_sz']
            observation_space_dict = {
                'camera': spaces.Box(low=0, high=255, shape=(self.obs_size, self.obs_size, 3), dtype=np.uint8),
            }
        else:
            raise ValueError(f'Unknown scenario category: {self.scenario_category}')

        # ======================== 【模型专属逻辑】扩展观测空间 ========================
        # 示例：TCP 添加 camera_raw 观测字段
        for obs_key in MODEL_OBS_KEYS:
            if obs_key == 'camera_raw':
                observation_space_dict[obs_key] = spaces.Box(
                    low=0, high=255,
                    shape=(MODEL_CAMERA_HEIGHT, MODEL_CAMERA_WIDTH, 3),
                    dtype=np.uint8
                )
            # 可添加其他观测字段（如 bev_feature / lidar_pointcloud）

        # 最终观测空间
        self.observation_space = spaces.Dict(observation_space_dict)

        # ======================== 【通用逻辑】动作空间定义 ========================
        self.discrete = env_params['discrete']
        self.discrete_act = [env_params['discrete_acc'], env_params['discrete_steer']]
        self.n_acc = len(self.discrete_act[0])
        self.n_steer = len(self.discrete_act[1])

        if self.discrete:
            self.action_space = spaces.Discrete(self.n_acc * self.n_steer)
        else:
            self.action_space = spaces.Box(
                np.array([-1, -1], dtype=np.float32),
                np.array([1, 1], dtype=np.float32),
                dtype=np.float32
            )

    # ======================== 【通用逻辑】传感器创建（可重载） ========================
    def _create_sensors(self):
        # 碰撞传感器
        self.collision_hist_l = 1
        self.collision_bp = self.world.get_blueprint_library().find('sensor.other.collision')

        # LiDAR 传感器（规划类场景启用）
        if self.scenario_category != 'perception':
            self.lidar_trans = carla.Transform(carla.Location(x=0.0, z=self.lidar_height))
            self.lidar_bp = self.world.get_blueprint_library().find('sensor.lidar.ray_cast')
            self.lidar_bp.set_attribute('channels', '16')
            self.lidar_bp.set_attribute('range', '1000')

        # ======================== 【模型专属逻辑】相机传感器配置 ========================
        # 示例：TCP 自定义相机尺寸、FOV、安装位置
        self.camera_img = np.zeros((MODEL_CAMERA_HEIGHT, MODEL_CAMERA_WIDTH, 3), dtype=np.uint8)
        self.camera_trans = carla.Transform(carla.Location(x=MODEL_CAMERA_POS[0], z=MODEL_CAMERA_POS[1]))
        self.camera_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
        self.camera_bp.set_attribute('image_size_x', str(MODEL_CAMERA_WIDTH))
        self.camera_bp.set_attribute('image_size_y', str(MODEL_CAMERA_HEIGHT))
        self.camera_bp.set_attribute('fov', str(MODEL_CAMERA_FOV))
        self.camera_bp.set_attribute('sensor_tick', '0.02')

    # ======================== 【通用逻辑】场景创建 ========================
    def _create_scenario(self, config, env_id):
        self.logger.log(f">> Loading scenario data id: {config.data_id}")

        if self.scenario_category == 'planning':
            scenario = RouteScenario(
                world=self.world,
                config=config,
                ego_id=env_id,
                max_running_step=self.max_episode_step,
                logger=self.logger
            )
        # ======================== 【模型专属逻辑】扩展场景类型 ========================
        # elif self.scenario_category == 'your_scenario':
        #     scenario = YourScenario(...)
        else:
            raise ValueError(f'Unknown scenario category: {self.scenario_category}')

        self.ego_vehicle = scenario.ego_vehicle
        self.scenario_manager.load_scenario(scenario)

    # ======================== 【通用逻辑】场景运行 ========================
    def _run_scenario(self, scenario_init_action):
        self.scenario_manager.run_scenario(scenario_init_action)

    # ======================== 【通用逻辑】路径解析 ========================
    def _parse_route(self, config):
        wmap = self.world.get_map()
        global_planner = GlobalRoutePlanner(wmap, sampling_resolution=2.0)
        start_location = config.trajectory[0]
        end_location = config.trajectory[-1]
        route = global_planner.trace_route(start_location, end_location)

        waypoints_list = []
        for transform_tuple in route:
            waypoints_list.append(transform_tuple[0])
        return waypoints_list

    # ======================== 【通用逻辑】静态观测获取 ========================
    def get_static_obs(self, config):
        wmap = self.world.get_map()
        global_planner = GlobalRoutePlanner(wmap, sampling_resolution=5.0)
        start_location = config.trajectory[0]
        end_location = config.trajectory[-1]
        route = global_planner.trace_route(start_location, end_location)

        waypoint_xy = []
        for transform_tuple in route:
            waypoint_xy.append([transform_tuple[0].transform.location.x, transform_tuple[0].transform.location.y])

        state = {
            'route': np.array(waypoint_xy),
            'target_speed': self.desired_speed,
        }
        return state

    # ======================== 【通用逻辑】环境重置 ========================
    def reset(self, config, env_id, scenario_init_action):
        self.config = config
        self.env_id = env_id

        # 传感器创建 → 场景创建 → 场景运行 → 传感器挂载
        self._create_sensors()
        self._create_scenario(config, env_id)
        self._run_scenario(scenario_init_action)
        self._attach_sensor()

        # 路径规划器初始化
        self.route_waypoints = self._parse_route(config)
        self.routeplanner = RoutePlanner(self.ego_vehicle, self.max_waypt, self.route_waypoints)
        self.waypoints, _, _, _, _, self.vehicle_front, = self.routeplanner.run_step()

        # 演员多边形/信息初始化
        self.vehicle_polygons = [self._get_actor_polygons('vehicle.*')]
        self.walker_polygons = [self._get_actor_polygons('walker.*')]

        vehicle_info_dict_list = self._get_actor_info('vehicle.*')
        self.vehicle_trajectories = [vehicle_info_dict_list[0]]
        self.vehicle_accelerations = [vehicle_info_dict_list[1]]
        self.vehicle_angular_velocities = [vehicle_info_dict_list[2]]
        self.vehicle_velocities = [vehicle_info_dict_list[3]]

        # 步数重置
        self.time_step = 0
        self.reset_step += 1

        # 仿真世界设置应用
        self.settings = self.world.get_settings()
        self.world.apply_settings(self.settings)

        # 预热步数
        for _ in range(self.warm_up_steps):
            self.world.tick()

        return self._get_obs(), self._get_info()

    # ======================== 【通用逻辑】传感器挂载 ========================
    def _attach_sensor(self):
        # 碰撞传感器挂载
        self.collision_sensor = self.world.spawn_actor(self.collision_bp, carla.Transform(), attach_to=self.ego_vehicle)
        self.collision_hist = []

        def get_collision_hist(event):
            impulse = event.normal_impulse
            intensity = np.sqrt(impulse.x ** 2 + impulse.y ** 2 + impulse.z ** 2)
            self.collision_hist.append(intensity)
            if len(self.collision_hist) > self.collision_hist_l:
                self.collision_hist.pop(0)

        self.collision_sensor.listen(lambda event: get_collision_hist(event))

        # LiDAR 传感器挂载
        if self.scenario_category != 'perception' and not self.disable_lidar:
            self.lidar_sensor = self.world.spawn_actor(self.lidar_bp, self.lidar_trans, attach_to=self.ego_vehicle)

            def get_lidar_data(data):
                self.lidar_data = data

            self.lidar_sensor.listen(lambda data: get_lidar_data(data))

        # ======================== 【模型专属逻辑】相机传感器挂载 ========================
        self.camera_sensor = self.world.spawn_actor(self.camera_bp, self.camera_trans, attach_to=self.ego_vehicle)

        def get_camera_img(data):
            array = np.frombuffer(data.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (data.height, data.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]
            self.camera_img = array
            # 示例：TCP 保存原始尺寸图像
            self.camera_img_raw = array.copy()
            # 可添加 BEV 特征提取逻辑

        self.camera_sensor.listen(lambda data: get_camera_img(data))

    # ======================== 【通用逻辑】步前动作执行 ========================
    def step_before_tick(self, ego_action, scenario_action):
        if self.world:
            snapshot = self.world.get_snapshot()
            if snapshot:
                timestamp = snapshot.timestamp

                # 感知类场景动作预处理
                if self.scenario_category in ['perception']:
                    assert isinstance(ego_action, dict), 'ego action should be a dict'
                    world_2_camera = np.array(self.camera_sensor.get_transform().get_inverse_matrix())
                    fov = self.camera_bp.get_attribute('fov').as_float()
                    image_w, image_h = MODEL_CAMERA_WIDTH, MODEL_CAMERA_HEIGHT
                    self.scenario_manager.background_scenario.evaluate(
                        ego_action, world_2_camera, image_w, image_h, fov, self.camera_img
                    )
                    ego_action = ego_action['ego_action']

                # 场景状态更新
                self.scenario_manager.get_update(timestamp, scenario_action)
                self.is_running = self.scenario_manager._running

                # 自车动作执行（非自动模式）
                if not self.auto_ego:
                    if self.discrete:
                        acc = self.discrete_act[0][ego_action // self.n_steer]
                        steer = self.discrete_act[1][ego_action % self.n_steer]
                    else:
                        acc = ego_action[0]
                        steer = ego_action[1]

                    # 动作归一化与裁剪
                    acc = acc * self.acc_max
                    steer = steer * self.steering_max
                    acc = max(min(self.acc_max, acc), -self.acc_max)
                    steer = max(min(self.steering_max, steer), -self.steering_max)

                    # 油门/刹车映射
                    if acc > 0:
                        throttle = np.clip(acc / 3, 0, 1)
                        brake = 0
                    else:
                        throttle = 0
                        brake = np.clip(-acc / 8, 0, 1)

                    # 应用车辆控制
                    act = carla.VehicleControl(throttle=float(throttle), steer=float(steer), brake=float(brake))
                    self.ego_vehicle.apply_control(act)
            else:
                self.logger.log('>> Can not get snapshot!', color='red')
                raise Exception()
        else:
            self.logger.log('>> Please specify a Carla world!', color='red')
            raise Exception()

    # ======================== 【通用逻辑】步后状态收集 ========================
    def step_after_tick(self):
        # 演员多边形更新
        vehicle_poly_dict = self._get_actor_polygons('vehicle.*')
        self.vehicle_polygons.append(vehicle_poly_dict)
        while len(self.vehicle_polygons) > self.max_past_step:
            self.vehicle_polygons.pop(0)

        walker_poly_dict = self._get_actor_polygons('walker.*')
        self.walker_polygons.append(walker_poly_dict)
        while len(self.walker_polygons) > self.max_past_step:
            self.walker_polygons.pop(0)

        # 演员信息更新
        vehicle_info_dict_list = self._get_actor_info('vehicle.*')
        self.vehicle_trajectories.append(vehicle_info_dict_list[0])
        while len(self.vehicle_trajectories) > self.max_past_step:
            self.vehicle_trajectories.pop(0)

        self.vehicle_accelerations.append(vehicle_info_dict_list[1])
        while len(self.vehicle_accelerations) > self.max_past_step:
            self.vehicle_accelerations.pop(0)

        self.vehicle_angular_velocities.append(vehicle_info_dict_list[2])
        while len(self.vehicle_angular_velocities) > self.max_past_step:
            self.vehicle_angular_velocities.pop(0)

        self.vehicle_velocities.append(vehicle_info_dict_list[3])
        while len(self.vehicle_velocities) > self.max_past_step:
            self.vehicle_velocities.pop(0)

        # 路径规划器更新
        self.waypoints, _, _, _, _, self.vehicle_front, = self.routeplanner.run_step()

        # 步数更新
        self.time_step += 1
        self.total_step += 1

        return (self._get_obs(), self._get_reward(), self._terminal(), self._get_info())

    # ======================== 【通用逻辑】信息获取 ========================
    def _get_info(self):
        info = {
            'waypoints': self.waypoints,
            'route_waypoints': self.route_waypoints,
            'vehicle_front': self.vehicle_front,
            'cost': self._get_cost()
        }
        info.update(self.scenario_manager.background_scenario.update_info())
        return info

    # ======================== 【通用逻辑】辅助方法（无需修改） ========================
    def _init_traffic_light(self):
        actor_list = self.world.get_actors()
        for actor in actor_list:
            if isinstance(actor, carla.TrafficLight):
                actor.set_red_time(3)
                actor.set_green_time(3)
                actor.set_yellow_time(1)

    def _create_vehicle_blueprint(self, actor_filter, color=None, number_of_wheels=[4]):
        blueprints = self.world.get_blueprint_library().filter(actor_filter)
        blueprint_library = []
        for nw in number_of_wheels:
            blueprint_library = blueprint_library + [x for x in blueprints if
                                                     int(x.get_attribute('number_of_wheels')) == nw]
        bp = random.choice(blueprint_library)
        if bp.has_attribute('color'):
            if not color:
                color = random.choice(bp.get_attribute('color').recommended_values)
            bp.set_attribute('color', color)
        return bp

    def _get_actor_polygons(self, filt):
        actor_poly_dict = {}
        for actor in self.world.get_actors().filter(filt):
            trans = actor.get_transform()
            x = trans.location.x
            y = trans.location.y
            yaw = trans.rotation.yaw / 180 * np.pi
            bb = actor.bounding_box
            l = bb.extent.x
            w = bb.extent.y
            poly_local = np.array([[l, w], [l, -w], [-l, -w], [-l, w]]).transpose()
            R = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
            poly = np.matmul(R, poly_local).transpose() + np.repeat([[x, y]], 4, axis=0)
            actor_poly_dict[actor.id] = poly
        return actor_poly_dict

    def _get_actor_info(self, filt):
        actor_trajectory_dict = {}
        actor_acceleration_dict = {}
        actor_angular_velocity_dict = {}
        actor_velocity_dict = {}

        for actor in self.world.get_actors().filter(filt):
            actor_trajectory_dict[actor.id] = actor.get_transform()
            actor_acceleration_dict[actor.id] = actor.get_acceleration()
            actor_angular_velocity_dict[actor.id] = actor.get_angular_velocity()
            actor_velocity_dict[actor.id] = actor.get_velocity()
        return actor_trajectory_dict, actor_acceleration_dict, actor_angular_velocity_dict, actor_velocity_dict

    # ======================== 【核心逻辑】观测获取（需定制） ========================
    def _get_obs(self):
        # 基础状态计算（横向偏移、航向角偏差、速度、前方车辆距离）
        ego_trans = self.ego_vehicle.get_transform()
        ego_x = ego_trans.location.x
        ego_y = ego_trans.location.y
        ego_yaw = ego_trans.rotation.yaw / 180 * np.pi
        lateral_dis, w = get_preview_lane_dis(self.waypoints, ego_x, ego_y)
        yaw = np.array([np.cos(ego_yaw), np.sin(ego_yaw)])
        delta_yaw = np.arcsin(np.cross(w, yaw))

        v = self.ego_vehicle.get_velocity()
        speed = np.sqrt(v.x ** 2 + v.y ** 2)
        state = np.array([lateral_dis, -delta_yaw, speed, self.vehicle_front])

        # 规划类场景（含 LiDAR/鸟瞰图）
        if self.scenario_category != 'perception':
            # 鸟瞰图渲染
            self.birdeye_render.set_hero(self.ego_vehicle, self.ego_vehicle.id)
            self.birdeye_render.vehicle_polygons = self.vehicle_polygons
            self.birdeye_render.walker_polygons = self.walker_polygons
            self.birdeye_render.waypoints = self.waypoints

            birdeye_render_types = ['roadmap', 'actors', 'waypoints']
            birdeye_surface = self.birdeye_render.render(birdeye_render_types)
            birdeye_surface = pygame.surfarray.array3d(birdeye_surface)
            center = (int(birdeye_surface.shape[0] / 2), int(birdeye_surface.shape[1] / 2))
            width = height = int(self.display_size / 2)
            birdeye = birdeye_surface[center[0] - width:center[0] + width, center[1] - height:center[1] + height]
            birdeye = display_to_rgb(birdeye, self.obs_size)

            # LiDAR 数据处理
            lidar = None
            if not self.disable_lidar:
                point_cloud = np.copy(np.frombuffer(self.lidar_data.raw_data, dtype=np.dtype('f4')))
                point_cloud = np.reshape(point_cloud, (int(point_cloud.shape[0] / 4), 4))
                x = point_cloud[:, 0:1]
                y = point_cloud[:, 1:2]
                z = point_cloud[:, 2:3]
                intensity = point_cloud[:, 3:4]
                point_cloud = np.concatenate([y, -x, z], axis=1)

                y_bins = np.arange(-(self.obs_range - self.d_behind), self.d_behind + self.lidar_bin, self.lidar_bin)
                x_bins = np.arange(-self.obs_range / 2, self.obs_range / 2 + self.lidar_bin, self.lidar_bin)
                z_bins = [-self.lidar_height - 1, -self.lidar_height + 0.25, 1]

                lidar, _ = np.histogramdd(point_cloud, bins=(x_bins, y_bins, z_bins))
                lidar[:, :, 0] = np.array(lidar[:, :, 0] > 0, dtype=np.uint8)
                lidar[:, :, 1] = np.array(lidar[:, :, 1] > 0, dtype=np.uint8)
                wayptimg = birdeye[:, :, 0] < 0
                wayptimg = np.expand_dims(wayptimg, axis=2)
                wayptimg = np.fliplr(np.rot90(wayptimg, 3))
                lidar = np.concatenate((lidar, wayptimg), axis=2)
                lidar = np.flip(lidar, axis=1)
                lidar = np.rot90(lidar, 1) * 255

            # 相机图像缩放（可视化用）
            camera = resize(self.camera_img, (self.obs_size, self.obs_size)) * 255

            # ======================== 【模型专属逻辑】构建观测字典 ========================
            obs = {
                'camera': camera.astype(np.uint8),
                'lidar': None if self.disable_lidar else lidar.astype(np.uint8),
                'birdeye': birdeye.astype(np.uint8),
                'state': state.astype(np.float32),
            }
            # 示例：TCP 添加原始尺寸相机图像
            if 'camera_raw' in MODEL_OBS_KEYS:
                obs['camera_raw'] = self.camera_img_raw.astype(np.uint8)
            # 可添加其他模型专属观测（如 bev_feature）

            # 可视化渲染
            if not self.disable_lidar:
                birdeye_surface = rgb_to_display_surface(birdeye, self.display_size)
                self.display.blit(birdeye_surface, (0, self.env_id * self.display_size))
                lidar_surface = rgb_to_display_surface(lidar, self.display_size)
                self.display.blit(lidar_surface, (self.display_size, self.env_id * self.display_size))
                camera_surface = rgb_to_display_surface(camera, self.display_size)
                self.display.blit(camera_surface, (self.display_size * 2, self.env_id * self.display_size))
            else:
                birdeye_surface = rgb_to_display_surface(birdeye, self.display_size)
                self.display.blit(birdeye_surface, (0, self.env_id * self.display_size))
                camera_surface = rgb_to_display_surface(camera, self.display_size)
                self.display.blit(camera_surface, (self.display_size, self.env_id * self.display_size))

        # 感知类场景（仅相机）
        else:
            camera = resize(self.camera_img, (self.obs_size, self.obs_size)) * 255
            camera_surface = rgb_to_display_surface(camera, self.display_size)
            self.display.blit(camera_surface, (0, self.env_id * self.display_size))

            obs = {
                'camera': camera.astype(np.uint8),
                'state': state.astype(np.float32),
            }
            # ======================== 【模型专属逻辑】添加感知类观测 ========================
            if 'camera_raw' in MODEL_OBS_KEYS:
                obs['camera_raw'] = self.camera_img_raw.astype(np.uint8)

        return obs

    # ======================== 【核心逻辑】奖励函数（需定制） ========================
    def _get_reward(self):
        """
        【模型专属逻辑】修改奖励函数以适配训练目标
        基础奖励项：碰撞惩罚、转向平滑惩罚、越线惩罚、速度跟踪奖励
        """
        # 基础奖励项（可调整权重）
        r_collision = -1 if len(self.collision_hist) > 0 else 0
        r_steer = -self.ego_vehicle.get_control().steer ** 2

        ego_x, ego_y = get_pos(self.ego_vehicle)
        dis, w = get_lane_dis(self.waypoints, ego_x, ego_y)
        r_out = -1 if abs(dis) > self.out_lane_thres else 0

        v = self.ego_vehicle.get_velocity()
        lspeed = np.array([v.x, v.y])
        lspeed_lon = np.dot(lspeed, w)
        r_fast = -1 if lspeed_lon > self.desired_speed / 3.6 else 0
        r_lat = -abs(self.ego_vehicle.get_control().steer) * lspeed_lon ** 2

        # 奖励加权求和（可调整权重）
        r = 1 * r_collision + 1 * lspeed_lon + 10 * r_fast + 1 * r_out + r_steer * 5 + 0.2 * r_lat
        return r

    # ======================== 【通用逻辑】代价计算 ========================
    def _get_cost(self):
        r_collision = 0
        if len(self.collision_hist) > 0:
            r_collision = -1
        return r_collision

    # ======================== 【通用逻辑】终止条件 ========================
    def _terminal(self):
        return not self.scenario_manager._running

        # ======================== 【通用逻辑】资源清理 ========================

    def _remove_sensor(self):
        if self.collision_sensor is not None:
            self.collision_sensor.stop()
            self.collision_sensor.destroy()
            self.collision_sensor = None
        if self.lidar_sensor is not None:
            self.lidar_sensor.stop()
            self.lidar_sensor.destroy()
            self.lidar_sensor = None
        if self.camera_sensor is not None:
            self.camera_sensor.stop()
            self.camera_sensor.destroy()
            self.camera_sensor = None

    def _remove_ego(self):
        if self.ego_vehicle is not None:
            self.ego_vehicle.destroy()
            self.ego_vehicle = None

    def clean_up(self):
        self._remove_sensor()
        self._remove_ego()
        self.scenario_manager.clean_up()