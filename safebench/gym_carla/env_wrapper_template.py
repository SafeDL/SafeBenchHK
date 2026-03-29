"""
通用端到端自动驾驶模型 - SafeBench环境包装器模板
使用说明:
1. 修改 【模型专属配置区】 的内容，适配你的模型
2. 确保模型所需的观测数据在 _preprocess_obs 中正确返回
3. 替换 CarlaEnvXXX 为你的模型专用环境类
"""
import gym
import numpy as np
import pygame

# ======================== 【模型专属配置区 - 开始】 ========================
# 1. 模型专用环境ID（自定义，避免与其他模型冲突）
MODEL_ENV_ID = "carla-yourmodel-v0"
# 2. 模型专用观测类型标识（建议从 10 开始，避免与原有 0-4 冲突）
MODEL_OBS_TYPE = 10
# 3. 导入你的模型专用环境类
# from safebench.gym_carla.envs.carla_env_yourmodel import CarlaEnvYourModel
# ======================== 【模型专属配置区 - 结束】 ========================


class VectorWrapperModel(gym.Wrapper):
    """
    通用多场景环境包装器
    作用：管理多个并行仿真场景，提供标准化的 reset/step 接口
    """
    def __init__(self, env_params, scenario_config, world, birdeye_render, display, logger):
        self.logger = logger
        self.world = world
        self.num_scenario = scenario_config['num_scenario']
        self.ROOT_DIR = scenario_config['ROOT_DIR']
        self.frame_skip = scenario_config['frame_skip']
        self.render = scenario_config['render']

        # 环境列表 + 动作空间列表
        self.env_list = []
        self.action_space_list = []
        for _ in range(self.num_scenario):
            # ---------------- 【模型专属】创建模型专用环境 ----------------
            env = carla_env_model(
                env_params, birdeye_render=birdeye_render,
                display=display, world=world, logger=logger
            )
            self.env_list.append(env)
            self.action_space_list.append(env.action_space)

        # 场景状态标记
        self.finished_env = [False] * self.num_scenario

    # ======================== 【通用逻辑】无需修改 ========================
    def obs_postprocess(self, obs_list):
        """
        观测后处理：根据模型需求选择返回格式
        通用规则：
        - 模型需要字典格式观测 → 返回 list（如 TCP）
        - 模型需要数组格式观测 → 返回 np.array
        """
        # ---------------- 【模型专属】修改观测后处理逻辑 ----------------
        # 示例1：TCP 用 → return obs_list
        # 示例2：数组格式用 → return np.array(obs_list)
        return obs_list

    def get_ego_vehicles(self):
        ego_vehicles = []
        for env in self.env_list:
            if env.ego_vehicle is not None:
                ego_vehicles.append(env.ego_vehicle)
        return ego_vehicles

    def get_static_obs(self, scenario_configs):
        static_obs_list = []
        for s_i in range(len(scenario_configs)):
            static_obs = self.env_list[s_i].get_static_obs(scenario_configs[s_i])
            static_obs_list.append(static_obs)
        return static_obs_list

    def reset(self, scenario_configs, scenario_init_action):
        """重置所有场景，创建自车和障碍物"""
        obs_list = []
        info_list = []
        for s_i in range(len(scenario_configs)):
            config = scenario_configs[s_i]
            self.world.set_weather(config.weather)
            obs, info = self.env_list[s_i].reset(
                config=config, env_id=s_i, scenario_init_action=scenario_init_action[s_i]
            )
            obs_list.append(obs)
            info_list.append(info)
            info_list[s_i].update({'data_id': config.data_id, 's_id': s_i})

        self.finished_env = [False] * self.num_scenario
        for s_i in range(len(scenario_configs), self.num_scenario):
            self.finished_env[s_i] = True

        return self.obs_postprocess(obs_list), info_list

    def step(self, ego_actions, scenario_actions, scenario_configs):
        """执行动作，推进仿真，收集数据"""
        # 1. 分发动作到未完成的场景
        action_idx = 0
        for e_i in range(self.num_scenario):
            if not self.finished_env[e_i]:
                processed_action = self.env_list[e_i]._postprocess_action(ego_actions[action_idx])
                self.env_list[e_i].step_before_tick(processed_action, scenario_actions[action_idx])
                action_idx += 1

        # 2. 推进CARLA仿真
        for _ in range(self.frame_skip):
            self.world.tick()

        # 3. 收集场景数据
        obs_list, reward_list, done_list, info_list = [], [], [], []
        for e_i in range(self.num_scenario):
            if not self.finished_env[e_i]:
                current_env = self.env_list[e_i]
                obs, reward, done, info = current_env.step_after_tick()
                info['data_id'] = scenario_configs[e_i].data_id

                # 标记场景结束并保存结果
                if done:
                    self.finished_env[e_i] = True
                    if current_env.config.data_id in current_env.running_results.keys():
                        self.logger.log(f"Scenario {current_env.config.data_id} duplicated")
                    current_env.running_results[current_env.config.data_id] = current_env.scenario_manager.running_record

                obs_list.append(obs)
                reward_list.append(reward)
                done_list.append(done)
                info_list.append(info)

        # 4. 格式转换 + 渲染更新
        rewards = np.array(reward_list)
        dones = np.array(done_list)
        infos = np.array(info_list)
        if self.render:
            pygame.display.flip()

        return self.obs_postprocess(obs_list), rewards, dones, infos

    def all_scenario_done(self):
        return np.sum(self.finished_env) == self.num_scenario

    def clean_up(self):
        """清理所有场景的资源"""
        for e_i in range(self.num_scenario):
            self.env_list[e_i].clean_up()
        self.world.tick()


class ObservationWrapperModel(gym.Wrapper):
    """
    通用观测包装器
    作用：
    1. 适配模型所需的观测格式（图像/状态/多模态）
    2. 标准化动作空间
    3. 可选：奖励预处理
    """
    def __init__(self, env, obs_type):
        super().__init__(env)
        self._env = env
        self.is_running = False
        self.obs_type = obs_type
        self._build_obs_space()

        # ---------------- 【模型专属】定义动作空间 ----------------
        # 示例：2维动作（油门/刹车，转向），范围 [-1, 1]
        act_dim = 2
        act_lim = np.ones((act_dim), dtype=np.float32)
        self.action_space = gym.spaces.Box(-act_lim, act_lim, dtype=np.float32)

    def _build_obs_space(self):
        """构建观测空间：根据 obs_type 定义维度和范围"""
        if self.obs_type == 0:
            # 传统算法：4维状态
            obs_dim = 4
            obs_lim = np.ones((obs_dim), dtype=np.float32)
            self.observation_space = gym.spaces.Box(-obs_lim, obs_lim)
        elif self.obs_type == 1:
            # 规则式算法：11维状态+导航
            obs_dim = 11
            obs_lim = np.ones((obs_dim), dtype=np.float32)
            self.observation_space = gym.spaces.Box(-obs_lim, obs_lim)
        elif self.obs_type == 2 or self.obs_type == 3:
            # 端到端算法：鸟瞰图/前视图 + 状态
            obs_dim = 128
            obs_lim = np.ones((obs_dim), dtype=np.float32)
            self.observation_space = gym.spaces.Box(-obs_lim, obs_lim)
        # ======================== 【模型专属】添加观测类型 ========================
        elif self.obs_type == MODEL_OBS_TYPE:
            # 你的模型：自定义观测维度（如 256 对应图像尺寸）
            obs_dim = 256  # 修改为你的模型输入维度
            obs_lim = np.ones((obs_dim), dtype=np.float32)
            self.observation_space = gym.spaces.Box(-obs_lim, obs_lim)
        else:
            raise NotImplementedError(f"Observation type {self.obs_type} not supported")

    def _preprocess_obs(self, obs):
        """
        核心：将CARLA原生观测转换为模型输入格式
        必须根据你的模型需求修改
        """
        if self.obs_type == 0:
            return obs['state'][:4].astype(np.float64)
        elif self.obs_type == 1:
            return np.array([
                obs['state'][0], obs['state'][1], obs['state'][2], obs['state'][3],
                obs.get('command', 4),
                obs.get('forward_vector', [0, 0])[0], obs.get('forward_vector', [0, 0])[1],
                obs.get('node_forward', [0, 0])[0], obs.get('node_forward', [0, 0])[1],
                obs.get('target_forward', [0, 0])[0], obs.get('target_forward', [0, 0])[1]
            ])
        elif self.obs_type == 2:
            return {"img": obs['birdeye'], "states": obs['state'][:4].astype(np.float64)}
        elif self.obs_type == 3:
            return {"img": obs['camera'], "states": obs['state'][:4].astype(np.float64)}
        # ======================== 【模型专属】观测预处理逻辑 ========================
        elif self.obs_type == MODEL_OBS_TYPE:
            # 示例：返回模型所需的原始图像 + 状态
            # 修改 obs['camera_raw'] 为你的模型所需的图像key
            return {"img": obs['camera_raw'], "states": obs['state'][:4].astype(np.float64)}
        else:
            raise NotImplementedError(f"Observation type {self.obs_type} not supported")

    # ======================== 【通用逻辑】无需修改 ========================
    def get_static_obs(self, config):
        return self._env.get_static_obs(config)

    def reset(self, **kwargs):
        obs, info = self._env.reset(** kwargs)
        return self._preprocess_obs(obs), info

    def step_before_tick(self, ego_action, scenario_action):
        self._env.step_before_tick(ego_action=ego_action, scenario_action=scenario_action)

    def step_after_tick(self):
        obs, reward, done, info = self._env.step_after_tick()
        self.is_running = self._env.is_running
        reward, info = self._preprocess_reward(reward, info)
        obs = self._preprocess_obs(obs)
        return obs, reward, done, info

    def _preprocess_reward(self, reward, info):
        """可选：修改奖励函数，适配你的模型训练"""
        return reward, info

    def _postprocess_action(self, action):
        """可选：动作后处理（如裁剪、映射）"""
        return action

    def clean_up(self):
        self._env.clean_up()


def carla_env_model(env_params, birdeye_render=None, display=None, world=None, logger=None):
    """
    模型专用环境构造函数
    作用：注册并创建模型专属的CARLA环境实例
    """
    # ---------------- 【模型专属】导入并注册环境 ----------------
    # 替换 CarlaEnvYourModel 为你的模型环境类
    # from safebench.gym_carla.envs.carla_env_yourmodel import CarlaEnvYourModel

    # 注册Gym环境（避免重复注册）
    if MODEL_ENV_ID not in gym.envs.registry.env_specs:
        gym.register(
            id=MODEL_ENV_ID,
            entry_point='safebench.gym_carla.envs.carla_env_yourmodel:CarlaEnvYourModel',
        )

    # 创建环境并包装观测器
    return ObservationWrapperModel(
        # CarlaEnvYourModel(
        #     env_params=env_params,
        #     birdeye_render=birdeye_render,
        #     display=display,
        #     world=world,
        #     logger=logger,
        # ),
        obs_type=env_params['obs_type']
    )