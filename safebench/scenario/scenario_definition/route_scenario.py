"""
处理主车生成、构建场景实例、测试评价
"""
import copy
import traceback
import numpy as np
import carla
from safebench.util.run_util import class_from_path
from safebench.scenario.scenario_manager.timer import GameTime
from safebench.scenario.scenario_manager.carla_data_provider import CarlaDataProvider
from safebench.scenario.scenario_manager.scenario_config import RouteScenarioConfig
from safebench.scenario.tools.route_parser import RouteParser
from safebench.carla_agents.navigation.global_route_planner import GlobalRoutePlanner
from safebench.scenario.tools.scenario_utils import (
    convert_json_to_transform,
    convert_json_to_actor, 
    convert_transform_to_location
)

from safebench.scenario.scenario_definition.atomic_criteria import (
    Status,
    CollisionTest,
    DrivenDistanceTest,
    AverageVelocityTest,
    OffRoadTest,
    KeepLaneTest,
    InRouteTest,
    RouteCompletionTest,
    RunningRedLightTest,
    RunningStopTest,
)


class RouteScenario:
    """
        Implementation of a RouteScenario, i.e. a scenario that consists of driving along a pre-defined route,
        along which several smaller scenarios are triggered
    """

    def __init__(self, world, config, ego_id, logger, max_running_step):
        self.world = world
        self.logger = logger
        self.config = config
        self.ego_id = ego_id
        self.max_running_step = max_running_step
        self.timeout = 60  # 默认设置的仿真超时时间
        # 这里的scenario_definition指的是经过route_scenario.py解析之后的具体场景定义,比如要执行的DynamicObjectCrossing名称,触发位置等
        self.route, self.ego_vehicle, scenario_definitions = self._update_route_and_ego(timeout=self.timeout)
        self.background_actors = []
        self.list_scenarios = self._build_scenario_instances(scenario_definitions)  # 根据场景定义来创建控制场景动作的实例
        self.criteria = self._create_criteria()

    def _update_route_and_ego(self, timeout=None):
        # transform the scenario file into a dictionary
        if self.config.scenario_file is not None:
            # scenario_file形如scenario_01.json
            world_annotations = RouteParser.parse_annotations_file(self.config.scenario_file)
        else:
            world_annotations = self.config.scenario_config

        # 使用CARLA的路径搜索功能,获取插值轨迹
        wmap = self.world.get_map()
        global_planner = GlobalRoutePlanner(wmap, sampling_resolution=2.0)
        start_location = self.config.trajectory[0]
        end_location = self.config.trajectory[-1]
        route = global_planner.trace_route(start_location, end_location)

        # 创建一个新的route列表
        updated_route = []
        for i in range(len(route)):
            waypoint = route[i][0]
            road_option = route[i][1]
            new_transform = carla.Transform(
                location=waypoint.transform.location,
                rotation=carla.Rotation(
                    pitch=waypoint.transform.rotation.pitch,
                    yaw=waypoint.transform.rotation.yaw,
                    roll=waypoint.transform.rotation.roll
                )
            )
            updated_route.append((new_transform, road_option))
        route = updated_route

        ego_vehicle = self._spawn_ego_vehicle(self.config.initial_transform, self.config.auto_ego)

        # possible_scenarios是一个字典,键值为数字,值为场景定义的列表,包含‘match_position’,'name', 'trigger_position'等字段
        possible_scenarios, _ = RouteParser.scan_route_for_scenariosHK(
            self.config.town,
            self.config.route_id,  # route_id是一个字符串,表示当前测试场景的道路编号
            world_annotations  # route是一个列表,列表中的每个元素都是一个长度为2的tuple,包含有Transform和RoadOption
        )

        # 将所有可能的场景定义从possible_scenarios字典中提取出来,此时scenarios_definitions存储着场景的名称、触发位置等信息
        scenarios_definitions = []
        for trigger in possible_scenarios.keys():
            scenarios_definitions.extend(possible_scenarios[trigger])

        assert len(scenarios_definitions) >= 1, f"There should be at least 1 scenario definition in the route"
        scenarios_definitions = [scenarios_definitions[0]]

        CarlaDataProvider.set_ego_vehicle_route(convert_transform_to_location(route))
        CarlaDataProvider.set_scenario_config(self.config)

        # Timeout of scenario in seconds
        self.timeout = self._estimate_route_timeout(route) if timeout is None else timeout
        return route, ego_vehicle, scenarios_definitions

    def _estimate_route_timeout(self, route):
        route_length = 0.0  # in meters
        min_length = 1000.0
        SECONDS_GIVEN_PER_METERS = 1

        if len(route) == 1:
            return int(SECONDS_GIVEN_PER_METERS * min_length)

        prev_point = route[0][0]
        for current_point, _ in route[1:]:
            dist = current_point.location.distance(prev_point.location)
            route_length += dist
            prev_point = current_point
        return int(SECONDS_GIVEN_PER_METERS * route_length)

    def _spawn_ego_vehicle(self, elevate_transform, autopilot=False):
        role_name = 'ego_vehicle' + str(self.ego_id)  # self.ego_id的取值与当前执行的是第几个场景有关

        success = False
        ego_vehicle = None
        while not success:
            try: # 这里指定了ego vehicle的类型为'vehicle.lincoln.mkz_2017'
                ego_vehicle = CarlaDataProvider.request_new_actor(
                    'vehicle.lincoln.mkz_2017',
                    elevate_transform,
                    rolename=role_name, 
                    autopilot=autopilot
                )
                ego_vehicle.set_autopilot(autopilot, CarlaDataProvider.get_traffic_manager_port())
                success = True
            except RuntimeError:
                elevate_transform.location.z += 0.1
        return ego_vehicle

    def _build_scenario_instances(self, scenario_definitions):
        """
        构建所有场景类的实例
        """
        scenario_instance_list = []
        for _, definition in enumerate(scenario_definitions):
            # get the class of the scenario
            scenario_path = [
                'safebench.scenario.scenario_definition',  # 场景定义的基础路径
                self.config.scenario_folder,  # 配置中指定的场景文件夹
                definition['name'],  # 场景定义的名称
            ]
            scenario_class = class_from_path('.'.join(scenario_path))

            # 如果场景定义中有其他参与者,则创建这些参与者的实例
            if definition['other_actors'] is not None:
                list_of_actor_conf_instances = self._get_actors_instances(definition['other_actors'])
            else:
                list_of_actor_conf_instances = []

            # 创建场景运行的触发位置
            egoactor_trigger_position = convert_json_to_transform(definition['trigger_position'])
            route_config = RouteScenarioConfig()
            route_config.other_actors = list_of_actor_conf_instances
            route_config.trigger_points = [egoactor_trigger_position]
            route_config.parameters = self.config.parameters
            route_config.num_scenario = self.config.num_scenario
            if self.config.weather is not None:
                route_config.weather = self.config.weather

            try:
                # 实例化场景类
                scenario_instance = scenario_class(self.world, self.ego_vehicle, route_config, timeout=self.timeout)
            except Exception as e:   
                traceback.print_exc()
                print("Skipping scenario '{}' due to setup error: {}".format(definition['name'], e))
                continue

            scenario_instance_list.append(scenario_instance)
        return scenario_instance_list

    def _get_actors_instances(self, list_of_antagonist_actors):
        def get_actors_from_list(list_of_actor_def):
            # receives a list of actor definitions and creates an actual list of ActorConfigurationObjects
            sublist_of_actors = []
            for actor_def in list_of_actor_def:
                sublist_of_actors.append(convert_json_to_actor(actor_def))
            return sublist_of_actors

        list_of_actors = []
        if 'left' in list_of_antagonist_actors:
            list_of_actors += get_actors_from_list(list_of_antagonist_actors['left'])
        if 'right' in list_of_antagonist_actors:
            list_of_actors += get_actors_from_list(list_of_antagonist_actors['right'])
        if 'center' in list_of_antagonist_actors:
            list_of_actors += get_actors_from_list(list_of_antagonist_actors['center'])
        return list_of_actors

    def initialize_actors(self):
        amount = 10  # 生成背景车流,指定生成的数量
        new_actors = CarlaDataProvider.request_new_batch_actors(
            'vehicle.*',
            self.ego_vehicle,
            amount,
            autopilot=True,
            random_location=False,
            rolename='background'
        )

        if new_actors is None:
            raise Exception("Error: Unable to add the background activity, all spawn points were occupied")
        for _actor in new_actors:
            self.background_actors.append(_actor)

    def get_running_status(self, running_record):
        running_status = {
            'ego_velocity': CarlaDataProvider.get_velocity(self.ego_vehicle),
            'ego_acceleration_x': self.ego_vehicle.get_acceleration().x,
            'ego_acceleration_y': self.ego_vehicle.get_acceleration().y,
            'ego_acceleration_z': self.ego_vehicle.get_acceleration().z,
            'ego_x': CarlaDataProvider.get_transform(self.ego_vehicle).location.x,
            'ego_y': CarlaDataProvider.get_transform(self.ego_vehicle).location.y,
            'ego_z': CarlaDataProvider.get_transform(self.ego_vehicle).location.z,
            'ego_roll': CarlaDataProvider.get_transform(self.ego_vehicle).rotation.roll,
            'ego_pitch': CarlaDataProvider.get_transform(self.ego_vehicle).rotation.pitch,
            'ego_yaw': CarlaDataProvider.get_transform(self.ego_vehicle).rotation.yaw,
            'current_game_time': GameTime.get_time()
        }
        # running_status首先获取其主车状态,其次获得各个预定义的atomic criteria的状态
        for criterion_name, criterion in self.criteria.items():
            running_status[criterion_name] = criterion.update()

        stop = False
        # collision with other objects
        if running_status['collision'] == Status.FAILURE:
            stop = True
            self.logger.log('>> Scenario stops due to collision', color='yellow')

        # out of the road detection
        if running_status['off_road'] == Status.FAILURE:
            stop = True
            self.logger.log('>> Scenario stops due to off road', color='yellow')

        # only check when evaluating
        if self.config.scenario_id != 0:  
            # route completed
            if running_status['route_complete'] == 100:
                stop = True
                self.logger.log('>> Scenario stops due to route completion', color='yellow')

        # stop at max step
        if len(running_record) >= self.max_running_step: 
            stop = True
            self.logger.log('>> Scenario stops due to max steps', color='yellow')

        for scenario in self.list_scenarios:
            # only check when evaluating
            if self.config.scenario_id != 0:  
                if running_status['driven_distance'] >= scenario.ego_max_driven_distance:
                    stop = True
                    self.logger.log('>> Scenario stops due to max driven distance', color='yellow')
                    break
            if running_status['current_game_time'] >= scenario.timeout:
                stop = True
                self.logger.log('>> Scenario stops due to timeout', color='yellow') 
                break

        return running_status, stop

    def _create_criteria(self):
        criteria = {}
        route = convert_transform_to_location(self.route)

        criteria['driven_distance'] = DrivenDistanceTest(actor=self.ego_vehicle, distance_success=1e4, distance_acceptable=1e4, optional=True)
        criteria['average_velocity'] = AverageVelocityTest(actor=self.ego_vehicle, avg_velocity_success=1e4, avg_velocity_acceptable=1e4, optional=True)
        criteria['lane_invasion'] = KeepLaneTest(actor=self.ego_vehicle, optional=True)
        criteria['off_road'] = OffRoadTest(actor=self.ego_vehicle, optional=True)
        criteria['collision'] = CollisionTest(actor=self.ego_vehicle, other_actor_type=['vehicle', 'walker', 'static'], terminate_on_failure=True)
        criteria['run_red_light'] = RunningRedLightTest(actor=self.ego_vehicle)
        criteria['run_stop'] = RunningStopTest(actor=self.ego_vehicle)
        if self.config.scenario_id != 0:  # only check when evaluating
            criteria['distance_to_route'] = InRouteTest(self.ego_vehicle, route=route, offroad_max=30)
            criteria['route_complete'] = RouteCompletionTest(self.ego_vehicle, route=route)
        return criteria

    @staticmethod
    def _get_actor_state(actor):
        actor_trans = actor.get_transform()
        actor_x = actor_trans.location.x
        actor_y = actor_trans.location.y
        actor_yaw = actor_trans.rotation.yaw / 180 * np.pi
        yaw = np.array([np.cos(actor_yaw), np.sin(actor_yaw)])
        velocity = actor.get_velocity()
        acc = actor.get_acceleration()
        return [actor_x, actor_y, actor_yaw, yaw[0], yaw[1], velocity.x, velocity.y, acc.x, acc.y]

    def update_info(self):
        ego_state = self._get_actor_state(self.ego_vehicle)
        actor_info = [ego_state]
        for s_i in self.list_scenarios:
            for a_i in s_i.other_actors:
                actor_state = self._get_actor_state(a_i)
                actor_info.append(actor_state)

        actor_info = np.array(actor_info)
        # get the info of the ego vehicle and the other actors
        return {
            'actor_info': actor_info
        }

    def clean_up(self):
        # stop criterion and destroy sensors
        for _, criterion in self.criteria.items():
            criterion.terminate()

        # each scenario remove its own actors
        for scenario in self.list_scenarios:
            scenario.clean_up()

        # remove background vehicles
        for s_i in range(len(self.background_actors)):
            if self.background_actors[s_i].type_id.startswith('vehicle'):
                self.background_actors[s_i].set_autopilot(enabled=False)
            if CarlaDataProvider.actor_id_exists(self.background_actors[s_i].id):
                CarlaDataProvider.remove_actor_by_id(self.background_actors[s_i].id)
        self.background_actors = []
