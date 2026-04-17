"""
场景运行器的类
场景可视化、场景数据加载和采样、场景测试环境
"""
import copy
import numpy as np
import carla
import pygame
from tqdm import tqdm
import time
from safebench.gym_carla.env_wrapper import VectorWrapper
from safebench.gym_carla.env_wrapper_tcp import VectorWrapperTCP
from safebench.gym_carla.envs.render import BirdeyeRender
from safebench.gym_carla.replay_buffer import RouteReplayBuffer, PerceptionReplayBuffer
from safebench.agent import AGENT_POLICY_LIST
from safebench.scenario import SCENARIO_POLICY_LIST
from safebench.scenario.scenario_manager.carla_data_provider import CarlaDataProvider
from safebench.scenario.scenario_data_loader import ScenarioDataLoader
from safebench.scenario.tools.scenario_utils import scenario_parse
from safebench.util.logger import Logger, setup_logger_kwargs
from safebench.util.metric_util import get_route_scores, get_perception_scores


class CarlaRunner:
    def __init__(self, agent_config, scenario_config):
        self.scenario_config = scenario_config
        self.agent_config = agent_config

        self.seed = scenario_config['seed']
        self.exp_name = scenario_config['exp_name']
        self.output_dir = scenario_config['output_dir']
        self.mode = scenario_config['mode']
        self.save_video = scenario_config['save_video']

        self.render = scenario_config['render']
        self.num_scenario = scenario_config['num_scenario']
        self.fixed_delta_seconds = scenario_config['fixed_delta_seconds']
        self.scenario_category = scenario_config['scenario_category']

        # continue training flag
        self.continue_agent_training = scenario_config['continue_agent_training']
        self.continue_scenario_training = scenario_config['continue_scenario_training']

        # apply settings to carla
        self.client = carla.Client('localhost', scenario_config['port'])
        self.client.set_timeout(60.0)
        self.world = None
        self.env = None

        # env_params 主要用于渲染的BirdEyeView展示
        self.env_params = {
            'auto_ego': scenario_config['auto_ego'],
            'obs_type': agent_config['obs_type'],
            'scenario_category': self.scenario_category,               # planning or perception
            'ROOT_DIR': scenario_config['ROOT_DIR'],
            'warm_up_steps': 9,                                        # number of ticks after spawning the vehicles
            'disable_lidar': True,                                     # show bird-eye view lidar or not
            'display_size': 256,                                       # screen size of one bird-eye view window
            'obs_range': 64,                                           # observation range (meter)
            'd_behind': 12,                                            # distance behind the ego vehicle (meter)
            'max_past_step': 1,                                        # the number of past steps to draw
            'discrete': False,                                         # whether to use discrete control space
            'discrete_acc': [-3.0, 0.0, 3.0],                          # discrete value of accelerations
            'discrete_steer': [-0.2, 0.0, 0.2],                        # discrete value of steering angles
            'continuous_accel_range': [-3.0, 3.0],                     # continuous acceleration range
            'continuous_steer_range': [-0.3, 0.3],                     # continuous steering angle range
            'max_episode_step': scenario_config['max_episode_step'],   # maximum timesteps per episode
            'max_waypt': 10,                                           # maximum number of waypoints
            'lidar_bin': 0.125,                                        # bin size of lidar sensor (meter)
            'out_lane_thres': 4,                                       # threshold for out of lane (meter)
            'desired_speed': 25,                                       # desired speed (km/h)
            'image_sz': 1024,                                          # TODO: move to config of od scenario
        }

        # pass config from scenario to agent
        agent_config['mode'] = scenario_config['mode']
        agent_config['ego_action_dim'] = scenario_config['ego_action_dim']
        agent_config['ego_state_dim'] = scenario_config['ego_state_dim']
        agent_config['ego_action_limit'] = scenario_config['ego_action_limit']

        # 定义日志记录器
        logger_kwargs = setup_logger_kwargs(
            self.exp_name, 
            self.output_dir, 
            self.seed,
            agent=agent_config['policy_type'],
            scenario=scenario_config['policy_type'],
            scenario_category=self.scenario_category
        )
        self.logger = Logger(**logger_kwargs)  # 对Kwargs进行解包成字典
        
        # prepare parameters
        if self.mode == 'train_agent':
            self.buffer_capacity = agent_config['buffer_capacity']
            self.eval_in_train_freq = agent_config['eval_in_train_freq']
            self.save_freq = agent_config['save_freq']
            self.train_episode = agent_config['train_episode']
            self.logger.save_config(agent_config)
            self.logger.create_training_dir()
        elif self.mode == 'train_scenario':
            self.buffer_capacity = scenario_config['buffer_capacity']
            self.eval_in_train_freq = scenario_config['eval_in_train_freq']
            self.save_freq = scenario_config['save_freq']
            self.train_episode = scenario_config['train_episode']
            self.logger.save_config(scenario_config)
            self.logger.create_training_dir()
        elif self.mode == 'eval':
            self.save_freq = scenario_config['save_freq']  # 指定数据的保存频率
            self.logger.log('>> Evaluation Mode, skip config saving', 'yellow')
            self.logger.create_eval_dir(load_existing_results=True)
        else:
            raise NotImplementedError(f"Unsupported mode: {self.mode}.")

        # define agent and scenario
        self.logger.log('>> Agent Policy: ' + agent_config['policy_type'])
        self.logger.log('>> Scenario Policy: ' + scenario_config['policy_type'])

        if self.scenario_config['auto_ego']:
            self.logger.log('>> Using auto-polit for ego vehicle, action of policy will be ignored', 'yellow')
        if scenario_config['policy_type'] == 'ordinary' and self.mode != 'train_agent':
            self.logger.log('>> Ordinary scenario can only be used in agent training', 'red')
            raise Exception()
        self.logger.log('>> ' + '-' * 40)

        # define agent and scenario policy
        self.agent_policy = AGENT_POLICY_LIST[agent_config['policy_type']](agent_config, logger=self.logger)
        self.scenario_policy = SCENARIO_POLICY_LIST[scenario_config['policy_type']](scenario_config, logger=self.logger)
        if self.save_video:
            assert self.mode == 'eval', "only allow video saving in eval mode"
            self.logger.init_video_recorder()

    def _init_world(self, town):
        self.logger.log(f">> Initializing carla world: {town}")
        self.world = self.client.load_world(town)
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = self.fixed_delta_seconds
        self.world.apply_settings(settings)
        CarlaDataProvider.set_client(self.client)
        CarlaDataProvider.set_world(self.world)
        CarlaDataProvider.set_traffic_manager_port(self.scenario_config['tm_port'])
        # 初始化的时候设置天气为晴天,这里可以根据需求配置为设定的其他天气种类
        self.world.set_weather(carla.WeatherParameters.ClearNoon)


    def _init_renderer(self):
        self.logger.log(">> Initializing pygame birdeye renderer")
        pygame.init()
        # 设置'pygame'显示窗口的标志位，HWSURFACE表示使用硬件加速，DOUBLEBUF表示双缓冲
        flag = pygame.HWSURFACE | pygame.DOUBLEBUF
        if not self.render:
            flag = flag | pygame.HIDDEN
        # 规划测试的显示窗口设置
        is_tcp = self.agent_config.get('agent_cfg', [''])[0] == 'tcp.yaml'
        if self.scenario_category == 'planning': 
            if is_tcp:
                # TCP相机 900x256, 按比例缩放后宽度 = display_size * 900/256
                tcp_cam_display_w = int(self.env_params['display_size'] * 900 / 256)
                if self.env_params['disable_lidar']:
                    window_size = (self.env_params['display_size'] + tcp_cam_display_w, self.env_params['display_size'] * self.num_scenario)
                else:
                    window_size = (self.env_params['display_size'] * 2 + tcp_cam_display_w, self.env_params['display_size'] * self.num_scenario)
            else:
                # [bird-eye view, Lidar, front view] or [bird-eye view, front view]
                if self.env_params['disable_lidar']:
                    window_size = (self.env_params['display_size'] * 2, self.env_params['display_size'] * self.num_scenario)
                else:
                    window_size = (self.env_params['display_size'] * 3, self.env_params['display_size'] * self.num_scenario)
        # 感知测试的显示窗口设置
        else:
            window_size = (self.env_params['display_size'], self.env_params['display_size'] * self.num_scenario)
        # flag设置窗口属性
        self.display = pygame.display.set_mode(window_size, flag)

        # 计算观察范围内每米对应的像素数
        pixels_per_meter = self.env_params['display_size'] / self.env_params['obs_range']
        pixels_ahead_vehicle = (self.env_params['obs_range'] / 2 - self.env_params['d_behind']) * pixels_per_meter
        self.birdeye_params = {
            'screen_size': [self.env_params['display_size'], self.env_params['display_size']],
            'pixels_per_meter': pixels_per_meter,
            'pixels_ahead_vehicle': pixels_ahead_vehicle,
        }
        self.birdeye_render = BirdeyeRender(self.world, self.birdeye_params, logger=self.logger)

    def train(self, data_loader, start_episode=0):
        # general buffer for both agent and scenario
        Buffer = RouteReplayBuffer if self.scenario_category == 'planning' else PerceptionReplayBuffer
        replay_buffer = Buffer(len(data_loader), self.mode, self.buffer_capacity)

        for e_i in tqdm(range(start_episode, self.train_episode)):  # e_i是episode的索引
            # reset the index counter to create endless loader
            data_loader.reset_idx_counter()
            # reset replay buffer
            replay_buffer.reset_init_buffer()
            replay_buffer.reset_buffer()
            # 我们假设所有场景播放过一遍为一个episode
            while len(data_loader) > 0:
                # sample scenarios
                sampled_scenario_configs, _ = data_loader.sampler()
                # get static obs and then reset with init action
                static_obs = self.env.get_static_obs(sampled_scenario_configs)
                scenario_init_action, additional_dict = self.scenario_policy.get_init_action(static_obs)
                obs, infos = self.env.reset(sampled_scenario_configs, scenario_init_action)
                replay_buffer.store_init([static_obs, scenario_init_action], additional_dict=additional_dict)

                # 从这行代码进入,实例化ego vehicle的控制策略
                self.agent_policy.set_ego_and_route(self.env.get_ego_vehicles(), infos, static_obs=static_obs)

                while not self.env.all_scenario_done():
                    # get action from agent policy and scenario policy (assume using one batch)
                    ego_actions = self.agent_policy.get_action(obs, infos, deterministic=False) # [油门(throttle), 转向(steer)]
                    scenario_actions = self.scenario_policy.get_action(obs, infos, deterministic=False)

                    # apply action to env and get obs
                    next_obs, rewards, dones, infos = self.env.step(ego_actions=ego_actions, scenario_actions=scenario_actions, scenario_configs=sampled_scenario_configs)
                    replay_buffer.store([ego_actions, scenario_actions, obs, next_obs, rewards, dones], additional_dict=infos)
                    obs = copy.deepcopy(next_obs)

                    # train off-policy agent or scenario
                    if self.mode == 'train_agent' and self.agent_policy.type == 'offpolicy':
                        self.agent_policy.train(replay_buffer)
                    elif self.mode == 'train_scenario' and self.scenario_policy.type == 'offpolicy':
                        self.scenario_policy.train(replay_buffer)
                # end up environment
                self.env.clean_up()

            # finish one episode
            replay_buffer.finish_one_episode()
            self.logger.add_training_results('episode', e_i)
            print(f'Episode {e_i} Reward: {np.sum(replay_buffer.buffer_episode_reward)}')
            self.logger.add_training_results('episode_reward', np.sum(replay_buffer.buffer_episode_reward))
            self.logger.save_training_results()

            # train on-policy agent or scenario
            if self.mode == 'train_agent' and self.agent_policy.type == 'onpolicy':
                self.agent_policy.train(replay_buffer)
            elif self.mode == 'train_scenario' and self.scenario_policy.type in ['init_state', 'onpolicy']:
                self.scenario_policy.train(replay_buffer)

            # eval during training
            if (e_i+1) % self.eval_in_train_freq == 0:
                #self.eval(env, data_loader)
                pass

            # save checkpoints
            if (e_i+1) % self.save_freq == 0:
                if self.mode == 'train_agent':
                    self.agent_policy.save_model(e_i)
                if self.mode == 'train_scenario':
                    self.scenario_policy.save_model(e_i)

    def eval(self, data_loader):
        num_finished_scenario = 0
        data_loader.reset_idx_counter()
        while len(data_loader) > 0:
            # sample_scenario_configs被选中的将要测试的场景config文件, num_sampled_scenario为同一个批次采样的场景数量
            sampled_scenario_configs, num_sampled_scenario = data_loader.sampler()
            num_finished_scenario += num_sampled_scenario

            # reset envs with new config, get init action from scenario policy, and run scenario
            static_obs = self.env.get_static_obs(sampled_scenario_configs)  # 获取行驶轨迹和设置的目标测试速度
            self.scenario_policy.load_model(sampled_scenario_configs)
            scenario_init_action, _ = self.scenario_policy.get_init_action(static_obs, deterministic=True)
            obs, infos = self.env.reset(sampled_scenario_configs, scenario_init_action)

            # 进入该函数初始化本车的agent
            self.agent_policy.set_ego_and_route(self.env.get_ego_vehicles(), infos, static_obs=static_obs)

            # 根据scenario的data_id进行记录，否则下面计算score的时候会越界
            score_list = {sampled_scenario_configs[scenario_i].data_id: [] for scenario_i in range(num_sampled_scenario)}
            while not self.env.all_scenario_done():
                # get action from agent policy and scenario policy (assume using one batch)
                ego_actions = self.agent_policy.get_action(obs, infos, deterministic=True)
                scenario_actions = self.scenario_policy.get_action(obs, infos, deterministic=True)

                # 计算强化学习分数使用车辆正确yaw，解决reward负值问题
                # apply action to env and get obs
                obs, rewards, _, infos = self.env.step(ego_actions=ego_actions, scenario_actions=scenario_actions, scenario_configs=sampled_scenario_configs) # 应用上面的ego_action对车辆进行控制
                # save video
                if self.save_video:
                    if self.scenario_category == 'planning':
                        self.logger.add_frame(pygame.surfarray.array3d(self.display).transpose(1, 0, 2))
                    else:
                        self.logger.add_frame({s_i['scenario_id']: ego_actions[n_i]['annotated_image'] for n_i, s_i in enumerate(infos)})

                # accumulate scores of corresponding scenario
                reward_idx = 0
                for s_i in infos:
                    score = rewards[reward_idx] if self.scenario_category == 'planning' else 1-infos[reward_idx]['iou_loss']
                    score_list[s_i['data_id']].append(score)
                    reward_idx += 1

            # clean up all things
            self.logger.log(">> All scenarios are completed. Clearning up all actors")
            self.env.clean_up()

            # save video
            if self.save_video:
                data_ids = [config.data_id for config in sampled_scenario_configs]
                self.logger.save_video(data_ids=data_ids)

            # print score for ranking
            self.logger.log(f'[{num_finished_scenario}/{data_loader.num_total_scenario}] Ranking scores for batch scenario:', 'yellow')
            for s_i in score_list.keys():
                self.logger.log('\t Env id ' + str(s_i) + ': ' + str(np.mean(score_list[s_i])), 'yellow')

            # env.env_list中的每一个scenario执行记录，保存在running_results中
            # 根据每一个scenario的data_id，对此次scenario执行记录进行打分
            for scenario_i in range(num_sampled_scenario):
                score_function = get_route_scores if self.scenario_category == 'planning' else get_perception_scores
                all_running_results = self.logger.add_eval_results(records=self.env.env_list[scenario_i].running_results)
                all_scores = score_function(all_running_results)
                self.logger.add_eval_results(scores=all_scores)
                self.logger.print_eval_results()
                if len(self.env.env_list[scenario_i].running_results) % self.save_freq == 0:
                    self.logger.save_eval_results()
        self.logger.save_eval_results()

    def run(self):
        # 解析场景配置文件,返回一个字典,key为地图名称,value为该地图下的所有场景配置
        config_by_map = scenario_parse(self.scenario_config, self.logger)
        for map in config_by_map.keys():  # map是地图的名称
            # initialize map and render
            self._init_world(map)

            start_time = time.time()
            self._init_renderer()
            end_time = time.time()
            print(f"Renderer initialization time: {end_time - start_time} seconds")

            # create scenarios within the vectorized wrapper, 其中self.env含有running_results属性
            if self.agent_config['agent_cfg'][0] == "tcp.yaml":
                self.logger.log(">> Using TCP Vector Wrapper for environment")
                self.env = VectorWrapperTCP(
                    self.env_params,
                    self.scenario_config,
                    self.world,
                    self.birdeye_render,
                    self.display,
                    self.logger
                )
            else:
                self.logger.log(">> Using Standard Vector Wrapper for environment")
                self.env = VectorWrapper(
                    self.env_params,
                    self.scenario_config,
                    self.world,
                    self.birdeye_render,
                    self.display,
                    self.logger
                )


            # Warm up the CARLA server in synchronous mode before building the
            # GlobalRoutePlanner graph.  Switching to sync mode and immediately
            # calling get_topology() can return an incomplete topology, causing
            # nx.astar_path to raise NetworkXNoPath even for valid routes.
            # A few ticks let the server settle before we read road topology.
            WARMUP_TICKS = 5
            self.logger.log(f">> Warming up CARLA server ({WARMUP_TICKS} ticks) before loading route topology...")
            for _ in range(WARMUP_TICKS):
                self.world.tick()

            # prepare data loader and buffer
            data_loader = ScenarioDataLoader(config_by_map[map], self.num_scenario, map, self.world)

            # run with different modes
            if self.mode == 'eval':
                self.agent_policy.load_model()
                self.agent_policy.set_mode('eval')
                self.scenario_policy.set_mode('eval')
                self.eval(data_loader)
            elif self.mode == 'train_agent':
                start_episode = self.check_continue_training(self.agent_policy)
                self.scenario_policy.load_model()
                self.agent_policy.set_mode('train')
                self.scenario_policy.set_mode('eval')
                self.train(data_loader, start_episode)
            elif self.mode == 'train_scenario':
                start_episode = self.check_continue_training(self.scenario_policy)
                self.agent_policy.load_model()
                self.agent_policy.set_mode('eval')
                self.scenario_policy.set_mode('train')
                self.train(data_loader, start_episode)
            else:
                raise NotImplementedError(f"Unsupported mode: {self.mode}.")


    def check_continue_training(self, policy):
        # 根据policy的continue_episode实现断点续训
        policy.load_model()
        if policy.continue_episode == 0:
            start_episode = 0
            self.logger.log('>> Previous checkpoint not found. Training from scratch.')
        else:
            start_episode = policy.continue_episode
            self.logger.log('>> Continue training from previous checkpoint.')
        return start_episode

    def close(self):
        # 在模拟运行结束后一定要将所有设定调回原来的模式，也即异步模式
        # 同步模式下server会等待client的信息来计算下一帧，如果保持同步模式关闭client，server将因为接收不到client的信息而卡死
        settings = self.world.get_settings()
        settings.synchronous_mode = False
        settings.fixed_delta_seconds = None
        self.world.apply_settings(settings)

        pygame.quit() 
        if self.env:
            self.env.clean_up()
