"""
SafeBench/CARLA 评测与训练主入口脚本。

脚本用途：
    1. 从命令行读取实验名称、运行模式、agent 配置、scenario 配置、端口等全局参数。
    2. 加载 safebench/agent/config 和 safebench/scenario/config 下的 yaml 配置文件。
    3. 将命令行参数合并进配置，并交给 CarlaRunner 统一执行训练或评测流程。

支持模式：
    train_agent: 训练自车 agent。
    train_scenario: 训练或搜索测试场景策略。
    eval: 使用指定 agent 和 scenario 配置进行评测。

典型输出：
    日志、评测记录、可选视频等会按 exp_name/output_dir 配置写入结果目录。
"""
import argparse
import traceback
import os.path as osp
import time
import torch
from safebench.util.run_util import load_config
from safebench.util.torch_util import set_seed, set_torch_variable
from safebench.carla_runner import CarlaRunner


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # 实验名称通常会参与日志目录命名，用于区分不同轮次或不同场景的结果。
    parser.add_argument('--exp_name', type=str, default='scenario_08_results')

    # 定义测试结果的输出目录
    parser.add_argument('--output_dir', type=str, default='log')
    # ROOT_DIR 默认取项目根目录，后续拼接配置文件路径时会使用。
    parser.add_argument('--ROOT_DIR', type=str, default=osp.abspath(osp.dirname(osp.dirname(osp.realpath(__file__)))))

    # 单个 episode 的最大仿真步数；超过后 runner 会结束当前场景。
    parser.add_argument('--max_episode_step', type=int, default=2000)
    # auto_ego 为 True 时通常表示 ego vehicle 由内置逻辑控制，而非外部 agent
    parser.add_argument('--auto_ego', type=bool, default=False)

    # 提供三种模式选择：训练agent、训练scenario、evaluation
    parser.add_argument('--mode', '-m', type=str, default='eval', choices=['train_agent', 'train_scenario', 'eval'])
    # 支持一次传入多个 agent/scenario 配置，脚本会两两组合依次运行。
    parser.add_argument('--agent_cfg', nargs='+', type=str, default=['tcp.yaml'])
    parser.add_argument('--scenario_cfg', nargs='+', type=str, default=['standard.yaml'])
    # 是否从已有 checkpoint 继续训练。
    parser.add_argument('--continue_agent_training', '-cat', type=bool, default=False)
    parser.add_argument('--continue_scenario_training', '-cst', type=bool, default=False)

    # 随机种子、PyTorch 线程数和运行设备，用于控制复现实验与计算资源
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu')   

    # 每个 episode 同时运行的场景数量，以及是否保存视频/渲染窗口。
    parser.add_argument('--num_scenario', '-ns', type=int, default=1, help='num of scenarios we run in one episode')
    parser.add_argument('--save_video', type=bool, default=True)
    parser.add_argument('--render', type=bool, default=False)

    # 每间隔多少帧再计算一次，数值太大会导致计算不及时，一直使用之前t计算得到的数据进行控制
    parser.add_argument('--frame_skip', '-fs', type=int, default=1, help='skip of frame in each step')
    parser.add_argument('--port', type=int, default=2000, help='port to communicate with carla')
    parser.add_argument('--tm_port', type=int, default=8000, help='traffic manager port')
    # carla world中每一帧的时间间隔
    parser.add_argument('--fixed_delta_seconds', type=float, default=0.1, help='time for each frame')

    args = parser.parse_args()
    # 转成字典后会合并到 agent_config 和 scenario_config，保证 runner 可统一读取参数。
    args_dict = vars(args)

    err_list = []
    for agent_cfg in args.agent_cfg:
        for scenario_cfg in args.scenario_cfg:
            # 设置全局计算参数，保证每组配置运行前的随机性和线程环境一致。
            set_torch_variable(args.device)
            torch.set_num_threads(args.threads)
            set_seed(args.seed)

            # 加载 agent 配置，例如 tcp.yaml、basic.yaml、behavior.yaml 等。
            agent_config_path = osp.join(args.ROOT_DIR, 'safebench/agent/config', agent_cfg)
            agent_config = load_config(agent_config_path)

            # 加载场景配置，例如 standard.yaml；场景配置决定测试任务和扰动策略。
            scenario_config_path = osp.join(args.ROOT_DIR, 'safebench/scenario/config', scenario_cfg)
            scenario_config = load_config(scenario_config_path)

            # 命令行参数优先级更高：合并后可覆盖 yaml 中同名配置。
            agent_config.update(args_dict)
            scenario_config.update(args_dict)
            # CarlaRunner 是训练/评测的统一调度器，内部会根据 mode 分派具体流程。
            runner = CarlaRunner(agent_config, scenario_config)

            # 启动当前 agent_cfg + scenario_cfg 组合的运行，并记录耗时。
            start_time = time.time()
            try:
                runner.run()
            except:
                traceback.print_exc()
                # agent_cfg: 当前被测试的agent配置文件; scenario_cfg: 生成测试场景的配置文件; traceback: 错误信息
                err_list.append([agent_cfg, scenario_cfg, traceback.format_exc()])
            runner.close()
            end_time = time.time()
            print(f"Total time for {agent_cfg} and {scenario_cfg}: {end_time - start_time} seconds")

    # 所有组合跑完后统一打印失败列表，便于批量实验结束后排查。
    for err in err_list:
        print(err[0], err[1], 'failed!')
        print(err[2])
