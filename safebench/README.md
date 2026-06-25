# safebench

`safebench/` 是本项目的核心评测框架，负责连接 CARLA、加载被测 agent、组织场景、运行 episode，并输出安全评测指标。

## 目录说明

- `carla_runner.py`: 训练/评测统一调度入口，由 `scripts/run.py` 创建并执行。
- `agent/`: 被测车辆策略接口与实现，包括 `basic`、`behavior`、`dummy`、`tcp` 等 agent；`agent/config/` 存放对应配置。
- `scenario/`: 场景数据加载、标准场景定义、场景管理器和场景策略；`scenario/config/standard.yaml` 是默认场景配置。
- `gym_carla/`: CARLA 环境封装，提供 SafeBench 使用的环境接口、渲染和 replay buffer。
- `carla_agents/`: 从 CARLA agent 逻辑整理出的导航、局部规划和控制工具。
- `util/`: 日志、指标、PID、PyTorch 和运行时配置等通用工具。

## 主要数据

- `scenario/scenario_data/`: 项目内置地图和场景路线数据，评测时由 scenario loader 读取。
- `agent/model_ckpt/`: TCP 等 agent 的模型权重。该目录体积较大，已在 `.gitignore` 中作为本地资产忽略。

## 调用关系

常规入口是根目录下的 `scripts/run.py`。它读取 `agent/config/*.yaml` 与 `scenario/config/*.yaml`，合并命令行参数后交给 `CarlaRunner` 执行。
