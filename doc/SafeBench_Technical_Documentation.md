# SafeBenchHK 自动驾驶安全测试平台技术文档

> **版本**: 1.1.0  
> **最后更新**: 2026-06-25  
> **依据**: 当前仓库代码实现

---

## 目录

1. [项目概述](#1-项目概述)
2. [代码结构](#2-代码结构)
3. [运行入口与数据流](#3-运行入口与数据流)
4. [输入输出规范](#4-输入输出规范)
5. [核心模块说明](#5-核心模块说明)
6. [TCP 接入实现](#6-tcp-接入实现)
7. [工具脚本与日志整理](#7-工具脚本与日志整理)
8. [环境与运行方式](#8-环境与运行方式)
9. [维护注意事项](#9-维护注意事项)

---

## 1. 项目概述

SafeBenchHK 是基于 CARLA 的自动驾驶安全评测框架。当前代码实现重点支持规划类场景评测，提供标准场景加载、被测 agent 接入、CARLA 同步仿真、评估记录、视频保存和 TCP 端到端模型接入。

### 1.1 当前支持能力

| 能力 | 当前实现 |
| --- | --- |
| 运行模式 | `eval`、`train_agent`、`train_scenario` |
| 被测 Agent | `dummy`、`basic`、`behavior`、`tcp` |
| 场景策略 | `standard`、`ordinary`、`advsim`、`advtraj`、`human`、`random` |
| 场景地图数据 | `central`、`new_central`、`ShaTin`、`new_ShaTin` |
| 场景类型 | 标准 1-8 类交通安全场景，部分地图可能只包含 1-7 |
| 输出结果 | `results.pkl`、`records.pkl`、可选 `.mp4` 视频 |
| TCP 支持 | 独立 TCP agent、TCP env wrapper、TCP CARLA env |

### 1.2 当前未实现或未启用内容

旧文档曾提到的 SAC/PPO/TD3/DDPG agent、Normalizing Flow 场景生成、REINFORCE 场景策略等模块，在当前仓库的 `safebench/agent/__init__.py` 和 `safebench/scenario/__init__.py` 中没有注册，也没有对应可运行实现。本文档不再将它们列为当前能力。

---

## 2. 代码结构

```text
SafeBenchHK/
  scripts/
    run.py                         # 项目级训练/评测入口

  safebench/
    carla_runner.py                # 主调度器
    agent/                         # 被测 agent
      __init__.py                  # AGENT_POLICY_LIST 注册表
      basic.py
      behavior.py
      dummy.py
      tcp.py
      config/
        basic.yaml
        behavior.yaml
        dummy.yaml
        tcp.yaml
      model_ckpt/                  # 本地模型权重，已被 .gitignore 忽略
    scenario/
      __init__.py                  # SCENARIO_POLICY_LIST 注册表
      config/standard.yaml         # 默认场景配置
      scenario_data/               # central/new_central/ShaTin/new_ShaTin 数据
      scenario_data_loader.py      # 场景采样与重叠过滤
      scenario_definition/         # 标准场景类与评价标准
      scenario_manager/            # CARLA actor、timer、traffic event 管理
      scenario_policy/             # dummy/hardcode 场景策略
      tools/                       # route/scenario 解析与操作工具
    gym_carla/
      env_wrapper.py               # 标准 VectorWrapper
      env_wrapper_tcp.py           # TCP VectorWrapper
      envs/
        carla_env.py
        carla_env_tcp.py
        render.py
        route_planner.py
      replay_buffer.py
    carla_agents/                  # Basic/Behavior agent 导航工具
    util/                          # logger、metric、torch、PID 工具

  tools/
    create_routes.py
    create_scenarios.py
    check_route_overlap.py
    export_routes.py
    export_scenarios.py
    visualize_routes_scenarios.py

  log/
    collect_corner_videos.py
    change_name.py
```

---

## 3. 运行入口与数据流

### 3.1 主入口

项目主入口是 `scripts/run.py`。它负责：

1. 解析命令行参数，例如 `--mode`、`--agent_cfg`、`--scenario_cfg`、`--port`、`--tm_port`。
2. 读取 `safebench/agent/config/*.yaml` 和 `safebench/scenario/config/*.yaml`。
3. 将命令行参数合并进 agent/scenario 配置。
4. 创建 `safebench.carla_runner.CarlaRunner`。
5. 对多个 agent 配置和 scenario 配置做两两组合运行。
6. 捕获异常并在批量实验结束后汇总失败项。

### 3.2 运行流程

```text
scripts/run.py
  -> load_config(agent_cfg, scenario_cfg)
  -> CarlaRunner(agent_config, scenario_config)
  -> scenario_parse()
  -> client.load_world(town)
  -> BirdeyeRender
  -> VectorWrapper 或 VectorWrapperTCP
  -> ScenarioDataLoader
  -> agent_policy + scenario_policy
  -> env.reset()/env.step()
  -> Logger 保存 records/results/video
```

### 3.3 Runner 关键逻辑

`safebench/carla_runner.py` 是核心调度器：

- 初始化 CARLA client，并设置同步模式和 `fixed_delta_seconds`。
- 根据 `agent_cfg` 是否为 `tcp.yaml` 选择 `VectorWrapperTCP` 或标准 `VectorWrapper`。
- 初始化 bird-eye renderer；TCP 模式会额外为 900x256 前视相机视图预留窗口宽度。
- 在构建 route topology 前 warm up CARLA 5 tick，避免 `GlobalRoutePlanner` 读取不完整拓扑。
- 在 `eval` 模式加载 agent 模型，并将 agent/scenario policy 切换为评估模式。
- 在训练模式下使用 `RouteReplayBuffer` 或 `PerceptionReplayBuffer` 收集经验。

---

## 4. 输入输出规范

### 4.1 命令行参数

`scripts/run.py` 当前支持的主要参数：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--exp_name` | `scenario_08_results` | 实验名，参与输出目录命名 |
| `--output_dir` | `log` | 输出根目录 |
| `--ROOT_DIR` | 当前项目根目录 | 配置、数据路径拼接根目录 |
| `--mode` | `eval` | `train_agent`、`train_scenario` 或 `eval` |
| `--agent_cfg` | `tcp.yaml` | agent 配置文件列表 |
| `--scenario_cfg` | `standard.yaml` | scenario 配置文件列表 |
| `--num_scenario` | `1` | 同时运行的场景数量 |
| `--save_video` | `True` | 是否保存视频，仅允许 eval 模式 |
| `--render` | `False` | 是否显示 pygame 渲染窗口 |
| `--port` | `2000` | CARLA server 端口 |
| `--tm_port` | `8000` | Traffic Manager 端口 |
| `--fixed_delta_seconds` | `0.1` | CARLA 同步仿真步长 |

### 4.2 Agent 配置

配置目录：`safebench/agent/config/`

已提供：

- `basic.yaml`: CARLA Basic Agent。
- `behavior.yaml`: CARLA Behavior Agent。
- `dummy.yaml`: 空策略。
- `tcp.yaml`: TCP 端到端模型。

`tcp.yaml` 当前关键字段：

```yaml
policy_type: 'tcp'
model_path: '/home/hp/STF/SafeBenchHK/SafeBenchHK/safebench/agent/model_ckpt/tcp/best_model.ckpt'
obs_type: 3
train_episode: 0
buffer_capacity: 10000
```

`policy_type` 必须在 `safebench/agent/__init__.py::AGENT_POLICY_LIST` 中注册。

### 4.3 Scenario 配置

默认配置：`safebench/scenario/config/standard.yaml`

当前关键字段：

```yaml
scenario_type_dir: 'safebench/scenario/scenario_data/new_ShaTin'
scenario_type: 'standard_scenario_08.json'
scenario_category: 'planning'
policy_type: 'standard'
route_dir: 'safebench/scenario/scenario_data/new_ShaTin'
scenario_id: null
route_id: null
ego_action_dim: 2
ego_state_dim: 4
ego_action_limit: 1.0
```

`policy_type` 必须在 `safebench/scenario/__init__.py::SCENARIO_POLICY_LIST` 中注册。

### 4.4 场景数据

当前内置场景数据目录：

```text
safebench/scenario/scenario_data/
  central/
  new_central/
  ShaTin/
  new_ShaTin/
```

每个地图目录通常包含：

```text
standard_scenario_XX.json
scenario_XX_routes/
  scenario_XX_route_YY_weather_ZZ.xml
scenarios/
  scenario_XX.json
```

`scenario_parse()` 会读取：

- `scenario_type_dir/scenario_type`: 测试 case 列表。
- `route_dir/scenario_%02d_routes/...xml`: route XML。
- `route_dir/scenarios/scenario_%02d.json`: scenario trigger/actor JSON。

case JSON 中主要字段：

| 字段 | 说明 |
| --- | --- |
| `data_id` | 测试 case 唯一编号 |
| `scenario_folder` | 场景分类目录名 |
| `scenario_id` | 标准场景 ID |
| `route_id` | route 编号 |
| `weather_id` | 天气编号 |
| `risk_level` | 风险等级，可为 `null` |
| `parameters` | 场景参数，可为 `null` |

### 4.5 输出结果

`Logger` 默认将输出保存到：

```text
<ROOT_DIR>/<output_dir>/<exp_name>/
```

评估模式输出：

```text
eval_results/
  results.pkl    # 每批次/场景评分列表
  records.pkl    # 每个 data_id 的逐帧运行记录
video/
  <timestamp>/
    *.mp4        # save_video=True 时保存
```

训练模式输出：

```text
config.json
config.yaml
training_results/
  results.pkl    # episode、episode_reward 等训练记录
```

评估模式当前会跳过配置备份，只创建 `eval_results/`；训练模式会调用 `Logger.save_config()` 保存 `config.json` 和 `config.yaml`。

---

## 5. 核心模块说明

### 5.1 Agent 注册与接口

注册表：`safebench/agent/__init__.py`

```python
AGENT_POLICY_LIST = {
    'dummy': DummyAgent,
    'basic': CarlaBasicAgent,
    'behavior': CarlaBehaviorAgent,
    'tcp': TCPAgent,
}
```

基础接口：`safebench/agent/base_policy.py::BasePolicy`

必须关注的方法：

- `set_ego_and_route(...)`: 绑定 ego vehicle 和路线信息。
- `get_action(obs, infos, deterministic)`: 输出 ego 控制动作。
- `load_model()`: 加载权重。
- `set_mode(mode)`: 切换 train/eval。
- `save_model(episode)`: 保存训练模型。

注意：`CarlaRunner.train()` 和 `CarlaRunner.eval()` 都会调用 `set_ego_and_route(self.env.get_ego_vehicles(), infos, static_obs=static_obs)`，所以具体 agent 实现需要能接受当前 runner 传入的参数形式。当前 `basic.py`、`behavior.py`、`tcp.py` 的签名略有差异，新增 agent 时应以 runner 的实际调用为准。

### 5.2 Scenario Policy 注册与接口

注册表：`safebench/scenario/__init__.py`

```python
SCENARIO_POLICY_LIST = {
    'standard': DummyPolicy,
    'ordinary': DummyPolicy,
    'advsim': HardCodePolicy,
    'advtraj': HardCodePolicy,
    'human': HardCodePolicy,
    'random': HardCodePolicy,
}
```

基础接口：`safebench/scenario/scenario_policy/base_policy.py::BasePolicy`

主要方法：

- `get_init_action(static_obs, deterministic=False)`: 生成场景初始化动作。
- `get_action(obs, infos, deterministic=False)`: 生成场景动态动作。
- `load_model(scenario_configs=None)`: 加载场景策略或按场景配置准备策略。
- `train(replay_buffer)`: 训练场景策略。
- `set_mode(mode)`: 切换模式。

当前标准场景使用 `DummyPolicy`，通常返回 `None`，标准场景类内部按硬编码逻辑更新 actor 行为。

### 5.3 场景解析与采样

解析入口：`safebench/scenario/tools/scenario_utils.py::scenario_parse`

功能：

1. 读取 scenario case JSON。
2. 根据 `scenario_id`、`route_id` 可选过滤。
3. 通过 `RouteParser.parse_routes_file()` 解析 XML route 和 scenario JSON。
4. 将解析结果按 `town` 分组，供 `CarlaRunner` 按地图逐个加载。
5. 跳过 `Logger` 已加载的 `eval_records` 中已有 `data_id`，支持继续未完成评测。

采样器：`safebench/scenario/scenario_data_loader.py::ScenarioDataLoader`

- 对非 SafeBenchHK 自定义地图，先用 `GlobalRoutePlanner` 插值路线，再按轨迹距离过滤重叠 route。
- 对包含 `safebench` 名称的地图，按 `route_region` 做区域去重。
- 如果 route 插值失败，会跳过对应 route，并打印失败 route id。

### 5.4 标准场景实现

标准场景位于 `safebench/scenario/scenario_definition/standard/`。

| ID | 类名 | 文件 | 当前行为概述 |
| --- | --- | --- | --- |
| 1 | `DynamicObjectCrossing` | `object_crash_vehicle.py` | 生成横穿道路的 walker |
| 2 | `VehicleTurningRoute` | `object_crash_intersection.py` | ego 转弯时遇到直行 actor |
| 3 | `OtherLeadingVehicle` | `other_leading_vehicle.py` | 前车行驶后减速，旁侧可能有第二辆车 |
| 4 | `LaneChange` | `lane_change.py` | 前方低速/相邻车道 actor 形成变道风险 |
| 5 | `OppositeVehicleRunningRedLight` | `junction_crossing_route.py` | 对向车辆闯红灯 |
| 6 | `SignalizedJunctionLeftTurn` | `junction_crossing_route.py` | 有信号灯左转冲突 |
| 7 | `SignalizedJunctionRightTurn` | `junction_crossing_route.py` | 有信号灯右转冲突 |
| 8 | `NoSignalJunctionCrossingRoute` | `junction_crossing_route.py` | 无信号交叉口穿越 |

这些场景继承 `BasicScenario`，通过 `initialize_actorsHK()` 生成 actor，通过 `update_behavior()` 在每步推进 actor 行为，通过 `check_stop_condition()` 判断场景停止条件。

### 5.5 评估指标

规划指标实现：`safebench/util/metric_util.py::get_route_scores`

输出字段：

| 字段 | 含义 |
| --- | --- |
| `collision_rate` | 最后一帧 collision 为 `Status.FAILURE` 的比例 |
| `out_of_road_length` | 平均离路累计距离 |
| `distance_to_route` | 平均偏离 route 距离 |
| `incomplete_route` | `1 - route_completion` |
| `running_time` | 平均仿真运行时长 |
| `penalty_score` | 归一化后按权重加权的惩罚分 |

当前归一化最大值与权重在 `metric_util.py` 中硬编码：

```python
predefined_max_values = {
    'collision_rate': 1,
    'out_of_road_length': 10,
    'distance_to_route': 5,
    'incomplete_route': 1,
    'running_time': time_out,
}

weights = {
    'collision_rate': 0.4,
    'out_of_road_length': 0.1,
    'distance_to_route': 0.1,
    'incomplete_route': 0.3,
    'running_time': 0.1,
}
```

感知指标实现：`get_perception_scores()`，输出 `mean_iou` 和 `mAP_evaluate`。当前默认配置 `scenario_category` 为 `planning`，感知链路不是默认主流程。

---

## 6. TCP 接入实现

当前 TCP 已经接入，不需要新建 `run_tcp.py`。使用统一入口 `scripts/run.py`，并传入 `--agent_cfg tcp.yaml`。

### 6.1 TCP 相关文件

| 文件 | 作用 |
| --- | --- |
| `safebench/agent/tcp.py` | SafeBench agent wrapper，加载 TCP 模型并输出控制 |
| `safebench/agent/config/tcp.yaml` | TCP agent 配置 |
| `safebench/gym_carla/env_wrapper_tcp.py` | TCP 专用 vector wrapper |
| `safebench/gym_carla/envs/carla_env_tcp.py` | TCP 专用 CARLA env |
| `TCP/` | 上游 TCP 模型、leaderboard、scenario_runner、roach 代码 |
| `safebench/agent/model_ckpt/tcp/` | 本地 TCP 权重目录，已被 `.gitignore` 忽略 |

### 6.2 TCP 运行命令

先启动 CARLA server，然后运行：

```bash
python scripts/run.py \
  --mode eval \
  --agent_cfg tcp.yaml \
  --scenario_cfg standard.yaml \
  --num_scenario 1 \
  --port 2000 \
  --tm_port 8000 \
  --render False
```

### 6.3 TCP wrapper 选择逻辑

`CarlaRunner.run()` 中的逻辑：

```python
if self.agent_config['agent_cfg'][0] == "tcp.yaml":
    self.env = VectorWrapperTCP(...)
else:
    self.env = VectorWrapper(...)
```

因此，当前代码用配置文件名判断是否走 TCP wrapper。若后续新增其他端到端模型，建议将该判断改为配置字段，例如 `env_wrapper_type: tcp`。

---

## 7. 工具脚本与日志整理

### 7.1 tools/

| 脚本 | 作用 |
| --- | --- |
| `create_routes.py` | 生成/整理 route XML |
| `create_scenarios.py` | 生成 scenario JSON |
| `check_route_overlap.py` | 检查候选 route 与历史 route 是否重叠 |
| `export_routes.py` | 导出 route |
| `export_scenarios.py` | 导出 scenario |
| `visualize_routes_scenarios.py` | 可视化 route/scenario |
| `get_map_data.py` | 提取地图相关数据 |
| `utilities.py` | 工具函数 |

`check_route_overlap.py` 在检查历史数据时会自动补充同地图的新旧目录组合：

- `central` 与 `new_central`
- `ShaTin` 与 `new_ShaTin`

这样新增 route 不只和单一历史目录比较，也会和同地图的新旧数据一起比较。

### 7.2 log/

`log/` 下保留两个整理脚本：

- `collect_corner_videos.py`: 对比 TCP 与 CarlaAgent 的 `records.pkl`，按 Safe/Corner/Risk 分类复制成对视频。
- `change_name.py`: 对整理后的视频做原地批量重命名。

实际评测视频和结果目录体积较大，已由 `.gitignore` 忽略。

---

## 8. 环境与运行方式

### 8.1 Python 依赖

项目依赖记录在：

- `doc/requirements.txt`: SafeBenchHK 主流程依赖。
- `doc/requirements_tcp.txt`: TCP 相关依赖。

`setup.py` 当前只声明了最小安装依赖：

```python
install_requires=['gym', 'pygame']
```

因此完整环境应优先按 `doc/requirements.txt` 和 TCP 依赖文件安装，而不是只依赖 `pip install -e .` 的自动依赖。

### 8.2 安装示例

```bash
conda create -n safebench python=3.8
conda activate safebench
pip install -r doc/requirements.txt
pip install -e .
```

TCP 相关环境如果单独复现上游 TCP，可参考 `TCP/environment.yml` 和 `doc/requirements_tcp.txt`。

### 8.3 CARLA 启动

代码通过 `carla.Client('localhost', port)` 连接 CARLA。运行前需要先启动 CARLA，并保证端口与 `--port` 一致。

示例：

```bash
./CarlaUE4.sh -RenderOffScreen -quality-level=Low -carla-rpc-port=2000
```

### 8.4 评测示例

Basic Agent：

```bash
python scripts/run.py \
  --mode eval \
  --agent_cfg basic.yaml \
  --scenario_cfg standard.yaml \
  --num_scenario 1 \
  --render False \
  --save_video False
```

TCP Agent：

```bash
python scripts/run.py \
  --mode eval \
  --agent_cfg tcp.yaml \
  --scenario_cfg standard.yaml \
  --num_scenario 1 \
  --render False \
  --save_video True
```

---

## 9. 维护注意事项

### 9.1 已知硬编码点

| 位置 | 当前值/逻辑 | 建议 |
| --- | --- | --- |
| `carla_runner.py` | `warm_up_steps=9`、`display_size=256`、`obs_range=64`、`d_behind=12` | 后续可移入 YAML |
| `carla_runner.py` | TCP wrapper 由 `agent_cfg[0] == "tcp.yaml"` 判断 | 建议改为显式配置字段 |
| `carla_runner.py` | CARLA world weather 固定为 `ClearNoon` | 如需天气评测，应接入 `weather_id` |
| `carla_runner.py` | topology warmup tick 固定为 5 | 可配置化 |
| `metric_util.py` | 指标权重和归一化最大值硬编码 | 可按实验需求配置 |
| `standard/*.py` | 标准场景 actor 类型、速度、触发阈值多处硬编码 | 可按场景配置参数化 |

### 9.2 新增 Agent 步骤

1. 在 `safebench/agent/` 新增 agent 实现，遵循 `BasePolicy` 接口。
2. 在 `safebench/agent/__init__.py` 的 `AGENT_POLICY_LIST` 注册。
3. 在 `safebench/agent/config/` 新增 YAML，设置 `policy_type`。
4. 若观测结构不同，新增或复用 `gym_carla/env_wrapper*.py` 和 `envs/carla_env*.py`。
5. 用 `scripts/run.py --agent_cfg your_agent.yaml` 运行。

### 9.3 新增场景步骤

1. 在 `safebench/scenario/scenario_definition/standard/` 新增场景类。
2. 在 `standard/__init__.py` 导入新类。
3. 在对应地图目录下补充：
   - `standard_scenario_XX.json`
   - `scenario_XX_routes/*.xml`
   - `scenarios/scenario_XX.json`
4. 根据需要更新 `safebench/scenario/tools/route_parser.py` 或场景映射逻辑。
5. 用 `tools/visualize_routes_scenarios.py` 和 `tools/check_route_overlap.py` 检查路线质量。

### 9.4 Git 与产物管理

当前 `.gitignore` 已覆盖：

- Python 缓存和测试缓存。
- CARLA 视频/录制文件。
- `log/` 下地图评测结果。
- `safebench/agent/model_ckpt/` 本地模型权重。
- `TCP/roach/log/`、`TCP/roach/obs_manager/birdview/maps/` 等本地运行资产。

代码、配置、README 和小型 JSON/YAML/XML 场景定义应保留在版本库中；大模型、视频、缓存和批量运行结果不应提交。

---

## 附录 A: 当前注册表

### Agent

```python
{
    'dummy': DummyAgent,
    'basic': CarlaBasicAgent,
    'behavior': CarlaBehaviorAgent,
    'tcp': TCPAgent,
}
```

### Scenario Policy

```python
{
    'standard': DummyPolicy,
    'ordinary': DummyPolicy,
    'advsim': HardCodePolicy,
    'advtraj': HardCodePolicy,
    'human': HardCodePolicy,
    'random': HardCodePolicy,
}
```

---

## 附录 B: 标准场景 ID

| ID | 类名 |
| --- | --- |
| 1 | `DynamicObjectCrossing` |
| 2 | `VehicleTurningRoute` |
| 3 | `OtherLeadingVehicle` |
| 4 | `LaneChange` |
| 5 | `OppositeVehicleRunningRedLight` |
| 6 | `SignalizedJunctionLeftTurn` |
| 7 | `SignalizedJunctionRightTurn` |
| 8 | `NoSignalJunctionCrossingRoute` |
