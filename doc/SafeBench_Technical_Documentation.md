# SafeBenchHK 自动驾驶安全测试平台技术文档

> **版本**: 1.0.0  
> **最后更新**: 2026-01-06  

---

## 目录

1. [项目概述与核心价值 (Executive Summary)]
2. [系统架构与技术栈 (System Architecture)]
3. [输入输出规范 (I/O Specification)]
4. [核心算法逻辑与模块说明 (Core Modules)]
5. [修改点与可定制化分析 (Modification & Customization)]
6. [环境部署与依赖 (Deployment & Dependencies)]
7. [端到端自动驾驶模型接入测试 TCP (End-to-End Autonomous Driving Model Integration & Testing)]
---

## 1. 项目概述与核心价值

### 1.1 功能描述

SafeBenchHK 是一个基于 **CARLA 仿真器** 的自动驾驶安全测试平台，提供以下核心功能：

| 功能模块 | 描述 |
|---------|------|
| **场景化安全测试** | 支持 8 种标准测试场景（路口穿越、变道、车辆碰撞等），覆盖 23 种天气条件 |
| **智能对抗生成** | 内置预定义的标准测试场景和对抗性场景生成算法（REINFORCE、SAC） |
| **支持多种ADS agent接入** | 支持多种被测 Agent（如 CARLA Basic Agent）的接入, 支持 RL 算法 |
| **自动化报告生成** | 自动输出碰撞率、路线完成度等多维度评估指标 |

### 1.2 技术优势

| 优势 | 说明 |
|-----|------|
| **标准化测试流程** | 基于 ScenarioRunner 范式，与 CARLA Leaderboard 标准兼容 |
| **可扩展架构** | 模块化设计，Agent/Scenario 策略可插拔替换 |
| **对抗性测试** | 支持基于强化学习的对抗场景生成，比随机测试更高效 |
| **多维度指标** | 涵盖安全性（碰撞率）、任务性能（路线完成度）、舒适性（横摆角速度）等 |
| **即插即用** | 预置 Carla 内置 Agent (Basic/Behavior) |

### 1.3 适用阶段

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   算法开发阶段    │ →  │    仿真验证阶段    │ →  │   量产前测试      │
│  (train_agent)  │    │     (eval)       │    │  (大规模回归)     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         ↑                      ↑                      ↑
 训练新AD强化学习算法       验证算法安全性边界         批量自动化执行
```

- **算法开发阶段**: 使用 `train_agent` 模式训练规划/感知算法
- **仿真验证阶段**: 使用 `eval` 模式进行定量安全评估
- **对抗测试阶段**: 使用 `train_scenario` 模式让场景策略学习攻击 Agent

---

## 2. 系统架构与技术栈

### 2.1 系统架构图

![Example Image](architecture.png)


### 2.2 技术选型

| 类别 | 技术/版本 | 说明 |
|------|----------|------|
| **编程语言** | Python 3.8+ | 主体开发语言 |
| **深度学习框架** | PyTorch 1.13.1 + CUDA 11.7 | 神经网络训练与推理 |
| **仿真环境** | CARLA 0.9.x | 自动驾驶仿真器 |
| **强化学习库** | 自研实现 (SAC/PPO/TD3/DDPG) |
| **可视化** | Pygame 2.3.0 | 鸟瞰图渲染 |
| **几何计算** | Shapely 1.8.5 | 碰撞检测与路径计算 |
| **科学计算** | NumPy 1.22.0 / Pandas 1.5.3 | 数据处理 |
| **环境接口** | Gym 0.23.1 | 标准 RL 环境接口 |

### 2.3 模块依赖关系

```
safebench/
├── agent/                     # Agent策略模块
│   ├── __init__.py            # AGENT_POLICY_LIST 策略注册表
│   ├── base_policy.py         # 基类接口定义
│   ├── basic.py               # Carla Basic Agent
│   ├── behavior.py            # Carla Behavior Agent
│   ├── dummy.py               # 空策略 (用于纯场景测试)
│   ├── rl/                    # 强化学习算法
│   │   ├── sac.py             # Soft Actor-Critic
│   │   ├── ppo.py             # Proximal Policy Optimization
│   │   ├── td3.py             # Twin Delayed DDPG
│   │   └── ddpg.py            # Deep Deterministic Policy Gradient
│   └── config/                # Agent配置文件目录
│
├── scenario/                  # 场景策略模块
│   ├── __init__.py            # SCENARIO_POLICY_LIST 策略注册表
│   ├── scenario_policy/       # 场景生成策略
│   │   ├── dummy_policy.py    # 标准场景 (无动态生成)
│   │   ├── hardcode_policy.py # 硬编码对抗轨迹
│   │   ├── reinforce_continuous.py  # REINFORCE策略梯度
│   │   ├── normalizing_flow_policy.py  # 条件NF生成
│   │   └── rl/sac.py          # SAC场景策略
│   ├── scenario_definition/   # 场景类型定义
│   │   ├── standard/          # 标准场景
│   │   └── atomic_criteria.py # 原子评价标准
│   ├── scenario_data/         # 场景数据文件
│   └── config/                # Scenario配置文件目录
│
├── gym_carla/                 # Gym环境封装
│   ├── env_wrapper.py         # VectorWrapper 向量化环境
│   ├── envs/
│   │   ├── carla_env.py       # 核心Carla环境
│   │   ├── render.py          # 鸟瞰图渲染
│   │   └── route_planner.py   # 用于鸟瞰图渲染的路径规划器
│   └── replay_buffer.py       # 经验回放缓冲区
│
├── util/                       # 工具模块
│   ├── logger.py              # 日志与视频记录
│   ├── metric_util.py         # 评估指标计算
│   └── torch_util.py          # PyTorch工具函数
│
└── carla_runner.py            # 主运行器
```

---

## 3. 输入输出规范

### 3.1 数据流概览

```
输入数据 → SafeBench核心 → 输出结果
   ↓              ↓              ↓
场景配置       CARLA仿真      评估报告
Agent配置      指标计算       可视化视频
路由数据       碰撞检测       训练权重
```

### 3.2 输入规范

| 数据类型 | 内容描述 | 格式/协议 | 备注 |
|---------|---------|----------|------|
| **场景配置** | 测试场景定义（场景类型、路由ID、天气条件） | `.json` | 见 `scenario_data/center/standard_scenario_*.json` |
| **Agent配置** | 被测算法参数（模型路径、超参数） | `.yaml` | 见 `agent/config/` 目录 |
| **Scenario配置** | 场景策略参数（策略类型、动作维度） | `.yaml` | 见 `scenario/config/` 目录 |
| **路由数据** | 测试路径的 waypoint 序列 | `.xml` | 存储在 `scenario_data/center/scenario_*_routes/` |
| **模型权重** | 预训练的 Agent/Scenario 模型 | `.pth` (PyTorch) | 可选加载 |
| **地图数据** | CARLA 地图资源 | CARLA 内置格式 | 支持 Town01-Town10 + SafeBench 自定义地图 |

#### 3.2.1 场景配置文件结构示例

```json
{
  "data_id": 0,
  "scenario_folder": "standard",
  "scenario_id": 8,
  "route_id": 0,
  "weather_id": 0,
  "risk_level": null,
  "parameters": null
}
```

**字段说明**:
- `data_id`: 唯一测试用例标识
- `scenario_id`: 场景类型 (1-8 对应不同标准场景)
- `route_id`: 路由编号
- `weather_id`: 天气条件 (0-22 共 23 种天气)

#### 3.2.2 Agent配置文件结构示例

```yaml
# behavior.yaml
policy_type: 'behavior'     # 策略类型，对应 AGENT_POLICY_LIST 中的 key
model_path: ''              # 预训练模型路径
model_id: 0                 # 模型版本ID
obs_type: 0                 # 观测类型 (0=4维状态, 1=含车道信息, 2=鸟瞰图, 3=前视图)

# 训练相关参数
train_episode: 2000         # 训练轮数
buffer_capacity: 10000      # 回放缓冲区容量
save_freq: 10               # 模型保存频率
```

### 3.3 中间参数

| 数据类型 | 内容描述 | 格式 | 影响范围 |
|---------|---------|------|---------|
| **阈值参数** | 碰撞检测距离、偏离道路时长 | 硬编码于 `atomic_criteria.py` | 测试灵敏度 |
| **动作空间** | `ego_action_dim=2`, `ego_action_limit=1.0` | `.yaml` 配置 | 控制精度 |
| **观测范围** | `obs_range=64m`, `d_behind=12m` | `carla_runner.py` | 感知范围 |
| **仿真帧率** | `fixed_delta_seconds=0.1s` | 命令行参数 | 仿真精度 |

### 3.4 输出规范

| 数据类型 | 内容描述 | 格式 | 存储路径 |
|---------|---------|------|---------|
| **评估分数** | 碰撞率、路线完成度、偏离道路长度等 | `.pkl` (Python pickle) | `log/{exp_name}/eval/` |
| **运行记录** | 每帧仿真状态（位置、速度、碰撞标记） | `.pkl` | 同上 |
| **训练曲线** | Episode reward、loss | `progress.txt` + `.json` | `log/{exp_name}/train/` |
| **可视化视频** | 测试过程鸟瞰图录像 | `.mp4` | `log/{exp_name}/video/` |
| **模型检查点** | Agent/Scenario 模型权重 | `.pth` | `log/{exp_name}/checkpoints/` |
| **配置备份** | 运行时参数快照 | `config.json` | `log/{exp_name}/` |

#### 3.4.1 评估指标输出结构

```python
# get_route_scores() 输出
{
    # 安全性指标
    'collision_rate': 0.15,        # 碰撞率 [0, 1]
    'out_of_road_length': 2.5,     # 平均偏离道路长度 (米)
    
    # 任务性能指标
    'distance_to_route': 1.2,      # 平均偏离规划路线距离 (米)
    'incomplete_route': 0.05,      # 未完成路线比例 [0, 1]
    'running_time': 25.3,          # 平均任务耗时 (秒)
    
    # 综合惩罚分
    'penalty_score': 0.23          # 加权惩罚分 (越低越好)
}

# get_perception_scores() 输出
{
    'mean_iou': [0.85, 0.82, ...],  # 各场景平均 IoU
    'mAP_evaluate': [0.78, 0.75, ...]  # 各场景 mAP@[0.5:0.95]
}
```

### 3.5 坐标系说明

| 坐标系 | 使用位置 | 定义 |
|-------|---------|------|
| **CARLA World** | 仿真环境内部 | 左手坐标系，X-前，Y-右，Z-上 |
| **Waypoint** | 路由数据 | 采用 CARLA Transform (location + rotation) |
| **鸟瞰图** | 可视化渲染 | 屏幕坐标系，原点在左上角 |

---

## 4. 核心算法逻辑与模块说明

### 4.1 场景生成/调度模块

#### 4.1.1 场景类型选择机制

```
┌─────────────────────────────────────────────────────────┐
│                   ScenarioDataLoader                    │
├─────────────────────────────────────────────────────────┤
│ 1. 加载 JSON 场景配置列表                                  │
│ 2. 计算各场景的插值轨迹                                    │
│ 3. 选择非重叠场景（避免同时测试位置相近的场景）                │
│ 4. 批量采样并返回配置                                      │
└─────────────────────────────────────────────────────────┘
```

**采样策略**:
- **基于区域去重** (SafeBench香港地图): 同一区域只采样一个场景
- **基于轨迹去重** (CARLA地图): 检测轨迹点距离，阈值 10m

#### 4.1.2 场景策略类型

| 策略类型 | 类名 | 算法原理 | 适用场景 |
|---------|------|---------|---------|
| `standard` | `DummyPolicy` | 无动态生成，使用预定义场景 | 标准化测试 |
| `ordinary` | `DummyPolicy` | 普通交通流 | Agent 训练 |
| `lc` | `REINFORCE` | 策略梯度优化 | 在线对抗学习 |
| `nf` | `NormalizingFlow` | 条件正则化流 | 分布式场景采样 |
| `sac` | `SAC` | Soft Actor-Critic | 高效对抗探索 |

#### 4.1.3 对抗场景生成算法 - Normalizing Flow

```python
# 核心架构: Conditional RealNVP
class ConditionalRealNVP:
    """
    n_flows: 流层数量 (默认 4)
    condition_dim: 条件维度 (如路由信息)
    data_dim: 生成数据维度 (如 NPC 轨迹)
    """
    def forward(x, c):  # 正向: 数据 → 潜变量
        for k in range(n_flows):
            x1, x2 = x[:, :n_half], x[:, n_half:]
            s, t = NN[k](concat(x1, c))
            x2 = x2 * exp(s) + t
            x = concat(x2, x1)
        return x, log_det_jacobian
    
    def inverse(z, c):  # 逆向: 潜变量 → 数据
        # 用于采样新场景
```

**训练目标**: 最大化高风险场景的似然概率

### 4.2 评估指标 (Metrics)

#### 4.2.1 规划任务评估指标

##### 安全性指标

| 指标 | 计算公式 | 权重 | 说明 |
|-----|---------|------|------|
| **碰撞率** (Collision Rate) | $CR = \frac{N_{collision}}{N_{total}}$ | 40% | 发生碰撞的场景比例 |
| **偏离道路长度** | $L_{offroad} = \sum_{t} d_t \cdot \mathbb{1}_{offroad}(t)$ | 10% | 累计偏离行驶车道的距离 |

##### 任务性能指标

| 指标 | 计算公式 | 权重 | 说明 |
|-----|---------|------|------|
| **路线完成度** | $RC = \frac{d_{traveled}}{d_{total}} \times 100\%$ | 30% | 实际完成的路线比例 |
| **路径偏离距离** | $D_{route} = \frac{1}{T} \sum_{t} dist(p_t, route)$ | 10% | 平均偏离规划路线的距离 |
| **任务耗时** | $T_{run}$ (秒) | 10% | 完成任务的总时间 |

##### 综合惩罚分计算

$$
S_{penalty} = 0.4 \times \frac{CR}{1} + 0.1 \times \frac{L_{offroad}}{10} + 0.1 \times \frac{D_{route}}{5} + 0.3 \times (1 - RC) + 0.1 \times \frac{T_{run}}{T_{timeout}}
$$

其中 $T_{timeout} = 60s$


#### 4.2.2 原子评价标准 (Atomic Criteria)

系统内置若干种原子检测器，主要包括：

| 检测器 | 类名 | 功能 |
|-------|------|------|
| **碰撞检测** | `CollisionTest` | 检测 ego 与其他物体的碰撞事件 |
| **偏离道路** | `OffRoadTest` | 检测离开可行驶区域 |
| **路线完成** | `RouteCompletionTest` | 计算规划路线完成百分比 |
| **闯红灯** | `RunningRedLightTest` | 检测违规通过红灯 |
| **速度测试** | `ActorSpeedAboveThresholdTest` | 检测速度异常（过慢/超速） |
| **行驶距离** | `DrivenDistanceTest` | 累计行驶里程 |
| **平均速度** | `AverageVelocityTest` | 计算任务期间平均车速 |

### 4.3 闭环/开环逻辑

```
┌─────────────────────────────────────────────────────────────────┐
│                        运行模式说明                               │
├────────────────┬────────────────────────────────────────────────┤
│    模式         │                     数据流                      │
├────────────────┼────────────────────────────────────────────────┤
│   eval         │  Scenario → Env → Agent → Metrics (开环评估)    │
│                │  场景固定，仅收集 Agent 表现数据                   │
├────────────────┼────────────────────────────────────────────────┤
│  train_agent   │  Scenario → Env ⇄ Agent (闭环训练)              │
│                │  场景固定，Agent 基于 reward 更新策略             │
├────────────────┼────────────────────────────────────────────────┤
│ train_scenario │  Scenario ⇄ Env ← Agent (对抗训练)              │
│                │  Agent 固定，Scenario 学习生成更难场景             │
└────────────────┴────────────────────────────────────────────────┘
```

**训练循环伪代码**:

```python
for episode in range(train_episode):
    while scenarios_remaining:
        # 采样场景配置
        configs = data_loader.sampler()
        
        # 获取场景初始动作（如 NPC 位置）
        init_action = scenario_policy.get_init_action(static_obs)
        obs = env.reset(configs, init_action)
        
        while not env.all_scenario_done():
            # Agent 决策
            ego_action = agent_policy.get_action(obs)
            # Scenario 决策（如 NPC 运动）
            scenario_action = scenario_policy.get_action(obs)
            
            # 环境执行
            next_obs, reward, done, info = env.step(ego_action, scenario_action)
            
            # 存储经验 & 训练
            buffer.store(...)
            if mode == 'train_agent':
                agent_policy.train(buffer)
            elif mode == 'train_scenario':
                scenario_policy.train(buffer)
```

---

## 5. 修改点与可定制化分析

### 5.1 硬编码限制

> [!WARNING]
> 以下参数目前硬编码在源码中，对接时需要提取到配置文件。

| 硬编码位置 | 当前值 | 建议处理 |
|-----------|-------|---------|
| `carla_runner.py:55` | `warm_up_steps = 9` | 提取到 YAML 配置 |
| `carla_runner.py:57` | `display_size = 256` | 提取到 YAML 配置 |
| `carla_runner.py:58-59` | `obs_range = 64, d_behind = 12` | 影响感知范围，需配置化 |
| `carla_runner.py:62-63` | 离散加速度/转向值 | 提取为控制模式配置 |
| `carla_runner.py:70` | `desired_speed = 25 km/h` | 提取到任务配置 |
| `metric_util.py:86-95` | 指标归一化最大值与权重 | 需根据测试需求调整 |
| `atomic_criteria.py` | 碰撞检测阈值、偏离时长 | 建议参数化 |

### 5.2 接口适配指南

#### 5.2.1 替换被测 Agent

**接口类**: `safebench/agent/base_policy.py::BasePolicy`

```python
class BasePolicy:
    """所有 Agent 策略必须实现的接口"""
    
    def set_ego_and_route(self, ego_vehicles, route_infos):
        """初始化 ego 车辆和路由信息"""
        pass
    
    def get_action(self, obs, infos, deterministic=False):
        """
        核心决策接口
        Args:
            obs: 观测数据 (dict 或 ndarray)
            infos: 场景附加信息
            deterministic: 是否确定性输出
        Returns:
            actions: list[dict] 每辆车的控制命令
                     {'throttle': float, 'steer': float, 'brake': float}
        """
        pass
    
    def load_model(self):
        """加载预训练权重"""
        pass
    
    def set_mode(self, mode):  # 'train' or 'eval'
        """切换训练/评估模式"""
        pass
```

**注册新 Agent**:

```python
# safebench/agent/__init__.py
from your_module import YourAgent

AGENT_POLICY_LIST = {
    ...
    'your_agent': YourAgent,  # 添加到策略注册表
}
```

#### 5.2.2 替换场景策略

**接口类**: `safebench/scenario/scenario_policy/base_policy.py::BasePolicy`

```python
class BasePolicy:
    name = 'base'
    type = 'init_state'  # 或 'onpolicy', 'offpolicy'
    
    def get_init_action(self, static_obs, deterministic=False):
        """
        生成场景初始状态（如 NPC 初始位置）
        Returns:
            init_action: 场景初始化动作
            additional_dict: 附加信息
        """
        pass
    
    def get_action(self, obs, infos, deterministic=False):
        """
        生成场景动态动作（如 NPC 运动控制）
        """
        pass
    
    def train(self, replay_buffer):
        """训练场景策略（对抗训练模式）"""
        pass
```

#### 5.2.3 自定义场景

**步骤**:

1. 在 `scenario/scenario_definition/standard/` 创建新场景类
2. 在 `scenario/scenario_data/center/` 添加对应路由 XML 和场景 JSON
3. 在 `scenario/scenario_definition/standard/__init__.py` 注册

**场景类模板**:

```python
from safebench.scenario.scenario_definition.basic_scenario import BasicScenario

class YourCustomScenario(BasicScenario):
    def __init__(self, world, config, route, ego_vehicle, logger):
        super().__init__(world, config, route, ego_vehicle, logger)
        # 初始化 NPC 车辆、行人等
        
    def create_behavior(self):
        """定义场景行为（使用 py_trees 行为树）"""
        pass
        
    def get_criteria(self):
        """返回评价标准列表"""
        return [
            CollisionTest(self.ego_vehicle),
            RouteCompletionTest(self.ego_vehicle, self.route),
        ]
```

### 5.3 性能瓶颈分析

| 瓶颈点 | 当前延迟 | 优化建议 |
|-------|---------|---------|
| **CARLA 渲染** | ~100ms/帧 (取决于场景复杂度) | 降低渲染质量或使用离屏渲染 |
| **PyTorch 推理** | ~1-5ms/帧 | TensorRT 加速或 ONNX 导出 |
| **鸟瞰图生成** | ~10-20ms/帧 | 可选关闭 `--render False` |
| **视频编码** | 写入时 ~30ms/帧 | 使用 GPU 编码或后处理 |
| **场景初始化** | 首帧 ~1-2s | 预热阶段不可避免 |

---

## 6. 环境部署与依赖

### 6.1 硬件要求

| 组件 | 推荐配置 |
|-----|---------|
| **GPU** | NVIDIA RTX 4090 (24GB) |
| **CPU** | Intel i9-12900K / AMD Ryzen 9 5950X |
| **RAM** | 32GB+ |
| **存储** | 100GB NVMe SSD |
| **操作系统** | Ubuntu 22.04 |


### 6.2 软件依赖

#### 6.2.1 系统级依赖

| 软件 | 版本要求 | 说明 |
|-----|---------|------|
| **CARLA Simulator** | 0.9.16 | 需与 API 版本匹配 |
| **CUDA Toolkit** | 11.7+ | PyTorch 编译匹配 |
| **NVIDIA Driver** | 515+ | 支持 CUDA 11.7 |
| **Python** | 3.8 - 3.10 | 建议 3.8 |

#### 6.2.2 Python 依赖

```txt
# 核心依赖 (requirements.txt)
--extra-index-url https://download.pytorch.org/whl/cu117
torch==1.13.1+cu117
torchvision==0.14.1+cu117
torchaudio==0.13.1

# 仿真与可视化
gym==0.23.1
pygame==2.3.0
carla==0.9.16  # 需与 CARLA 版本匹配

# 科学计算
numpy==1.22.0
pandas==1.5.3
shapely==1.8.5

# 工具库
pyyaml==6.0
tqdm==4.65.0
joblib==1.2.0
cpprb==10.7.0
opencv-python==4.7.0.72
matplotlib==3.5.3
seaborn==0.12.2
moviepy==1.0.3
scikit-image==0.19.3
```

### 6.3 安装步骤

#### 6.3.1 CARLA 安装

```bash
# 1. 下载 CARLA (以 0.9.16 为例)
wget https://carla-releases.s3.eu-west-3.amazonaws.com/Linux/CARLA_0.9.16.tar.gz
tar -xzf CARLA_0.9.16.tar.gz

# 2. 安装 CARLA Python API
# 也可以使用 carla 包
pip install carla==0.9.16

# 3. 启动 CARLA 服务器
./CarlaUE4.sh -RenderOffScreen -quality-level=Low  # Linux
# CarlaUE4.exe -RenderOffScreen -quality-level=Low  # Windows
```

#### 6.3.2 SafeBench 安装

```bash
# 1. 克隆项目
cd SafeBenchHK

# 2. 创建虚拟环境
conda create -n safebench python=3.8
conda activate safebench

# 3. 安装依赖
pip install -r requirements.txt

# 4. 安装项目包
pip install -e .
```

#### 6.3.3 验证安装

```bash
# 确保 CARLA 服务器运行后执行
python scripts/run.py --mode eval \
    --agent_cfg behavior.yaml \
    --scenario_cfg standard.yaml \
    --num_scenario 1 \
    --render True
```

### 6.4 常见问题

| 问题 | 解决方案 |
|-----|---------|
| `carla.Client` 连接超时 | 确保 CARLA 服务器已启动，检查端口 2000 是否被占用 |
| CUDA out of memory | 减少 `--num_scenario` 或使用更大显存 GPU |
| Pygame 显示黑屏 | 添加 `--render False` 或检查 DISPLAY 环境变量 |
| 场景加载卡住 | 检查 CARLA 日志，可能地图资源缺失 |

---
## 7.端到端自动驾驶模型接入测试指南（TCP 为例）
本文档详细介绍基于 SafeBench 框架的端到端自动驾驶模型（如 TCP）的完整接入流程，包含环境配置、模型部署、测试启动等核心步骤。

### 7.1 Python 依赖安装
#### 7.1.1 依赖清单
```txt
absl-py==0.15.0
aiohttp==3.8.1
aiosignal==1.2.0
albumentations==1.1.0
antlr4-python3-runtime==4.8
argcomplete==1.12.3
argon2-cffi==21.3.0
carla==0.9.13  # 与CARLA模拟器版本对应
gym==0.21.0
numpy==1.21.6
pygame==2.1.2
scikit-image==0.19.3
torch==1.12.1
torchvision==0.13.1
```

#### 7.1.2 安装命令
执行以下命令安装依赖：
```bash
pip install -r requirements_tcp.txt
```

### 7.2 端到端自动驾驶算法配置
#### 7.2.1 测试配置文件（YAML）
配置路径
```
safebench/agent/config/*.yaml
```

#### 7.2.2 YAML 模板
在上述路径下新建模型配置文件（如 `tcp_config.yaml`），内容如下：
```yaml
# TCP (Trajectory-guided Control Prediction) 端到端自动驾驶算法配置
# 模型实现在 agent/tcp.py 中
policy_type: 'tcp'
model_path: '/home/hp/STF/SafeBenchHK/SafeBenchHK/safebench/agent/model_ckpt/tcp/best_model.ckpt'
obs_type: 3  # 观测类型：使用前视图+状态

# TCP为预训练模型，无需在线训练
train_episode: 0
eval_in_train_freq: 1
save_freq: 10
buffer_capacity: 10000
buffer_start_training: 100
```

#### 7.2.3 权重文件目录配置
在 SafeBench 框架中创建模型权重存储目录：
```
safebench/
├── agent/
    └── model_ckpt/
        ├── ppo/
        ├── sac/
        └── tcp/  # 放入TCP模型的权重文件
```

#### 7.2.4 模型代码添加
将端到端模型的核心代码文件（如 `tcp.py`）放入以下路径：
```
SafeBenchHK/SafeBenchHK/TCP
```

### 7.3 Agent 代码文件添加
#### 7.3.1 添加路径
```
safebench/agent/*.py
```

#### 7.3.2 参考模板
可参考同路径下的 `template.py` 或其他算法文件（如 `ppo.py`）编写模型的 Agent 逻辑，需实现**模型加载、推理、动作输出**等核心功能。

### 7.4 模型管理接口添加（Env Wrapper）
Env Wrapper 用于统一模型的观测空间、动作空间和数据预处理逻辑，是框架与模型的核心交互层。

#### 7.4.1 接口添加路径
```
safebench/gym_carla/env_wrapper_*.py
```

#### 7.4.2 参考模板
参考 `safebench/gym_carla/env_wrapper_template.py`，按以下 5 步修改：

**步骤 1：配置模型专属参数**
在文件头部的【模型专属配置区】修改以下参数：
```python
# 1. 自定义环境ID（避免与其他模型冲突）
MODEL_ENV_ID = "carla-tcp-v0"
# 2. 自定义观测类型标识（建议从10开始，避免与内置类型冲突）
MODEL_OBS_TYPE = 10
# 3. 导入模型专用的CARLA环境类
from safebench.gym_carla.envs.carla_env_tcp import CarlaEnvTCP
```

**步骤 2：修改观测空间定义**
在 `_build_obs_space` 方法中，添加 `MODEL_OBS_TYPE` 对应的观测空间：
```python
elif self.obs_type == MODEL_OBS_TYPE:
    # 示例：TCP模型输入为 900x256 图像 + 4维状态，需根据实际调整
    obs_space_dict = {
        'camera_raw': spaces.Box(low=0, high=255, shape=(256, 900, 3), dtype=np.uint8),
        'state': spaces.Box(np.array([-2, -1, -5, 0]), np.array([2, 1, 30, 1]), dtype=np.float32)
    }
    self.observation_space = spaces.Dict(obs_space_dict)
```

**步骤 3：修改观测预处理逻辑**
在 `_preprocess_obs` 方法中，定义框架观测到模型输入的转换规则：
```python
elif self.obs_type == MODEL_OBS_TYPE:
    # 示例：TCP模型直接使用原始相机图像和状态
    return {
        'camera_raw': obs['camera_raw'],
        'state': obs['state']
    }
```

**步骤 4：修改动作空间（可选）**
若模型动作维度或范围与默认不同（默认：2维连续动作，油门/转向），调整动作空间定义：
```python
# 示例：TCP动作空间为 2维连续值 [-1,1]（无需修改则跳过）
act_dim = 2
act_lim = np.ones((act_dim), dtype=np.float32)
self.action_space = spaces.Box(-act_lim, act_lim, dtype=np.float32)
```

**步骤 5：完善环境构造函数**
在 `carla_env_model` 函数中，关联模型专用的 CARLA 环境类：
```python
return ObservationWrapperModel(
    CarlaEnvTCP(
        env_params=env_params,
        birdeye_render=birdeye_render,
        display=display,
        world=world,
        logger=logger,
    ),
    obs_type=env_params['obs_type']
)
```

### 7.5 CARLA 环境配置（CarlaEnv）
CarlaEnv 是与 CARLA 模拟器直接交互的模块，负责传感器数据采集、场景运行、奖励计算等。

#### 7.5.1 配置路径
```
safebench/gym_carla/envs/carla_env_*.py
```

#### 7.5.2 参考模板
参考 `safebench/gym_carla/envs/carla_env_template.py`，按以下 5 步修改模型专属逻辑：

**步骤 1：配置模型专属传感器参数**
在文件头部的【模型专属配置区】定义传感器参数（以 TCP 为例）：
```python
# TCP 相机配置
MODEL_CAMERA_WIDTH = 900
MODEL_CAMERA_HEIGHT = 256
MODEL_CAMERA_FOV = 100
# 修正语法错误：使用字典存储相机位置
MODEL_CAMERA_POS = {"x": -1.5, "z": 2.0}
# 模型专属观测键值
MODEL_OBS_KEYS = ['camera_raw']
```

**步骤 2：扩展传感器数据存储变量**
在 `__init__` 方法的【模型专属逻辑】区域，添加传感器数据的存储变量：
```python
# TCP 专用：存储原始尺寸相机图像
self.camera_img_raw = None
```

**步骤 3：定制相机传感器配置**
在 `_create_sensors` 方法的【模型专属逻辑】区域，配置传感器蓝图和参数：
```python
# TCP 相机传感器配置
self.camera_img = np.zeros((MODEL_CAMERA_HEIGHT, MODEL_CAMERA_WIDTH, 3), dtype=np.uint8)
# 使用字典解包传递位置参数
self.camera_trans = carla.Transform(carla.Location(**MODEL_CAMERA_POS))
self.camera_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
self.camera_bp.set_attribute('image_size_x', str(MODEL_CAMERA_WIDTH))
self.camera_bp.set_attribute('image_size_y', str(MODEL_CAMERA_HEIGHT))
self.camera_bp.set_attribute('fov', str(MODEL_CAMERA_FOV))
self.camera_bp.set_attribute('sensor_tick', '0.02')
```

**步骤 4：扩展观测空间和观测字典**
1.  在 `__init__` 方法中，将 `MODEL_OBS_KEYS` 对应的观测字段加入 `observation_space_dict`：
    ```python
    # 添加 TCP 原始相机图像观测
    for obs_key in MODEL_OBS_KEYS:
        if obs_key == 'camera_raw':
            observation_space_dict[obs_key] = spaces.Box(
                low=0, high=255,
                shape=(MODEL_CAMERA_HEIGHT, MODEL_CAMERA_WIDTH, 3),
                dtype=np.uint8
            )
    ```
2.  在 `_get_obs` 方法的【模型专属逻辑】区域，将传感器数据加入观测字典：
    ```python
    # 添加 TCP 原始相机图像到观测
    if 'camera_raw' in MODEL_OBS_KEYS:
        obs['camera_raw'] = self.camera_img_raw.astype(np.uint8)
    ```

**步骤 5：修改奖励函数**
在 `_get_reward` 方法中，调整奖励项的权重或新增奖励项，适配模型训练目标（以 TCP 为例）：
```python
def _get_reward(self):
    # TCP 奖励函数：碰撞惩罚 + 距离奖励 + 平滑惩罚
    is_collision = len(self.collision_hist) > 0
    r_collision = -25.0 if is_collision else 0.0

    # 与参考车辆的距离奖励
    adv_veh = self.scenario_manager.scenario_list[0].reference_actor
    dist = self.ego_vehicle.get_location().distance(adv_veh.get_location()) if adv_veh else 100.0
    r_distance = np.exp(-0.2 * dist) * 5.0

    # 转向平滑惩罚
    r_steer = -self.ego_vehicle.get_control().steer ** 2 * 2.0

    total_reward = r_collision + r_distance + r_steer
    return total_reward
```

### 7.6 启动模型测试
执行以下命令启动端到端模型的评估测试：
```bash
python scripts/run_tcp.py \
    --mode eval \
    --agent_cfg safebench/agent/config/tcp_config.yaml \
    --scenario_cfg standard.yaml \
    --num_scenario 1 \
    --render True
```

### 参数说明
| 参数名 | 说明 |
|--------|------|
| `--mode` | 运行模式，`eval` 为评估模式，`train` 为训练模式 |
| `--agent_cfg` | 模型配置文件路径 |
| `--scenario_cfg` | 测试场景配置文件路径 |
| `--num_scenario` | 测试场景数量 |
| `--render` | 是否可视化渲染（`True`/`False`） |

### 7.7 目录结构总结
```
SafeBenchHK/
├── safebench/
│   ├── agent/
│   │   ├── config/
│   │   │   └── tcp_config.yaml  # 模型配置文件
│   │   ├── model_ckpt/
│   │   │   └── tcp/             # 模型权重目录
│   │   └── tcp.py               # Agent 代码文件
│   └── gym_carla/
│       ├── env_wrapper_tcp.py   # 模型管理接口
│       └── envs/
│           └── carla_env_tcp.py # CARLA 环境配置
├── TCP/
│   └── tcp.py                   # 端到端自动驾驶模型代码
├── scripts/
│   └── run_tcp.py               # 测试启动脚本
└── requirements_tcp.txt         # 依赖清单
```

```bash
## 附录

### A. 支持的场景类型

| ID | 场景名称 | 描述 |
|----|---------|------|
| 1 | DynamicObjectCrossing | 行人穿越场景 |
| 2 | VehicleTurningRoute | 主车转弯场景 |
| 3 | OtherLeadingVehicle | 前车减速场景 |
| 4 | LaneChange | 前车低速行驶，主车变道场景 |
| 5 | OppositeVehicleRunningRedLight | 对向车辆闯红灯场景 |
| 6 | SignalizedJunctionLeftTurn | 主车向左汇入车流场景 |
| 7 | SignalizedJunctionRightTurn | 主车向右汇入车流场景 |
| 8 | NoSignalJunctionCrossingRoute | 交叉口穿越场景 |

### B. 支持的天气条件

| ID | 天气 |
|----|-----|
| 0 | ClearNoon |
| 1 | CloudyNoon |
| 2 | WetNoon |
| 3 | WetCloudyNoon |
| ... | ... |

### C. 命令行参数完整列表

```bash
python scripts/run.py --XX

参数:
  --exp_name        实验名称 (默认: 'exp')
  --output_dir      输出目录 (默认: 'log')
  --mode, -m        运行模式 [train_agent|train_scenario|eval]
  --agent_cfg       Agent 配置文件列表
  --scenario_cfg    Scenario 配置文件列表
  --num_scenario    并行场景数量 (默认: 1)
  --max_episode_step  每轮最大步数 (默认: 2000)
  --fixed_delta_seconds  仿真时间步长 (默认: 0.1s)
  --port            CARLA 端口 (默认: 2000)
  --render          是否显示渲染窗口 (默认: True)
  --save_video      是否保存视频 (默认: True)
  --seed, -s        随机种子 (默认: 0)
```

---

> **文档维护**: 如有更新需求，请联系项目维护团队
