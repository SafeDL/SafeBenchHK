# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Please always respond in Simplified Chinese, regardless of the language of my input. 请始终使用简体中文回复，无论我输入的是什么语言

## Project Overview

**SafeBenchHK** is an autonomous driving safety testing platform built on the CARLA simulator. It supports:
- **8 standard test scenarios** (intersection crossing, lane changing, collisions, etc.)
- **23 weather conditions** for environmental variation
- **Multiple agent types**: CARLA built-in agents, RL-based agents (SAC, DDPG, PPO, TD3), and end-to-end models (TCP)
- **Adversarial scenario generation** using RL algorithms or normalizing flows
- **Multi-dimensional evaluation metrics**: collision rate, route completion, off-road distance, comfort metrics

## Common Commands

### Basic Evaluation

```bash
python scripts/run.py \
    --mode eval \
    --agent_cfg basic.yaml \
    --scenario_cfg standard.yaml \
    --exp_name my_test \
    --seed 0
```

### Training RL Agent

```bash
python scripts/run.py \
    --mode train_agent \
    --agent_cfg sac.yaml \
    --scenario_cfg standard.yaml \
    --exp_name train_sac_agent \
    --seed 0
```

### Training Adversarial Scenarios

```bash
python scripts/run.py \
    --mode train_scenario \
    --agent_cfg basic.yaml \
    --scenario_cfg sac.yaml \
    --exp_name adv_scenario_gen \
    --seed 0
```

### Multi-Config Grid Search

```bash
python scripts/run.py \
    --agent_cfg basic.yaml behavior.yaml \
    --scenario_cfg standard.yaml lc.yaml \
    --mode eval
```

Runs all 4 combinations of agents × scenarios sequentially.

### Key Parameters

- `--exp_name` - Output saved to `log/{exp_name}/{agent}_{scenario}_seed_{seed}/`
- `--mode` - One of `eval`, `train_agent`, `train_scenario`
- `--agent_cfg` - Agent config file(s) in `safebench/agent/config/`
- `--scenario_cfg` - Scenario config file(s) in `safebench/scenario/config/`
- `--seed` - Random seed
- `--device` - `cuda:0` or `cpu`
- `--port` - CARLA server port (default 2000)
- `--save_video` - Save bird's-eye-view video
- `--num_scenario` - Number of scenarios to run in parallel per episode

### Quick Validation (single scenario)

```bash
python scripts/run.py --mode eval --agent_cfg basic.yaml --scenario_cfg standard.yaml \
    --exp_name test_quick --num_scenario 1
```

### Inspect Results

```python
import pickle
with open('log/exp_name/.../eval_results/results.pkl', 'rb') as f:
    results = pickle.load(f)  # Aggregated metrics dict
with open('log/exp_name/.../eval_results/records.pkl', 'rb') as f:
    records = pickle.load(f)  # Per-timestep records list
```

## Setup

```bash
pip install -r requirements.txt
# CARLA must be running separately:
./CarlaUE4.sh -world-port=2000
```

- Python 3.8+, PyTorch 1.13.1, CARLA 0.9.x
- TCP module: see `TCP/README.md`

## Architecture

### Three Operational Modes

1. **`eval`** - Evaluate agents against scenarios, compute metrics
2. **`train_agent`** - Train RL agent to drive; RL update after each step (off-policy) or episode (on-policy)
3. **`train_scenario`** - Train scenario policy to generate adversarial scenarios maximizing agent errors

### Agent Registry (`safebench/agent/__init__.py:AGENT_POLICY_LIST`)

| Key | Class | Type |
|-----|-------|------|
| `basic` | `CarlaBasicAgent` | unlearnable |
| `behavior` | `CarlaBehaviorAgent` | unlearnable |
| `tcp` | `TCPAgent` | unlearnable |
| `dummy` | `DummyAgent` | unlearnable |
| `sac`, `ddpg`, `ppo`, `td3` | RL algorithms | learnable |

Base class: `safebench/agent/base_policy.py:BasePolicy`. Key method: `get_action(obs, deterministic)`.

Pre-trained checkpoints in `safebench/agent/model_ckpt/`.

**TCP agent** requires a special environment: uses `VectorWrapperTCP` / `carla_env_tcp.py` instead of the standard wrappers.

### Scenario Policy Registry (`safebench/scenario/__init__.py:SCENARIO_POLICY_LIST`)

| Key | Class | Description |
|-----|-------|-------------|
| `standard`, `ordinary` | `DummyPolicy` | Static predefined scenarios |
| `advsim`, `advtraj`, `human`, `random` | `HardCodePolicy` | Hard-coded adversarial |
| `lc` | `REINFORCE` | Policy gradient adversarial generation |
| `nf` | `NormalizingFlow` | Generative model adversarial |
| `sac` | `SAC` | SAC-based adversarial generation |

Base class: `safebench/scenario/scenario_policy/base_policy.py:BasePolicy`. Key method: `get_action(obs, deterministic)`.

### Configuration System

- YAML configs in `agent/config/` and `scenario/config/` loaded via `safebench/util/run_util.py:load_config()`
- **CLI args override YAML values** (both dicts are merged: `config.update(vars(args))`)
- `CarlaRunner.__init__` copies `mode`, `ego_action_dim`, `ego_action_limit` from scenario config into agent config
- Key scenario config field: `scenario_category` — either `'planning'` (standard route scenarios) or `'perception'` (detection tasks) — controls which replay buffer and metrics function is used

### Execution Flow (`safebench/carla_runner.py:CarlaRunner`)

```
scripts/run.py
  └─ load_config(agent_cfg), load_config(scenario_cfg)
  └─ CarlaRunner(agent_config, scenario_config)
      └─ carla.Client('localhost', port)
      └─ AGENT_POLICY_LIST[type](config, logger)
      └─ SCENARIO_POLICY_LIST[type](config, logger)
  └─ runner.run()
      └─ scenario_parse() → configs grouped by map
      └─ for each map:
          ├─ _init_world(town) → sync mode, CarlaDataProvider
          ├─ _init_renderer() → pygame bird's-eye view
          ├─ VectorWrapper(num_scenario CarlaEnv instances)
          ├─ ScenarioDataLoader → non-overlapping scenario sampling
          └─ eval() or train()
```

**`eval()`**: Iterates batches, resets env, loops `agent.get_action()` + `scenario.get_action()` + `env.step()`, calls `get_route_scores()`, saves to logger.

**`train()`**: Same loop but stores transitions in `RouteReplayBuffer`. Off-policy agents update per step; on-policy agents update per episode.

### Environment Layers

1. **`CarlaEnv`** (`gym_carla/envs/carla_env.py`) - Single scenario, Gym-compatible
   - `reset(config, env_id, scenario_init_action)` → spawns actors, attaches sensors
   - `get_static_obs(config)` → routes and target speed (constant during episode)
   - `step_before_tick(action, scenario_action)` → apply controls
   - `step_after_tick()` → compute reward, check terminal conditions
   - `clean_up()` → remove all actors and sensors

2. **`VectorWrapper`** (`gym_carla/env_wrapper.py`) - Orchestrates `num_scenario` parallel envs, batches actions/observations, ticks CARLA world once per step

3. **`CarlaRunner`** - High-level training/eval loop

### Scenario Data & Loading

- Scenario configs: JSON files under `safebench/scenario/scenario_data/`
- `ScenarioDataLoader` samples non-overlapping scenarios based on `route_region` (SafeBench maps) or waypoint distance (CARLA maps)
- Each scenario config has: `data_id`, `route_id`, `trajectory`, `weather`, `scenario_type`

### Metrics (`safebench/util/metric_util.py`)

- `get_route_scores(record_dict)` for `planning` scenarios
- `get_perception_scores(record_dict)` for `perception` scenarios

Key `planning` metrics:
- `collision_rate`, `out_of_road_length`, `distance_to_route`, `incomplete_route`, `running_time`
- `penalty_score = 0.4*collision + 0.1*out_of_road + 0.1*dist_to_route + 0.3*incomplete + 0.1*time`

Output files: `results.pkl` (aggregated metrics dict), `records.pkl` (per-timestep records list)

### Extensibility Points

**Adding a new agent:**
1. Create class inheriting `BasePolicy` in `safebench/agent/`
2. Implement `get_action(obs, deterministic)` and other abstract methods
3. Add to `AGENT_POLICY_LIST` in `safebench/agent/__init__.py`
4. Create `safebench/agent/config/{name}.yaml` with `policy_type: name`

**Adding a new scenario strategy:**
1. Create class inheriting `BasePolicy` in `safebench/scenario/scenario_policy/`
2. Implement `get_action(obs, deterministic)` and `get_init_action()`
3. Add to `SCENARIO_POLICY_LIST` in `safebench/scenario/__init__.py`
4. Create `safebench/scenario/config/{name}.yaml` with `policy_type: name`

## Documentation

Comprehensive technical documentation in `doc/SafeBench_Technical_Documentation.md` (metric definitions, scenario generation algorithms, module I/O specs).
