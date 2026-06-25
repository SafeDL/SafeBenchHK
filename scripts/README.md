# scripts

`scripts/` 保存项目级运行入口。目前核心入口是 `run.py`，用于启动 SafeBenchHK 的训练或评测流程。

## run.py

功能：

- 解析实验名、运行模式、agent 配置、scenario 配置、CARLA 端口和设备等命令行参数。
- 加载 `safebench/agent/config/` 与 `safebench/scenario/config/` 下的 YAML 配置。
- 将命令行参数覆盖到配置中，并实例化 `safebench.carla_runner.CarlaRunner`。
- 支持批量组合多个 agent 配置和 scenario 配置，逐组运行并汇总失败信息。

示例：

```bash
python scripts/run.py \
  --mode eval \
  --agent_cfg tcp.yaml \
  --scenario_cfg standard.yaml \
  --port 2000 \
  --tm_port 8000
```

运行前需要先启动匹配版本的 CARLA server，并确认配置文件中的地图、路线、模型权重路径可用。
