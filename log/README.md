# log

`log/` 用于存放 SafeBenchHK/CARLA 评测输出，以及整理评测视频的辅助脚本。真实评测产生的视频、结果目录体积较大，默认由 `.gitignore` 忽略；仓库只保留脚本和目录说明。

## 文件说明

- `collect_corner_videos.py`: 对比同一批场景在 TCP 和 CarlaAgent 两套评测中的最终碰撞状态，将成对视频复制并分类到 `Safe`、`Corner`、`Risk`。
- `change_name.py`: 对整理后的 `.mp4` 视频做原地批量重命名，统一为场景名前缀加连续编号的格式。
- `template/`: 期望输出结构示例，实际视频结果不建议提交到 Git。

## 典型目录结构

```text
log/<map_name>/
  Safe/S01/*.mp4
  Corner/S01/*.mp4
  Risk/S01/*.mp4
```

`collect_corner_videos.py` 依赖 `records.pkl` 和原始视频目录；`change_name.py` 会直接修改文件名，运行前先确认脚本顶部或命令行参数中的路径指向正确结果目录。
