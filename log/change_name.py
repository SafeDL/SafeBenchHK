import os
import re

"""
脚本用途：
    对某一次评测生成的视频目录进行原地批量重命名。

命名逻辑：
    从原始文件名最后一个下划线后的数字中提取编号，例如：
        video_0000_id_0000.mp4 -> 0000
    再拼接为：
        {prefix}_0000.mp4

注意事项：
    1. 该脚本会直接修改 video_folder 内的原文件名，运行前建议备份或确认路径。
    2. prefix 需要在下方按场景类型手动启用一个，否则脚本会因为 prefix 未定义而报错。
    3. 若目标文件名已存在，os.rename 可能覆盖或失败，取决于操作系统行为。
"""

# 配置参数：先指定实验目录和视频所在的时间戳目录。
exp_folder = "exp07tcp/exp_tcp_standard_seed_0"  # 可以修改这个路径
video_folder = os.path.join(exp_folder, "video/2026-01-12_11-50-53")  # 视频所在文件夹

# 场景名前缀：运行前只保留一个有效 prefix，用于表示该批视频所属场景类型。
# prefix = "DynamicCrossing"  # 01
# prefix = "VehicleTurning"  # 02
# prefix = "OtherLeadingVehicle"  # 03
# prefix = "LaneChange"  # 04
# prefix = "OppositeVehicleRunningRedLight"  # 05
# prefix = "JunctionLeftTurn"  # 06
# prefix = "JunctionRightTurn"  # 07]
# prefix = "NoSignalJunctionCrossingRoute" # 08


# 获取待处理目录下所有 mp4 文件；非 mp4 文件不会参与重命名。
video_files = [f for f in os.listdir(video_folder) if f.endswith('.mp4')]

# 逐个重命名文件。
for old_name in video_files:
    # 提取最后一个_后面的编号，例如 video_0000_id_0000.mp4 -> 0000.mp4
    match = re.search(r'_(\d+)\.mp4$', old_name)
    if match:
        number = match.group(1)  # 获取编号，如 "0000"
        new_name = f"{prefix}_{number}.mp4"

        # old_path 是原始视频路径，new_path 是同目录下的新视频路径。
        old_path = os.path.join(video_folder, old_name)
        new_path = os.path.join(video_folder, new_name)

        print(f"重命名: {old_name} -> {new_name}")
        os.rename(old_path, new_path)

print(f"\n完成！共重命名 {len(video_files)} 个文件")
