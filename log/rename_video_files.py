import os
import shutil

"""
脚本用途：
    将某个目录中的 CARLA 评测视频按 id 顺序整理到新目录中，并统一重命名。

典型输入文件名：
    video_0000_id_0000.mp4
    video_0001_id_0007.mp4

处理流程：
    1. 扫描 input_folder 下的所有 mp4 文件。
    2. 从文件名中的 id 字段提取轨迹编号。
    3. 按 id 从小到大排序，并对完全重复的 (id, 文件名) 记录去重。
    4. 按排序后的顺序改名为 NoSignalJunctionCrossingRoute_0000.mp4 等。
    5. 使用 shutil.move 移动到 output_folder，原目录中的对应文件会被移走。

注意事项：
    该脚本当前把新文件名前缀固定为 NoSignalJunctionCrossingRoute。
    如需处理其他场景，可把 new_file_name 中的前缀改成 scenario_name_list 中对应名称，
    或进一步改造成函数参数。
"""


def rename_and_move_videos(input_folder, output_folder):
    """
    读取指定文件夹下的 mp4 文件，根据 id 后的序号排序并去重，重命名后存入新的文件夹。

    Args:
        input_folder (str): 输入文件夹路径，包含原始 mp4 文件。
        output_folder (str): 输出文件夹路径，用于存储重命名后的 mp4 文件。
    """
    # 检查输入文件夹是否存在
    if not os.path.exists(input_folder):
        raise FileNotFoundError(f"输入文件夹不存在: {input_folder}")

    # 创建输出文件夹（如果不存在）
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 获取所有 mp4 文件；其他文件会被忽略。
    files = [f for f in os.listdir(input_folder) if f.endswith(".mp4")]

    # 提取文件名中的 id 序号
    video_data = []
    for file in files:
        try:
            # 期望格式为 video_XXXX_id_YYYY.mp4；YYYY 用于排序。
            parts = file.split("_")
            if len(parts) >= 4 and parts[0] == "video" and parts[2] == "id":
                id_number = int(parts[3].split(".")[0])  # 提取 id 序号
                video_data.append((id_number, file))  # 保存序号和文件名
        except (ValueError, IndexError):
            # 跳过无法解析的文件
            continue

    # 对序号进行排序并去重，确保输出文件名编号连续且顺序稳定。
    video_data = sorted(set(video_data), key=lambda x: x[0])

    # 重命名并移动文件
    for idx, (_, original_file) in enumerate(video_data):
        # 构造新的文件名；idx 是整理后的连续编号，不一定等于原始 id。
        new_file_name = f"NoSignalJunctionCrossingRoute_{idx:04d}.mp4"
        original_path = os.path.join(input_folder, original_file)
        new_path = os.path.join(output_folder, new_file_name)

        # 移动并重命名文件；shutil.move 会让原文件从 input_folder 消失。
        shutil.move(original_path, new_path)
        print(f"文件重命名并移动: {original_file} -> {new_file_name}")

    print(f"所有文件已处理完成，共处理 {len(video_data)} 个文件。")


# 示例调用
if __name__ == "__main__":
    input_folder = "./video"  # 原始文件夹路径
    output_folder = "./scenario_08_videos"  # 输出文件夹路径

    # SafeBench 八类场景名称，当前脚本仅用作人工参考。
    scenario_name_list = ["DynamicCrossing", "VehicleTurning", "OtherLeadingVehicle",
                          "LaneChange", "OppositeVehicleRunningRedLight", "JunctionLeftTurn",
                          "JunctionRightTurn", "NoSignalJunctionCrossingRoute"]

    rename_and_move_videos(input_folder, output_folder)
