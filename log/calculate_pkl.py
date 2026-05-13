import os
import re
import shutil
import pickle
from collections import Counter

"""
脚本用途：
    批量读取 SafeBench/CARLA 评测输出中的 records.pkl，根据每条轨迹最后一帧的
    collision 状态判断该轨迹对应的视频属于 Safe 还是 Risk，并复制到统一目录。

适用场景：
    1. 一次评测已经生成 scenario_01_results、scenario_02_results 等结果目录。
    2. 每个结果目录下包含 eval_results/records.pkl 和 video/时间戳/*.mp4。
    3. 希望按场景编号整理出 S01/Safe、S01/Risk 这样的人工复核视频集。

主要输入：
    source_base: 原始评测结果根目录。
    dest_base: 分类后视频的目标根目录。
    records.pkl: 字典结构，key 通常对应视频 id，value 为该轨迹逐帧记录。

主要输出：
    dest_base/Sxx/Safe/*.mp4
    dest_base/Sxx/Risk/*.mp4
"""


def load_pickle_data(pkl_path):
    """读取 records.pkl，并返回反序列化后的轨迹数据。"""
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    return data


# 少于此帧数的轨迹通常是不完整评测或异常中断结果，跳过可减少误分类。
MIN_FRAMES = 10


def analyze_final_collision_status(data):
    """
    提取每条轨迹最后一帧的 collision 状态。
    对于帧数少于 MIN_FRAMES 的轨迹，标记为 'TOO_SHORT' 并跳过分类。
    """
    final_collision_status = {}

    for key, frames in data.items():
        # records.pkl 中每个 key 对应一条轨迹；value 应该是按时间排列的帧列表。
        if not isinstance(frames, list) or len(frames) == 0:
            final_collision_status[key] = 'EMPTY'
            continue

        # 过短场景跳过分类
        if len(frames) < MIN_FRAMES:
            print(f"  ⏭️  轨迹 {key} 仅有 {len(frames)} 帧（< {MIN_FRAMES}），已跳过分类")
            final_collision_status[key] = 'TOO_SHORT'
            continue

        # 只使用最后一帧状态作为最终评测结果，避免中间 RUNNING 状态影响分类。
        last_frame = frames[-1]
        status = str(last_frame.get('collision', 'UNKNOWN'))

        if 'RUNNING' in status:
            final_collision_status[key] = 'RUNNING'
        elif 'COLLISION' in status:
            final_collision_status[key] = 'COLLISION'
        elif 'SUCCESS' in status:
            final_collision_status[key] = 'SUCCESS'
        elif 'FAILURE' in status:
            final_collision_status[key] = 'FAILURE'
        else:
            final_collision_status[key] = status  # 其他枚举或未知状态

    return final_collision_status


def print_status_summary(status_dict):
    """打印最终状态统计，便于快速检查本轮评测是否存在异常失败。"""
    # 先定位 FAILURE 轨迹；当前只保留入口，若需要可在 pass 处打印 key。
    print("\n❗ 所有最终状态为 FAILURE 的轨迹：")
    failure_keys = [k for k, v in status_dict.items() if v == 'FAILURE']
    if failure_keys:
        for k in failure_keys:
            pass
    else:
        print("没有 FAILURE 状态的轨迹。")

    # 汇总 SUCCESS、COLLISION、FAILURE、RUNNING、TOO_SHORT、EMPTY 等状态数量。
    print("\n📊 各状态数量统计：")
    status_counts = Counter(status_dict.values())
    for status, count in status_counts.items():
        print(f"{status}: {count} 条")


def build_video_id_map(video_dir):
    """
    扫描 video 文件夹下所有时间戳子文件夹，建立 video_id -> 文件绝对路径 的映射。
    视频文件名格式: video_XXXX_id_YYYY.mp4, 其中 YYYY 是 video_id。
    """
    video_id_map = {}
    pattern = re.compile(r'video_\d+_id_(\d+)\.mp4')

    if not os.path.isdir(video_dir):
        print(f"⚠️  video 文件夹不存在：{video_dir}")
        return video_id_map

    # 遍历所有时间戳子文件夹
    for timestamp_folder in sorted(os.listdir(video_dir)):
        folder_path = os.path.join(video_dir, timestamp_folder)
        if not os.path.isdir(folder_path):
            continue
        for filename in os.listdir(folder_path):
            # key 使用文件名中的 id 字段，而不是 video_ 后面的序号。
            match = pattern.match(filename)
            if match:
                video_id = int(match.group(1))
                video_id_map[video_id] = os.path.join(folder_path, filename)

    return video_id_map


def copy_videos_by_collision(status_dict, video_id_map, safe_dir, risk_dir):
    """
    根据碰撞状态将视频复制到对应的安全/危险路径下。
    危险 (Risk): COLLISION 或 FAILURE
    安全 (Safe): SUCCESS、RUNNING 及其他
    """
    os.makedirs(safe_dir, exist_ok=True)
    os.makedirs(risk_dir, exist_ok=True)

    safe_count = 0
    risk_count = 0
    missing_count = 0
    skipped_count = 0

    for key in sorted(status_dict.keys()):
        status = status_dict[key]

        # 跳过无效/过短场景，不进行视频复制
        if status in ('TOO_SHORT', 'EMPTY'):
            skipped_count += 1
            continue

        video_path = video_id_map.get(key)

        if video_path is None:
            missing_count += 1
            continue

        filename = os.path.basename(video_path)

        # COLLISION 和 FAILURE 都视为危险场景
        if status in ('COLLISION', 'FAILURE'):
            dest = os.path.join(risk_dir, filename)
            risk_count += 1
        else:
            dest = os.path.join(safe_dir, filename)
            safe_count += 1

        shutil.copy2(video_path, dest)

    print(f"\n📁 视频分类复制完成：")
    print(f"  ✅ Safe: {safe_count} 个视频 → {safe_dir}")
    print(f"  ⚠️  Risk: {risk_count} 个视频 → {risk_dir}")
    if skipped_count > 0:
        print(f"  ⏭️  已跳过 (TOO_SHORT/EMPTY): {skipped_count} 个轨迹")
    if missing_count > 0:
        print(f"  ❌ 未找到对应视频: {missing_count} 个轨迹")


def process_scenario(scenario_idx, source_base, dest_base):
    """
    处理单个场景：加载 records.pkl，分析碰撞，复制视频到 Safe/Risk。
    scenario_idx: 场景编号 (1-8)
    """
    scenario_name = f"scenario_{scenario_idx:02d}_results"
    scenario_tag = f"S{scenario_idx:02d}"

    # 源目录遵循 SafeBench 默认输出结构：scenario_xx_results/eval_results + video。
    source_dir = os.path.join(source_base, scenario_name)
    pkl_path = os.path.join(source_dir, "eval_results", "records.pkl")
    video_dir = os.path.join(source_dir, "video")

    # 目标目录按场景编号和风险类别拆分，方便后续人工筛查或数据集制作。
    safe_dir = os.path.join(dest_base, scenario_tag, "Safe")
    risk_dir = os.path.join(dest_base, scenario_tag, "Risk")

    print(f"\n{'='*60}")
    print(f"🔄 处理场景 {scenario_tag} ({scenario_name})")
    print(f"{'='*60}")

    if not os.path.exists(pkl_path):
        print(f"  ❌ 文件不存在：{pkl_path}，跳过")
        return

    # 1. 加载轨迹数据
    print(f"  ⏳ 正在加载 {pkl_path} ...")
    data = load_pickle_data(pkl_path)
    print(f"  ✅ 数据类型：{type(data)}，共包含 {len(data)} 条轨迹")

    # 2. 分析碰撞状态
    final_status = analyze_final_collision_status(data)
    print_status_summary(final_status)

    # 3. 构建 video_id → 文件路径 映射
    print(f"\n  🔍 正在扫描 video 文件夹：{video_dir}")
    video_id_map = build_video_id_map(video_dir)
    print(f"  ✅ 共找到 {len(video_id_map)} 个视频文件")

    # 4. 根据碰撞状态复制视频
    copy_videos_by_collision(final_status, video_id_map, safe_dir, risk_dir)


if __name__ == "__main__":
    # ========== 配置路径 ==========
    # 原始数据根目录（包含 scenario_01_results ~ scenario_08_results）
    source_base = "/media/hp/DATA/ShaTin/first_round_of_TCP"
    # 目标根目录（下设 S01~S08，每个含 Safe/ 和 Risk/）
    dest_base = "/home/hp/STF/SafeBenchHK/SafeBenchHK/log/ShaTin/TCP"
    # 要处理的场景编号范围
    scenario_range = range(1,8)  # 当前会处理 S01 ~ S07；若要包含 S08，应改为 range(1, 9)。
    # ==============================

    for idx in scenario_range:
        process_scenario(idx, source_base, dest_base)

    print(f"\n{'='*60}")
    print("🎉 所有场景处理完毕！")
    print(f"{'='*60}")
