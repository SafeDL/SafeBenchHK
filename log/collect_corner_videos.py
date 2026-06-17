import os
import re
import shutil
import pickle
from enum import Enum
from collections import Counter
from pathlib import Path

"""
脚本用途：
    对比同一批场景在 TCP 和 CarlaAgent 两套评测中的碰撞结果，筛选出任一侧
    发生碰撞/失败的 corner case，并把对应测试视频复制到统一目录。

适用场景：
    1. 原始评测结果根目录包含 TCP/scenario_01_results、carla_agent/scenario_01_results 等。
    2. 每个结果目录下包含 eval_results/records.pkl 和 video/时间戳/*.mp4。
    3. TCP 和 CarlaAgent 视频都位于 scenario_xx_results/video/时间戳/*.mp4。

筛选逻辑：
    1. 使用 records.pkl 中每条轨迹最后一帧的 collision 状态作为最终状态。
    2. 同一 video_id 下，只要 TCP 或 CarlaAgent 任意一侧状态为 COLLISION/FAILURE，
       就视为 corner case 候选。
    3. 只有 TCP 和 CarlaAgent 两侧视频都存在时才复制；缺任意一侧视频则整组跳过。
    4. 输出视频根据来源和状态追加后缀：
        *_tcp_safe.mp4
        *_tcp_risk.mp4
        *_autopilot_safe.mp4
        *_autopilot_risk.mp4

主要输出：
    log/Central/Corner/S01/*.mp4
    log/Central/Corner/S02/*.mp4
    ...
"""


PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 原始数据根目录（包含 TCP/ 和 carla_agent/）
SOURCE_BASE = "/media/hp/DATA/STFProject/Scenarios/new_ShaTin"

# corner case 视频输出目录
DEST_BASE = PROJECT_ROOT / "log" / "new_ShaTin" / "Corner"

# 默认处理 S02 ~ S08；如果只想验证单个场景，可运行：python log/collect_corner_videos.py --scenarios 1
SCENARIO_RANGE = range(1, 9)

MIN_FRAMES = 10
RISK_STATUSES = ("COLLISION", "FAILURE")
SAFE_STATUSES = ("SUCCESS", "RUNNING")
COPYABLE_STATUSES = RISK_STATUSES + SAFE_STATUSES

RAW_VIDEO_PATTERN = re.compile(r"video_\d+_id_(\d+)\.mp4$", re.IGNORECASE)

SCENE_PREFIX = {
    "S01": "DynamicCrossing",
    "S02": "VehicleTurning",
    "S03": "OtherLeadingVehicle",
    "S04": "LaneChange",
    "S05": "OppositeVehicleRunningRedLight",
    "S06": "JunctionLeftTurn",
    "S07": "JunctionRightTurn",
    "S08": "NoSignalJunctionCrossingRoute",
}

AGENTS = {
    "tcp": {
        "source_folder": "TCP",
        "safe_suffix": "_tcp_safe",
        "risk_suffix": "_tcp_risk",
    },
    "autopilot": {
        "source_folder": "carla_agent",
        "safe_suffix": "_autopilot_safe",
        "risk_suffix": "_autopilot_risk",
    },
}


class SafeBenchStatus(Enum):
    """避免 records.pkl 反序列化时依赖 shapely 等运行时模块。"""

    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"
    RUNNING = "RUNNING"
    INVALID = "INVALID"


class SafeBenchRecordsUnpickler(pickle.Unpickler):
    """只替换 records.pkl 中用到的 SafeBench Status 枚举，其余对象按默认方式加载。"""

    STATUS_MODULE = "safebench.scenario.scenario_definition.atomic_criteria"

    def find_class(self, module, name):
        if module == self.STATUS_MODULE and name == "Status":
            return SafeBenchStatus
        return super().find_class(module, name)


def parse_scenario_range(value):
    """
    解析命令行场景范围。
    支持: 1、1,3,8、1-8。
    """
    scenario_set = set()

    for part in value.split(","):
        part = part.strip()
        if not part:
            continue

        if "-" in part:
            start_text, end_text = part.split("-", 1)
            start_idx = int(start_text)
            end_idx = int(end_text)
            scenario_set.update(range(start_idx, end_idx + 1))
        else:
            scenario_set.add(int(part))

    return [idx for idx in sorted(scenario_set) if 1 <= idx <= 8]


def load_pickle_data(pkl_path):
    """读取 records.pkl，并返回反序列化后的轨迹数据。"""
    with open(pkl_path, "rb") as f:
        return SafeBenchRecordsUnpickler(f).load()


def normalize_status(status):
    """把 SafeBench Status 枚举或字符串统一归一化为可比较的状态名。"""
    status_text = str(status)

    if "COLLISION" in status_text:
        return "COLLISION"
    if "FAILURE" in status_text:
        return "FAILURE"
    if "SUCCESS" in status_text:
        return "SUCCESS"
    if "RUNNING" in status_text:
        return "RUNNING"
    if "INVALID" in status_text:
        return "INVALID"

    return status_text


def analyze_final_collision_status(data):
    """
    提取每条轨迹最后一帧的 collision 状态。
    对于帧数少于 MIN_FRAMES 的轨迹，标记为 TOO_SHORT 并跳过复制。
    """
    final_collision_status = {}

    for key, frames in data.items():
        video_id = int(key)

        if not isinstance(frames, list) or len(frames) == 0:
            final_collision_status[video_id] = "EMPTY"
            continue

        if len(frames) < MIN_FRAMES:
            final_collision_status[video_id] = "TOO_SHORT"
            continue

        last_frame = frames[-1]
        if not isinstance(last_frame, dict):
            final_collision_status[video_id] = "INVALID"
            continue

        final_collision_status[video_id] = normalize_status(last_frame.get("collision", "UNKNOWN"))

    return final_collision_status


def get_scenario_result_dir(source_base, agent_key, scenario_idx):
    scenario_name = f"scenario_{scenario_idx:02d}_results"
    return os.path.join(source_base, AGENTS[agent_key]["source_folder"], scenario_name)


def get_video_dir(source_base, agent_key, scenario_idx):
    result_dir = get_scenario_result_dir(source_base, agent_key, scenario_idx)
    return os.path.join(result_dir, "video")


def build_video_id_map(video_dir):
    """扫描视频文件夹，建立 video_id -> 文件绝对路径 的映射。"""
    video_id_map = {}

    if not os.path.isdir(video_dir):
        print(f"  video 文件夹不存在：{video_dir}")
        return video_id_map

    for root, dirnames, filenames in os.walk(video_dir):
        dirnames.sort()
        for filename in sorted(filenames):
            if not filename.lower().endswith(".mp4"):
                continue

            match = RAW_VIDEO_PATTERN.match(filename)
            if match is None:
                continue

            video_id = int(match.group(1))
            video_id_map[video_id] = os.path.join(root, filename)

    return video_id_map


def load_agent_result(source_base, agent_key, scenario_idx):
    """加载单个 agent 在单个 scenario 下的状态和视频映射。"""
    result_dir = get_scenario_result_dir(source_base, agent_key, scenario_idx)
    pkl_path = os.path.join(result_dir, "eval_results", "records.pkl")
    video_dir = get_video_dir(source_base, agent_key, scenario_idx)

    if not os.path.exists(pkl_path):
        print(f"  records.pkl 不存在：{pkl_path}")
        return {}, {}

    data = load_pickle_data(pkl_path)
    status_dict = analyze_final_collision_status(data)
    video_id_map = build_video_id_map(video_dir)

    return status_dict, video_id_map


def is_risk_status(status):
    return status in RISK_STATUSES


def is_copyable_status(status):
    return status in COPYABLE_STATUSES


def get_status_suffix(agent_key, status):
    agent = AGENTS[agent_key]
    if is_risk_status(status):
        return agent["risk_suffix"]
    return agent["safe_suffix"]


def format_video_id(video_id):
    return f"{int(video_id):04d}"


def build_dest_filename(scenario_tag, video_id, suffix):
    prefix = SCENE_PREFIX[scenario_tag]
    return f"{prefix}_{format_video_id(video_id)}{suffix}.mp4"


def copy_corner_video(source_path, dest_path, dry_run=False):
    if dry_run:
        return

    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    shutil.copy2(source_path, dest_path)


def build_pair_copy_plan(
    scenario_tag,
    scenario_dest_dir,
    video_id,
    tcp_status,
    autopilot_status,
    tcp_videos,
    autopilot_videos,
):
    pair = (
        ("tcp", tcp_status.get(video_id, "MISSING"), tcp_videos.get(video_id)),
        ("autopilot", autopilot_status.get(video_id, "MISSING"), autopilot_videos.get(video_id)),
    )

    if any(not is_copyable_status(status) for _, status, _ in pair):
        return None, "invalid_status"
    if any(source_path is None for _, _, source_path in pair):
        return None, "missing_video"

    copy_plan = []
    for agent_key, status, source_path in pair:
        suffix = get_status_suffix(agent_key, status)
        dest_filename = build_dest_filename(scenario_tag, video_id, suffix)
        dest_path = os.path.join(scenario_dest_dir, dest_filename)
        copy_plan.append((source_path, dest_path))

    return copy_plan, None


def process_scenario(scenario_idx, source_base, dest_base, dry_run=False):
    """
    处理单个 scenario：
    1. 分别读取 TCP / CarlaAgent 状态。
    2. 找出任意一侧碰撞/失败的 video_id。
    3. 只复制两侧视频都存在的成对 corner case。
    """
    scenario_tag = f"S{scenario_idx:02d}"
    scenario_dest_dir = os.path.join(str(dest_base), scenario_tag)

    tcp_status, tcp_videos = load_agent_result(source_base, "tcp", scenario_idx)
    autopilot_status, autopilot_videos = load_agent_result(source_base, "autopilot", scenario_idx)

    if not tcp_status and not autopilot_status:
        print(f"{scenario_tag}: 没有可用状态数据，跳过")
        return Counter()

    all_video_ids = sorted(set(tcp_status.keys()) | set(autopilot_status.keys()))
    corner_ids = [
        video_id for video_id in all_video_ids
        if is_risk_status(tcp_status.get(video_id, "MISSING"))
        or is_risk_status(autopilot_status.get(video_id, "MISSING"))
    ]

    tcp_risk_ids = {video_id for video_id, status in tcp_status.items() if is_risk_status(status)}
    autopilot_risk_ids = {video_id for video_id, status in autopilot_status.items() if is_risk_status(status)}

    summary = Counter()
    summary["corner_triggered"] = len(corner_ids)
    summary["tcp_records"] = len(tcp_status)
    summary["autopilot_records"] = len(autopilot_status)
    summary["tcp_videos"] = len(tcp_videos)
    summary["autopilot_videos"] = len(autopilot_videos)
    summary["tcp_only_risk"] = len(tcp_risk_ids - autopilot_risk_ids)
    summary["autopilot_only_risk"] = len(autopilot_risk_ids - tcp_risk_ids)
    summary["both_risk"] = len(tcp_risk_ids & autopilot_risk_ids)

    for video_id in corner_ids:
        copy_plan, skip_reason = build_pair_copy_plan(
            scenario_tag,
            scenario_dest_dir,
            video_id,
            tcp_status,
            autopilot_status,
            tcp_videos,
            autopilot_videos,
        )

        if copy_plan is None:
            summary[skip_reason] += 1
            continue

        for source_path, dest_path in copy_plan:
            copy_corner_video(source_path, dest_path, dry_run=dry_run)
        summary["corner_pairs"] += 1
        summary["videos_copied"] += len(copy_plan)

    action_text = "可复制" if dry_run else "已复制"
    print(
        f"{scenario_tag}: 触发 {summary['corner_triggered']} 个，"
        f"实际找到成对 corner case {summary['corner_pairs']} 个，"
        f"{action_text}视频 {summary['videos_copied']} 个"
    )
    print(
        f"  记录/视频: TCP {summary['tcp_records']}/{summary['tcp_videos']}，"
        f"CarlaAgent {summary['autopilot_records']}/{summary['autopilot_videos']}；"
        f"仅TCP风险 {summary['tcp_only_risk']}，仅CarlaAgent风险 {summary['autopilot_only_risk']}，"
        f"双方风险 {summary['both_risk']}"
    )
    if summary["missing_video"] or summary["invalid_status"]:
        print(
            f"  跳过未成对: 缺视频 {summary['missing_video']} 个，"
            f"状态不可复制 {summary['invalid_status']} 个"
        )

    return summary


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(
        description="Collect corner-case videos where TCP or CarlaAgent has collision/failure."
    )
    parser.add_argument(
        "--source-base",
        default=SOURCE_BASE,
        help="原始测试结果根目录，需包含 TCP/ 和 carla_agent/。",
    )
    parser.add_argument(
        "--dest-base",
        default=str(DEST_BASE),
        help="corner 视频输出目录。",
    )
    parser.add_argument(
        "--scenarios",
        default=f"{SCENARIO_RANGE.start}-{SCENARIO_RANGE.stop - 1}",
        help="要处理的场景编号，例如 1、1,3,8、1-8。",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只打印复制计划，不实际复制文件。",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    scenario_range = parse_scenario_range(args.scenarios)

    if not scenario_range:
        raise ValueError(f"未解析到有效场景编号: {args.scenarios}")

    total_summary = Counter()

    print(f"source_base: {args.source_base}")
    print(f"dest_base: {args.dest_base}")
    print(f"scenarios: {', '.join(f'S{idx:02d}' for idx in scenario_range)}")
    print(f"dry_run: {args.dry_run}")

    for idx in scenario_range:
        total_summary.update(process_scenario(idx, args.source_base, args.dest_base, dry_run=args.dry_run))

    action_text = "可复制" if args.dry_run else "复制"
    print("\n处理完成")
    print(f"触发 corner case: {total_summary['corner_triggered']} 个")
    print(f"实际找到成对 corner case: {total_summary['corner_pairs']} 个")
    print(f"{action_text}视频: {total_summary['videos_copied']} 个")


if __name__ == "__main__":
    main()
