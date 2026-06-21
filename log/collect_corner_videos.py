import argparse
import pickle
import re
import shutil
from collections import Counter
from enum import Enum
from pathlib import Path

"""
脚本用途：
    对比同一批场景在 TCP 和 CarlaAgent 两套评测中的最终碰撞状态，并把成对视频
    按 Safe / Corner / Risk 三类复制到地图输出目录。

分类定义：
    Safe:   TCP 和 CarlaAgent 都没有碰撞/失败。
    Corner: 只有其中一方发生碰撞/失败。
    Risk:   TCP 和 CarlaAgent 都发生碰撞/失败。

目录结构：
    log/Central/Safe/S01/*.mp4
    log/Central/Corner/S01/*.mp4
    log/Central/Risk/S01/*.mp4

命名规则：
    Safe / Risk:
        {scene_prefix}_{video_id}_tcp.mp4
        {scene_prefix}_{video_id}_autopilot.mp4
    Corner:
        {scene_prefix}_{video_id}_tcp_safe.mp4
        {scene_prefix}_{video_id}_tcp_risk.mp4
        {scene_prefix}_{video_id}_autopilot_safe.mp4
        {scene_prefix}_{video_id}_autopilot_risk.mp4
"""


PROJECT_ROOT = Path(__file__).resolve().parents[1]

SOURCE_BASE = Path("/media/hp/DATA/STFProject/Scenarios/new_ShaTin")
DEST_BASE = PROJECT_ROOT / "log" / "new_shatin"
SCENARIO_RANGE = range(1, 9)

MIN_FRAMES = 10
CATEGORIES = ("Safe", "Corner", "Risk")
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

AGENT_FOLDERS = {
    "tcp": "TCP",
    "autopilot": "carla_agent",
}
AGENT_SUFFIXES = {
    "tcp": "_tcp",
    "autopilot": "_autopilot",
}


class SafeBenchStatus(Enum):
    """避免 records.pkl 反序列化时依赖 shapely 等运行时模块。"""

    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"
    RUNNING = "RUNNING"
    INVALID = "INVALID"


class SafeBenchRecordsUnpickler(pickle.Unpickler):
    STATUS_MODULE = "safebench.scenario.scenario_definition.atomic_criteria"

    def find_class(self, module, name):
        if module == self.STATUS_MODULE and name == "Status":
            return SafeBenchStatus
        return super().find_class(module, name)


RISK_STATUSES = {"COLLISION", SafeBenchStatus.FAILURE.name}
SAFE_STATUSES = {SafeBenchStatus.SUCCESS.name, SafeBenchStatus.RUNNING.name}
COPYABLE_STATUSES = RISK_STATUSES | SAFE_STATUSES
KNOWN_STATUSES = COPYABLE_STATUSES | {SafeBenchStatus.INVALID.name}


def parse_scenarios(value):
    scenario_set = set()
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_idx, end_idx = (int(item) for item in part.split("-", 1))
            scenario_set.update(range(start_idx, end_idx + 1))
        else:
            scenario_set.add(int(part))

    return [idx for idx in sorted(scenario_set) if 1 <= idx <= 8]


def load_pickle_data(pkl_path):
    with open(pkl_path, "rb") as f:
        return SafeBenchRecordsUnpickler(f).load()


def normalize_status(status):
    status_name = getattr(status, "name", None)
    if status_name in KNOWN_STATUSES:
        return status_name

    status_text = str(status)
    for known_status in KNOWN_STATUSES:
        if known_status in status_text:
            return known_status
    return status_text


def analyze_final_collision_status(data):
    final_status = {}

    for key, frames in data.items():
        video_id = int(key)
        if not isinstance(frames, list) or not frames:
            final_status[video_id] = "EMPTY"
            continue
        if len(frames) < MIN_FRAMES:
            final_status[video_id] = "TOO_SHORT"
            continue
        if not isinstance(frames[-1], dict):
            final_status[video_id] = "INVALID"
            continue

        final_status[video_id] = normalize_status(frames[-1].get("collision", "UNKNOWN"))

    return final_status


def get_scenario_result_dir(source_base, agent_key, scenario_idx):
    scenario_name = f"scenario_{scenario_idx:02d}_results"
    return source_base / AGENT_FOLDERS[agent_key] / scenario_name


def build_video_id_map(video_dir):
    video_id_map = {}

    if not video_dir.is_dir():
        print(f"  video 文件夹不存在：{video_dir}")
        return video_id_map

    for path in sorted(video_dir.rglob("*.mp4")):
        match = RAW_VIDEO_PATTERN.match(path.name)
        if match:
            video_id_map[int(match.group(1))] = path

    return video_id_map


def load_agent_result(source_base, agent_key, scenario_idx):
    result_dir = get_scenario_result_dir(source_base, agent_key, scenario_idx)
    pkl_path = result_dir / "eval_results" / "records.pkl"
    video_dir = result_dir / "video"

    if not pkl_path.exists():
        print(f"  records.pkl 不存在：{pkl_path}")
        return {}, {}

    return analyze_final_collision_status(load_pickle_data(pkl_path)), build_video_id_map(video_dir)


def classify_pair(tcp_status, autopilot_status):
    if tcp_status not in COPYABLE_STATUSES or autopilot_status not in COPYABLE_STATUSES:
        return None

    risk_count = int(tcp_status in RISK_STATUSES) + int(autopilot_status in RISK_STATUSES)
    if risk_count == 2:
        return "Risk"
    if risk_count == 1:
        return "Corner"
    return "Safe"


def build_dest_filename(scenario_tag, video_id, agent_key, status, category):
    suffix = AGENT_SUFFIXES[agent_key]
    if category == "Corner":
        suffix += "_risk" if status in RISK_STATUSES else "_safe"

    return f"{SCENE_PREFIX[scenario_tag]}_{int(video_id):04d}{suffix}.mp4"


def build_pair_copy_plan(scenario_tag, dest_base, category, video_id, pair):
    scenario_dest_dir = dest_base / category / scenario_tag
    copy_plan = []

    for agent_key, status, source_path in pair:
        dest_filename = build_dest_filename(scenario_tag, video_id, agent_key, status, category)
        copy_plan.append((source_path, scenario_dest_dir / dest_filename))

    return copy_plan


def copy_videos(copy_plan):
    for source_path, dest_path in copy_plan:
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, dest_path)


def ensure_category_dirs(dest_base, scenario_tag):
    for category in CATEGORIES:
        (dest_base / category / scenario_tag).mkdir(parents=True, exist_ok=True)


def process_scenario(scenario_idx, source_base, dest_base):
    scenario_tag = f"S{scenario_idx:02d}"
    ensure_category_dirs(dest_base, scenario_tag)

    tcp_status, tcp_videos = load_agent_result(source_base, "tcp", scenario_idx)
    autopilot_status, autopilot_videos = load_agent_result(source_base, "autopilot", scenario_idx)

    if not tcp_status and not autopilot_status:
        print(f"{scenario_tag}: 没有可用状态数据，跳过")
        return Counter()

    summary = Counter()

    all_video_ids = sorted(set(tcp_status) | set(autopilot_status))
    for video_id in all_video_ids:
        pair = (
            ("tcp", tcp_status.get(video_id, "MISSING"), tcp_videos.get(video_id)),
            ("autopilot", autopilot_status.get(video_id, "MISSING"), autopilot_videos.get(video_id)),
        )
        category = classify_pair(pair[0][1], pair[1][1])

        if category is None:
            summary["invalid_status"] += 1
            continue
        if any(source_path is None for _, _, source_path in pair):
            summary["missing_video"] += 1
            continue

        copy_plan = build_pair_copy_plan(scenario_tag, dest_base, category, video_id, pair)
        copy_videos(copy_plan)
        summary[category] += 1
        summary["paired_cases"] += 1
        summary["videos_copied"] += len(copy_plan)

    print(
        f"{scenario_tag}: Safe {summary['Safe']}，Corner {summary['Corner']}，"
        f"Risk {summary['Risk']}；已复制视频 {summary['videos_copied']} 个"
    )
    if summary["invalid_status"] or summary["missing_video"]:
        print(
            f"  跳过: 状态不可复制 {summary['invalid_status']}，"
            f"缺视频 {summary['missing_video']}"
        )

    return summary


def parse_args():
    parser = argparse.ArgumentParser(
        description="Classify paired TCP/CarlaAgent videos into Safe, Corner, and Risk."
    )
    parser.add_argument(
        "--source-base",
        type=Path,
        default=SOURCE_BASE,
        help="原始测试结果根目录，需包含 TCP/ 和 carla_agent/。",
    )
    parser.add_argument(
        "--dest-base",
        type=Path,
        default=DEST_BASE,
        help="地图输出根目录，脚本会在其下创建 Safe/Corner/Risk/Sxx。",
    )
    parser.add_argument(
        "--scenarios",
        default=f"{SCENARIO_RANGE.start}-{SCENARIO_RANGE.stop - 1}",
        help="场景编号，例如 1、1,3,8、1-8。",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    scenario_range = parse_scenarios(args.scenarios)
    if not scenario_range:
        raise ValueError(f"未解析到有效场景编号: {args.scenarios}")

    total = Counter()
    print(f"source_base: {args.source_base}")
    print(f"dest_base: {args.dest_base}")
    print(f"scenarios: {', '.join(f'S{idx:02d}' for idx in scenario_range)}")

    for scenario_idx in scenario_range:
        total.update(process_scenario(scenario_idx, args.source_base, args.dest_base))

    print("\n处理完成")
    print(f"Safe: {total['Safe']} 个")
    print(f"Corner: {total['Corner']} 个")
    print(f"Risk: {total['Risk']} 个")
    print(f"成对 case: {total['paired_cases']} 个")
    print(f"已复制视频: {total['videos_copied']} 个")
    print(f"跳过: 状态不可复制 {total['invalid_status']} 个，缺视频 {total['missing_video']} 个")


if __name__ == "__main__":
    main()
