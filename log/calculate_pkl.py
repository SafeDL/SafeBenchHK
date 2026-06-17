import argparse
import os
import pickle
import re
import shutil
from collections import Counter

"""
脚本用途：
    批量读取 SafeBench/CARLA 评测输出中的 records.pkl，根据每条轨迹最后一帧的
    collision 状态判断视频属于 Safe 还是 Risk，并复制到统一目录。

默认输入：
    /media/hp/DATA/STFProject/Scenarios/new_ShaTin/TCP/scenario_xx_results

默认输出：
    log/new_ShaTin/TCP/Sxx/Safe/*.mp4
    log/new_ShaTin/TCP/Sxx/Risk/*.mp4
"""


SOURCE_BASE = "/media/hp/DATA/STFProject/Scenarios/new_ShaTin/TCP"
DEST_BASE = "/home/hp/STF/SafeBenchHK/SafeBenchHK/log/new_ShaTin/TCP"
SCENARIO_RANGE = range(1, 9)

MIN_FRAMES = 10
RISK_STATUSES = ("COLLISION", "FAILURE")
SAFE_STATUSES = ("SUCCESS", "RUNNING")
SKIPPED_STATUSES = ("TOO_SHORT", "EMPTY", "INVALID")
RAW_VIDEO_PATTERN = re.compile(r"video_\d+_id_(\d+)\.mp4$", re.IGNORECASE)


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
        return pickle.load(f)


def normalize_status(status):
    status_text = str(status)
    for known_status in RISK_STATUSES + SAFE_STATUSES + ("INVALID",):
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


def build_video_id_map(video_dir):
    video_id_map = {}

    if not os.path.isdir(video_dir):
        return video_id_map

    for root, dirnames, filenames in os.walk(video_dir):
        dirnames.sort()
        for filename in sorted(filenames):
            match = RAW_VIDEO_PATTERN.match(filename)
            if match:
                video_id_map[int(match.group(1))] = os.path.join(root, filename)

    return video_id_map


def copy_videos_by_collision(status_dict, video_id_map, safe_dir, risk_dir, dry_run=False):
    summary = Counter()

    if not dry_run:
        os.makedirs(safe_dir, exist_ok=True)
        os.makedirs(risk_dir, exist_ok=True)

    for video_id, status in sorted(status_dict.items()):
        if status in SKIPPED_STATUSES:
            summary["skipped"] += 1
            continue

        video_path = video_id_map.get(video_id)
        if video_path is None:
            summary["missing"] += 1
            continue

        if status in RISK_STATUSES:
            dest_dir = risk_dir
            summary["risk"] += 1
        else:
            dest_dir = safe_dir
            summary["safe"] += 1

        if not dry_run:
            shutil.copy2(video_path, os.path.join(dest_dir, os.path.basename(video_path)))

    return summary


def process_scenario(scenario_idx, source_base, dest_base, dry_run=False):
    scenario_name = f"scenario_{scenario_idx:02d}_results"
    scenario_tag = f"S{scenario_idx:02d}"
    source_dir = os.path.join(source_base, scenario_name)
    pkl_path = os.path.join(source_dir, "eval_results", "records.pkl")
    video_dir = os.path.join(source_dir, "video")

    if not os.path.exists(pkl_path):
        print(f"{scenario_tag}: records.pkl 不存在，跳过")
        return Counter(missing_records=1)

    status_dict = analyze_final_collision_status(load_pickle_data(pkl_path))
    video_id_map = build_video_id_map(video_dir)
    summary = copy_videos_by_collision(
        status_dict,
        video_id_map,
        os.path.join(dest_base, scenario_tag, "Safe"),
        os.path.join(dest_base, scenario_tag, "Risk"),
        dry_run=dry_run,
    )

    action_text = "可复制" if dry_run else "已复制"
    print(
        f"{scenario_tag}: 记录 {len(status_dict)} 个，视频 {len(video_id_map)} 个；"
        f"{action_text} Risk {summary['risk']} 个，Safe {summary['safe']} 个，"
        f"跳过 {summary['skipped']} 个，缺视频 {summary['missing']} 个"
    )
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description="Classify SafeBench videos into Safe/Risk folders.")
    parser.add_argument("--source-base", default=SOURCE_BASE)
    parser.add_argument("--dest-base", default=DEST_BASE)
    parser.add_argument(
        "--scenarios",
        default=f"{SCENARIO_RANGE.start}-{SCENARIO_RANGE.stop - 1}",
        help="场景编号，例如 1、1,3,8、1-8。",
    )
    parser.add_argument("--dry-run", action="store_true", help="只统计，不复制视频。")
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
    print(f"dry_run: {args.dry_run}")

    for scenario_idx in scenario_range:
        total.update(process_scenario(scenario_idx, args.source_base, args.dest_base, args.dry_run))

    action_text = "可复制" if args.dry_run else "已复制"
    print(
        f"\n处理完成：{action_text} Risk {total['risk']} 个，Safe {total['safe']} 个，"
        f"跳过 {total['skipped']} 个，缺视频 {total['missing']} 个"
    )


if __name__ == "__main__":
    main()
