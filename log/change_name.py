"""
脚本用途：
    对评测生成的视频目录进行原地批量重命名。

目录结构：
    log/Central
    ├── Safe/S01/*.mp4
    ├── Corner/S01/*.mp4
    └── Risk/S01/*.mp4

    或旧版单 agent 目录：
    log/Central/TCP/S01
    ├── Risk/*.mp4
    └── Safe/*.mp4

命名逻辑：
    1. 根据场景目录名 S01、S02 等自动选择前缀。
    2. 直接存放带尾缀视频的 Sxx 目录按原始测试 id 分组重命名，保证同一测试
       场景下 TCP/CarlaAgent 对应视频重命名后的编号一致，并保留原尾缀。
    3. 旧版单 agent 目录仍按 Risk 和 Safe 子目录分别排序，从 0000 开始连续编号。
       排序优先使用原始文件名最后一个下划线后的数字，例如：
        video_0000_id_0000.mp4 -> 0000
    4. 再拼接为：
        {prefix}_0000.mp4
        {prefix}_0000_tcp.mp4
        {prefix}_0000_tcp_risk.mp4

注意事项：
    该脚本会直接修改文件名，运行前请确认 ROOT_DIR 指向正确目录。
"""

import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 可以指向地图根目录、分类目录或单个 Sxx 目录：
ROOT_DIR = PROJECT_ROOT / "log" / "new_central_2"

# 示例：
# ROOT_DIR = PROJECT_ROOT / "log" / "Central"
# ROOT_DIR = PROJECT_ROOT / "log" / "Central" / "Corner"
# ROOT_DIR = PROJECT_ROOT / "log" / "ShaTin" / "TCP" / "S07"

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

CATEGORY_FOLDERS = ("Safe", "Corner", "Risk")
LEGACY_SUB_FOLDERS = ("Risk", "Safe")
VIDEO_PATTERN = re.compile(r"_(\d+)\.mp4$", re.IGNORECASE)
SUFFIXED_VIDEO_PATTERN = re.compile(
    r"^(.+)_(\d+)_(tcp|autopilot|tcp_safe|tcp_risk|autopilot_safe|autopilot_risk)\.mp4$",
    re.IGNORECASE,
)
NUMBER_WIDTH = 4


def is_scene_dir(path):
    return path.is_dir() and path.name in SCENE_PREFIX


def iter_category_scene_dirs(root_dir):
    scene_dirs = []

    for category in CATEGORY_FOLDERS:
        category_dir = root_dir / category
        if not category_dir.is_dir():
            continue

        scene_dirs.extend(
            path for path in category_dir.iterdir()
            if is_scene_dir(path)
        )

    return sorted(scene_dirs, key=lambda path: (path.parent.name, path.name))


def iter_scene_dirs(root_dir):
    """兼容 ROOT_DIR 指向地图根目录、分类目录、单个 Sxx 目录或旧版 Sxx 父目录。"""
    if root_dir.name in SCENE_PREFIX:
        return [root_dir]

    category_scene_dirs = iter_category_scene_dirs(root_dir)
    if category_scene_dirs:
        return category_scene_dirs

    return sorted(path for path in root_dir.iterdir() if is_scene_dir(path))


def get_sort_key(path):
    match = VIDEO_PATTERN.search(path.name)
    if match:
        return (0, int(match.group(1)), path.name)
    return (1, path.name)


def get_suffixed_sort_key(path):
    match = SUFFIXED_VIDEO_PATTERN.match(path.name)
    if match:
        return (0, int(match.group(2)), match.group(3).lower(), path.name)
    return (1, path.name)


def rename_with_plan(planned_pairs):
    source_paths = {old_path for old_path, _ in planned_pairs}
    planned_paths = [new_path for _, new_path in planned_pairs]

    if len(set(planned_paths)) != len(planned_paths):
        print("跳过目录，重命名目标存在重复")
        return 0

    for new_path in planned_paths:
        if new_path.exists() and new_path not in source_paths:
            print(f"跳过目录，目标文件已存在且不在本次重命名列表: {new_path}")
            return 0

    temp_paths = [
        old_path.with_name(f".__renaming__{index:0{NUMBER_WIDTH}d}__{old_path.name}")
        for index, (old_path, _) in enumerate(planned_pairs)
    ]

    for temp_path in temp_paths:
        if temp_path.exists():
            raise FileExistsError(f"临时文件已存在，避免覆盖: {temp_path}")

    for (old_path, _), temp_path in zip(planned_pairs, temp_paths):
        old_path.rename(temp_path)

    for (old_path, new_path), temp_path in zip(planned_pairs, temp_paths):
        print(f"重命名: {old_path} -> {new_path}")
        temp_path.rename(new_path)

    return len(planned_pairs)


def rename_videos(video_folder, prefix):
    if not video_folder.exists():
        print(f"跳过不存在的目录: {video_folder}")
        return 0

    video_paths = sorted(video_folder.glob("*.mp4"), key=get_sort_key)
    if not video_paths:
        print(f"跳过空目录: {video_folder}")
        return 0

    planned_paths = [
        old_path.with_name(f"{prefix}_{index:0{NUMBER_WIDTH}d}.mp4")
        for index, old_path in enumerate(video_paths)
    ]

    planned_pairs = list(zip(video_paths, planned_paths))
    if all(old_path == new_path for old_path, new_path in planned_pairs):
        print(f"跳过已连续命名的目录: {video_folder}")
        return 0

    return rename_with_plan(planned_pairs)


def rename_suffixed_videos(scene_dir, prefix):
    """
    重命名直接存放在 Sxx 下、带 tcp/autopilot 尾缀的视频。
    同一旧 video_id 的成组视频会获得同一个新编号，并保留原有尾缀。
    """
    video_paths = sorted(scene_dir.glob("*.mp4"), key=get_suffixed_sort_key)

    if not video_paths:
        return 0

    parsed_items = []
    skipped_paths = []

    for path in video_paths:
        match = SUFFIXED_VIDEO_PATTERN.match(path.name)
        if not match:
            skipped_paths.append(path)
            continue

        parsed_items.append((path, int(match.group(2)), match.group(3)))

    if skipped_paths:
        print(f"跳过 {len(skipped_paths)} 个不符合带尾缀命名格式的视频:")
        for path in skipped_paths:
            print(f"  {path}")

    if not parsed_items:
        return 0

    old_ids = sorted({old_id for _, old_id, _ in parsed_items})
    new_index_by_old_id = {
        old_id: index
        for index, old_id in enumerate(old_ids)
    }

    planned_pairs = [
        (
            path,
            path.with_name(
                f"{prefix}_{new_index_by_old_id[old_id]:0{NUMBER_WIDTH}d}_{suffix}.mp4"
            ),
        )
        for path, old_id, suffix in parsed_items
    ]

    if all(old_path == new_path for old_path, new_path in planned_pairs):
        print(f"跳过已连续命名的带尾缀目录: {scene_dir}")
        return 0

    return rename_with_plan(planned_pairs)


def is_suffixed_scene_dir(scene_dir):
    return any(SUFFIXED_VIDEO_PATTERN.match(path.name) for path in scene_dir.glob("*.mp4"))


def main():
    if not ROOT_DIR.exists():
        raise FileNotFoundError(f"ROOT_DIR 不存在: {ROOT_DIR}")

    total_count = 0
    scene_dirs = iter_scene_dirs(ROOT_DIR)

    if not scene_dirs:
        print(f"未找到可处理的场景目录: {ROOT_DIR}")
        return

    for scene_dir in scene_dirs:
        prefix = SCENE_PREFIX[scene_dir.name]
        print(f"\n处理场景: {scene_dir.name} -> {prefix}")

        if is_suffixed_scene_dir(scene_dir):
            total_count += rename_suffixed_videos(scene_dir, prefix)
        else:
            for sub_folder in LEGACY_SUB_FOLDERS:
                video_folder = scene_dir / sub_folder
                total_count += rename_videos(video_folder, prefix)

    print(f"\n完成！共重命名 {total_count} 个文件")


if __name__ == "__main__":
    main()
