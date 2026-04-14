"""
检查新测试路线是否与已有XML路线重叠。

关键背景：
- 新路线：scenario_origin/{map}/scenario_{id}_routes/route_XX.npy
  每个文件包含用户在地图上点选的 2 个稀疏路点（起点、终点），坐标为 CARLA 世界坐标。
- 已知路线：scenario_data/{known_dir}/scenario_{id}_routes/*.xml
  每个文件同样只包含 2 个稀疏路点（由 export_routes.py 从 npy 直接导出），
  与新路线属于不同路线集合，坐标范围不同。
- 由于两边都只有稀疏起终点，必须对两边都调用 CARLA GlobalRoutePlanner 规划密集轨迹，
  才能进行有意义的重叠度比较。

工作流程：
1. 解析已知 XML（每条路线取 weather_00 版本），用 CARLA GRP 生成密集轨迹并缓存。
2. 逐条加载新 npy 路线，用 CARLA GRP 生成密集轨迹。
3. 端点快速筛选：比较新路线密集轨迹首尾 vs 已知密集轨迹首尾（同一坐标系，可靠比较）。
4. 精细检查：计算双向轨迹重叠比例，超过阈值则认为重叠。
5. 若重叠，删除对应的 route_{idx}.npy 及 scenario_{idx}.npy / scenario_{idx}_sides.npy。
6. 最后对剩余文件重编号，保持序号连续。

用法示例：
    python tools/check_route_overlap.py \\
        --map center \\
        --scenario 1 \\
        --origin_dir scenario_origin/center \\
        --known_dir scenario_data/central \\
        --port 2000 \\
        --endpoint_thresh 30.0 \\
        --overlap_thresh 0.6
"""

import os
import re
import sys
import argparse

import numpy as np
import carla

# 将项目根目录加入路径，以便导入 safebench 模块
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from safebench.scenario.tools.route_manipulation import interpolate_trajectory
from utilities import parse_route


# ────────────────────────────────────────────────────────────────────────────
# 辅助函数
# ────────────────────────────────────────────────────────────────────────────

def _try_remove(path: str):
    """安全删除文件，不存在时静默忽略。"""
    try:
        if os.path.exists(path):
            os.remove(path)
            print(f"  [删除] {path}")
    except Exception as exc:
        print(f"  [警告] 无法删除 {path}: {exc}")


def _sort_key(filename: str) -> int:
    """按文件名中第一个数字排序。"""
    nums = re.findall(r'\d+', filename)
    return int(nums[0]) if nums else 0


def npy_to_carla_locations(waypoints_array: np.ndarray):
    """将 npy 路线数组（每行 [x,y,z,pitch,yaw,roll]）转为 carla.Location 列表。"""
    locations = []
    for row in waypoints_array:
        x, y, z = float(row[0]), float(row[1]), float(row[2])
        locations.append(carla.Location(x=x, y=y, z=z))
    return locations


def xml_to_carla_locations(xml_file: str):
    """
    从 XML 路线文件中读取所有 waypoints，转为 carla.Location 列表。
    XML 中每条路线通常只有 2 个原始稀疏路点（起点、终点），
    需要交给 CARLA GRP 规划密集轨迹。
    返回空列表表示解析失败。
    """
    try:
        waypoints_arr, _ = parse_route(xml_file)   # shape: (n_routes, n_wps, 6)
        if waypoints_arr.size == 0:
            return []
        wps = waypoints_arr[0]  # 取第一条 route，列顺序 [x,y,z,pitch,yaw,roll]
        locs = []
        for row in wps:
            locs.append(carla.Location(x=float(row[0]), y=float(row[1]), z=float(row[2])))
        return locs
    except Exception as exc:
        print(f"  [警告] 解析 XML 失败 {xml_file}: {exc}")
        return []


def get_dense_trajectory(locations, world, grp, route_name: str = "") -> np.ndarray:
    """
    调用 CARLA interpolate_trajectory 将稀疏路点规划为密集轨迹。
    返回 np.ndarray shape=(N,2)，失败时返回空数组。

    Args:
        locations: 稀疏路点列表（carla.Location）
        world: CARLA 世界对象
        grp: GlobalRoutePlanner 实例
        route_name: 用于日志输出的路线名称

    Returns:
        np.ndarray: 密集轨迹，shape=(N,2)；失败时返回空数组
    """
    if len(locations) < 2:
        print(f"  [警告] {route_name} 路点不足（<2个）")
        return np.empty((0, 2))
    try:
        route = interpolate_trajectory(world, locations, hop_resolution=2.0, grp=grp)
    except Exception as exc:
        print(f"  [警告] {route_name} 轨迹规划失败: {exc}")
        return np.empty((0, 2))
    if len(route) < 2:
        print(f"  [警告] {route_name} 规划结果为空")
        return np.empty((0, 2))
    return np.array([[t[0].location.x, t[0].location.y] for t in route])


def endpoints_close(traj_a: np.ndarray, traj_b: np.ndarray, thresh: float) -> bool:
    """
    比较两条密集轨迹的首尾端点是否足够接近（正向和反向均检查）。
    正向：a首↔b首 且 a尾↔b尾；反向：a首↔b尾 且 a尾↔b首。
    两条轨迹均由 CARLA GRP 在同一地图上规划，坐标系完全一致。
    """
    def dist(p, q):
        return float(np.linalg.norm(np.array(p) - np.array(q)))

    sa, ea = traj_a[0], traj_a[-1]
    sb, eb = traj_b[0], traj_b[-1]
    forward  = dist(sa, sb) < thresh and dist(ea, eb) < thresh
    backward = dist(sa, eb) < thresh and dist(ea, sb) < thresh
    return forward or backward


def trajectory_overlap_ratio(traj_a: np.ndarray, traj_b: np.ndarray,
                              radius: float = 5.0) -> float:
    """
    计算 traj_a 中有多少比例的点落在 traj_b 的 radius 范围内。
    返回值范围 [0, 1]。
    对 traj_b 均匀下采样至最多 300 个点，避免 O(N²) 过慢。
    """
    if len(traj_a) == 0 or len(traj_b) == 0:
        return 0.0
    step = max(1, len(traj_b) // 300)
    b_sampled = traj_b[::step]
    count = 0
    for pt in traj_a:
        if np.linalg.norm(b_sampled - pt, axis=1).min() < radius:
            count += 1
    return count / len(traj_a)


# ────────────────────────────────────────────────────────────────────────────
# 核心逻辑
# ────────────────────────────────────────────────────────────────────────────

def collect_known_trajectories(known_routes_dir: str, world, grp):
    """
    对已知 XML 目录中每条不同的路线（route_id），调用 CARLA GRP 规划密集轨迹并缓存。

    - 每条路线只取一个 weather 版本（weather_00 优先），路线坐标与天气无关。
    - 每条路线通常只有 2 个稀疏路点，需要 GRP 规划才能做重叠度比较。

    返回：list of np.ndarray，每项 shape=(N,2)，按 route_id 升序排列。
    """
    if not os.path.isdir(known_routes_dir):
        print(f"[错误] 已知路线目录不存在: {known_routes_dir}")
        return []

    all_files = os.listdir(known_routes_dir)

    # 按 route_id 分组，每组只保留一个文件（weather_00 覆盖其他）
    route_groups: dict = {}   # route_id_str -> filename
    for f in sorted(all_files):
        if not f.endswith('.xml'):
            continue
        m = re.search(r'route_(\d+)', f)
        if not m:
            continue
        rid = m.group(1)
        if rid not in route_groups:
            route_groups[rid] = f          # 先到先得
        elif 'weather_00' in f:
            route_groups[rid] = f          # weather_00 覆盖

    if not route_groups:
        print(f"[警告] {known_routes_dir} 中未找到任何 XML 文件")
        return []

    print(f"[已知路线] 共 {len(route_groups)} 条不同路线，开始调用 CARLA GRP 规划密集轨迹 ...")
    trajs = []
    for rid in sorted(route_groups.keys(), key=lambda x: int(x)):
        fname = route_groups[rid]
        fpath = os.path.join(known_routes_dir, fname)
        locs = xml_to_carla_locations(fpath)
        if len(locs) < 2:
            print(f"  [警告] {fname} 路点不足，跳过")
            continue
        traj = get_dense_trajectory(locs, world, grp)
        if len(traj) < 2:
            print(f"  [警告] {fname} GRP 规划失败，跳过")
            continue
        trajs.append(traj)
        print(f"  已知路线[{rid}] {fname}: "
              f"规划得 {len(traj)} 个点，"
              f"首=({traj[0,0]:.1f},{traj[0,1]:.1f}), "
              f"尾=({traj[-1,0]:.1f},{traj[-1,1]:.1f})")

    print(f"[已知路线] 成功缓存 {len(trajs)} 条密集轨迹\n")
    return trajs


def check_and_remove_overlapping_routes(config):
    """主检查函数：逐条检查新路线是否与已知路线重叠，并删除重叠路线。"""

    scenario_id      = config.scenario
    origin_dir       = config.origin_dir
    known_dir        = config.known_dir
    map_name         = config.map

    routes_dir       = os.path.join(origin_dir, f"scenario_{scenario_id:02d}_routes")
    scenarios_dir    = os.path.join(origin_dir, f"scenario_{scenario_id:02d}_scenarios")
    known_routes_dir = os.path.join(known_dir,  f"scenario_{scenario_id:02d}_routes")

    if not os.path.isdir(routes_dir):
        print(f"[错误] 新路线目录不存在: {routes_dir}")
        return

    # 连接 CARLA（新路线和已知路线都需要调用 GRP）
    print(f"[CARLA] 连接到 localhost:{config.port} ...")
    client = carla.Client("localhost", config.port)
    client.set_timeout(30.0)
    print(f"[CARLA] 加载地图 {map_name} ...")
    world = client.load_world(map_name)

    # 复用同一个 GlobalRoutePlanner，避免重复初始化
    from safebench.carla_agents.navigation.global_route_planner import GlobalRoutePlanner
    grp = GlobalRoutePlanner(world.get_map(), 2.0)

    # 预先规划所有已知路线的密集轨迹并缓存
    known_trajs = collect_known_trajectories(known_routes_dir, world, grp)
    if not known_trajs:
        print("[信息] 没有已知路线可比较，退出。")
        return

    #  逐条检查新路线
    new_route_files = sorted(
        [f for f in os.listdir(routes_dir) if f.startswith("route_") and f.endswith(".npy")],
        key=_sort_key
    )
    print(f"\n[检查] 共 {len(new_route_files)} 条新路线，"
          f"逐条与 {len(known_trajs)} 条已知路线比较")

    to_delete = []   # 记录需要删除的路线编号字符串，如 "03"

    for route_file in new_route_files:
        route_path = os.path.join(routes_dir, route_file)
        m = re.search(r'\d+', route_file)
        route_idx = m.group() if m else route_file

        try:
            waypoints_array = np.load(route_path)
        except Exception as exc:
            print(f"  [警告] 加载失败 {route_file}: {exc}，跳过")
            continue

        print(f"\n  ── 检查 {route_file} "
              f"(原始起点=({waypoints_array[0,0]:.1f},{waypoints_array[0,1]:.1f}), "
              f"原始终点=({waypoints_array[-1,0]:.1f},{waypoints_array[-1,1]:.1f})) ──")

        # ① 用 CARLA GRP 规划新路线的密集轨迹
        new_locs = npy_to_carla_locations(waypoints_array)
        new_traj = get_dense_trajectory(new_locs, world, grp, route_name=route_file)

        if len(new_traj) < 2:
            print(f"    → 新路线轨迹生成失败（无法规划密集轨迹），标记为无效并删除")
            to_delete.append(route_idx)
            continue

        print(f"    → GRP 规划得 {len(new_traj)} 个点，"
              f"首=({new_traj[0,0]:.1f},{new_traj[0,1]:.1f}), "
              f"尾=({new_traj[-1,0]:.1f},{new_traj[-1,1]:.1f})")

        # ② 端点快速筛选：两条密集轨迹首尾均在同一 CARLA 世界坐标系下，可直接比较
        candidates = []
        for i, ktraj in enumerate(known_trajs):
            if endpoints_close(new_traj, ktraj, config.endpoint_thresh):
                candidates.append(i)

        if not candidates:
            print(f" → 与所有已知路线端点距离均 > {config.endpoint_thresh}m，不重叠 ✓")
            continue

        print(f" → 端点与已知路线 {candidates} 接近，进行轨迹重叠度精细检查 ...")

        # ③ 精细检查：双向重叠比例，任一方向超过阈值即判定为重叠
        overlapped = False
        for i in candidates:
            ktraj = known_trajs[i]
            ratio_new2known = trajectory_overlap_ratio(new_traj, ktraj)
            ratio_known2new = trajectory_overlap_ratio(ktraj, new_traj)
            print(f"    → 已知路线[{i:02d}] 重叠度: "
                  f"new→known={ratio_new2known:.2f}, known→new={ratio_known2new:.2f} "
                  f"(阈值={config.overlap_thresh})")
            if ratio_new2known >= config.overlap_thresh or ratio_known2new >= config.overlap_thresh:
                overlapped = True
                break   # 发现一条重叠即可判定，无需继续

        if overlapped:
            print(f"    → ✗ 路线重叠，标记为待删除: {route_file}")
            to_delete.append(route_idx)
        else:
            print(f"    → ✓ 路线不重叠，保留")

    # 执行删除
    if not to_delete:
        print("\n[结果] 没有重叠路线，无需删除。")
        return

    print(f"\n[删除] 共 {len(to_delete)} 条重叠路线待删除: {to_delete}")
    for idx in to_delete:
        _try_remove(os.path.join(routes_dir, f"route_{idx}.npy"))
        if os.path.isdir(scenarios_dir):
            _try_remove(os.path.join(scenarios_dir, f"scenario_{idx}.npy"))
            _try_remove(os.path.join(scenarios_dir, f"scenario_{idx}_sides.npy"))

    # 重编号剩余文件
    _renumber(routes_dir, scenarios_dir)


# 重编号（与 export.py 保持相同风格）
def _renumber(routes_dir: str, scenarios_dir: str):
    """
    删除重叠路线后，对 routes 和 scenarios 目录中的文件重新连续编号。
    两阶段重命名：先改为临时名，再改为最终名，避免命名冲突。
    """
    tmp_prefix = "._tmp_rename_"

    # 预清理残留临时文件（防止上次崩溃遗留）
    for d in [routes_dir, scenarios_dir]:
        if not os.path.isdir(d):
            continue
        for f in os.listdir(d):
            if f.startswith(tmp_prefix):
                _try_remove(os.path.join(d, f))

    remaining_routes = sorted(
        [f for f in os.listdir(routes_dir) if f.startswith("route_") and f.endswith(".npy")],
        key=_sort_key
    )

    if not remaining_routes:
        print("[重编号] 路线目录为空，跳过重编号。")
        return

    print(f"\n[重编号] 对 {len(remaining_routes)} 条剩余路线重新编号 ...")

    # 第一阶段：→ 临时名
    for new_idx, route_file in enumerate(remaining_routes):
        old_idx = re.search(r'\d+', route_file).group()

        os.replace(
            os.path.join(routes_dir, f"route_{old_idx}.npy"),
            os.path.join(routes_dir, f"{tmp_prefix}{new_idx:02d}.npy")
        )
        if os.path.isdir(scenarios_dir):
            scn = os.path.join(scenarios_dir, f"scenario_{old_idx}.npy")
            if os.path.exists(scn):
                os.replace(scn, os.path.join(scenarios_dir, f"{tmp_prefix}{new_idx:02d}.npy"))
            sides = os.path.join(scenarios_dir, f"scenario_{old_idx}_sides.npy")
            if os.path.exists(sides):
                os.replace(sides, os.path.join(scenarios_dir,
                                                f"{tmp_prefix}_sides_{new_idx:02d}.npy"))

    # 第二阶段：→ 最终名
    for new_idx in range(len(remaining_routes)):
        tmp = os.path.join(routes_dir, f"{tmp_prefix}{new_idx:02d}.npy")
        if os.path.exists(tmp):
            os.replace(tmp, os.path.join(routes_dir, f"route_{new_idx:02d}.npy"))

        if os.path.isdir(scenarios_dir):
            tmp = os.path.join(scenarios_dir, f"{tmp_prefix}{new_idx:02d}.npy")
            if os.path.exists(tmp):
                os.replace(tmp, os.path.join(scenarios_dir, f"scenario_{new_idx:02d}.npy"))
            tmp = os.path.join(scenarios_dir, f"{tmp_prefix}_sides_{new_idx:02d}.npy")
            if os.path.exists(tmp):
                os.replace(tmp, os.path.join(scenarios_dir,
                                              f"scenario_{new_idx:02d}_sides.npy"))

    print(f"[重编号] 完成，序号重排为 00 ~ {len(remaining_routes)-1:02d}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="检查新路线是否与已知 XML 路线重叠，并删除重叠路线及对应场景文件"
    )
    parser.add_argument('--map',      type=str, default='center',
                        help='CARLA 地图名称（需与 CARLA 服务器一致）')
    parser.add_argument('--scenario', type=int, default=8,
                        help='场景编号（如 1 对应 scenario_01）')
    parser.add_argument('--origin_dir', type=str, default='scenario_origin/center',
                        help='新路线 npy 文件的根目录（含 scenario_XX_routes 子目录）')
    parser.add_argument('--known_dir',  type=str, default='scenario_data/central',
                        help='已知路线 XML 文件的根目录（含 scenario_XX_routes 子目录）')
    parser.add_argument('--port', type=int, default=2000,
                        help='CARLA 服务器端口')
    parser.add_argument('--endpoint_thresh', type=float, default=20.0,
                        help='端点距离阈值（米）：新路线与已知路线的 GRP 密集轨迹首/尾距离'
                             '均低于此值时才进入精细重叠度检查')
    parser.add_argument('--overlap_thresh', type=float, default=0.6,
                        help='轨迹重叠比例阈值（0~1）：任一方向重叠比例超过此值则判定为重叠')

    args = parser.parse_args()
    check_and_remove_overlapping_routes(args)
