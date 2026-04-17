"""
创建和管理用于场景测试的route,它允许用户通过交互式绘图界面选择航迹点,并根据这些航迹点来创建测试路线
具体功能包括：
(1) 使用鼠标滚轮拖拽和缩放地图视图
(2) 绘制测试路线：通过鼠标点击事件处理函数onclick，用户可以选择或移除航迹点
(3) 右键保存路线：将选定的航迹点保存为scenario_origin目录下的npy文件
"""
import os
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from utilities import select_waypoints, get_nearist_waypoints, get_map_centers
import argparse
import carla
import numpy as np
from safebench.scenario.tools.route_manipulation import interpolate_trajectory


# 缩放拖拽相关全局变量
is_dragging = False
last_mouse_pos = None
last_update_time = 0
fig = None  # 全局 figure 引用

def on_key_press(event):
    """ESC 键直接关闭窗口，退出 plt.show(block=True)"""
    if event.key == 'escape':
        plt.close(event.canvas.figure)

def draw(ax, center, dist, road_waypoints, selected_waypoints_idx):
    ax.cla()
    # 绘制道路航迹点
    ax.plot(road_waypoints[:, 0], -road_waypoints[:, 1], 'o', color='y', markersize=3)

    # 用 quiver 一次性绘制所有方向箭头（比循环调用 plt.arrow 快几十倍）
    length = 4
    yaws = road_waypoints[:, 4] / 180 * np.pi
    dx = length * np.cos(yaws)
    dy = -length * np.sin(yaws)
    ax.quiver(road_waypoints[:, 0], -road_waypoints[:, 1], dx, dy,
              color='y', angles='xy', scale_units='xy', scale=1, width=0.003)

    # 绘制选中的航迹点
    if len(selected_waypoints_idx) > 0:
        waypoints_idx = np.array(selected_waypoints_idx)
        waypoints = np.take(road_waypoints, waypoints_idx, axis=0)
        ax.plot(waypoints[:, 0], -waypoints[:, 1], '-o', color='b')
        ax.plot(waypoints[0, 0], -waypoints[0, 1], 'o', color='g')
        ax.text(waypoints[0, 0] + 8, -waypoints[0, 1] + 8, "Start", bbox=dict(facecolor='green', alpha=0.7))
        if len(selected_waypoints_idx) > 1:
            ax.plot(waypoints[-1, 0], -waypoints[-1, 1], 'o', color='r')
            ax.text(waypoints[-1, 0] + 8, -waypoints[-1, 1] + 8, "End", bbox=dict(facecolor='red', alpha=0.7))

    # 设置显示范围
    x_min, x_max = center[0] - dist, center[0] + dist
    y_min, y_max = -center[1] - dist, -center[1] + dist
    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])

def set_title(ax, title=None):
    if title is None:
        title = "Left click: select/remove waypoints.\nRight click: save route.\nScroll: zoom | Middle drag: move."
    ax.set_title(title, fontsize=18, loc='left')

def save_waypoints(config, save_dir, selected_waypoints):
    route_id = config.route
    if config.route < 0:
        route_id = 0
        while os.path.isfile(os.path.join(save_dir, f'route_{route_id:02d}.npy')):
            route_id += 1
    os.makedirs(save_dir, exist_ok=True)
    save_file = os.path.join(save_dir, f'route_{route_id:02d}.npy')
    np.save(save_file, selected_waypoints)

def load_route(config, dist, waypoints_sparse, save_dir):
    center = None
    road_waypoints = None
    selected_waypoints_idx = []
    if config.route >= 0:
        route_file = os.path.join(save_dir, f'route_{config.route:02d}.npy')
        if os.path.isfile(route_file):
            selected_waypoints = np.load(route_file)
            geo_center = selected_waypoints[:, :2].mean(0)
            centers = get_map_centers(config.map)[0]
            centers = np.asarray([centers, centers])
            center_dists = np.linalg.norm(np.array(centers) - geo_center, axis=1)
            center = centers[center_dists.argmin()]
            road_waypoints = select_waypoints(waypoints_sparse, center, dist)
            for waypoint in selected_waypoints:
                idx, dist_wp = get_nearist_waypoints(waypoint, road_waypoints)
                if dist_wp > 5:
                    print(f"waypoint {waypoint} not found, assigned nearest {road_waypoints[idx]}")
                selected_waypoints_idx.append(idx)
    return center, road_waypoints, selected_waypoints_idx

# ==== 新增交互功能 ====
# 事件节流相关变量
update_interval = 0.05  # 最小更新间隔（秒），约20fps

def on_scroll(event):
    """鼠标滚轮缩放地图"""
    global dist, center, road_waypoints, last_update_time
    import time

    zoom_factor = 0.9
    if event.button == 'up':
        dist *= zoom_factor
    elif event.button == 'down':
        dist /= zoom_factor
    dist = max(20, min(dist, 2000))

    road_waypoints = select_waypoints(waypoints_sparse, center, dist)
    draw(ax, center, dist, road_waypoints, selected_waypoints_idx)
    set_title(ax, f"Zoom dist={dist:.1f}")
    fig.canvas.draw_idle()
    last_update_time = time.time()

def on_middle_press(event):
    """按下鼠标中键开始拖拽"""
    global is_dragging, last_mouse_pos
    if event.button == 2:
        is_dragging = True
        last_mouse_pos = (event.xdata, event.ydata)

def on_motion(event):
    """拖拽地图（带节流）"""
    global center, last_mouse_pos, road_waypoints, last_update_time
    import time

    if is_dragging and event.xdata is not None and event.ydata is not None and last_mouse_pos is not None:
        dx = event.xdata - last_mouse_pos[0]
        dy = event.ydata - last_mouse_pos[1]
        center = (center[0] - dx, center[1] + dy)
        last_mouse_pos = (event.xdata, event.ydata)

        # 节流：只在时间间隔足够大时才更新显示
        current_time = time.time()
        if current_time - last_update_time >= update_interval:
            road_waypoints = select_waypoints(waypoints_sparse, center, dist)
            draw(ax, center, dist, road_waypoints, selected_waypoints_idx)
            fig.canvas.draw_idle()
            last_update_time = current_time

def on_middle_release(event):
    """释放鼠标中键结束拖拽"""
    global is_dragging, road_waypoints
    if event.button == 2:
        is_dragging = False
        # 释放时强制更新一次，确保最终位置正确
        road_waypoints = select_waypoints(waypoints_sparse, center, dist)
        draw(ax, center, dist, road_waypoints, selected_waypoints_idx)
        fig.canvas.draw_idle()

def calculate_interpolate_trajectory(origin_waypoints_loc, world):
    """
    给定 route.npy 中的路径，调用 CARLA 的 interpolate_trajectory 尝试生成可行路径
    trajectory: list of carla.Location 或 [x,y,z] 形式坐标
    world: CARLA 世界对象
    """
    try:
        route = interpolate_trajectory(world, origin_waypoints_loc, 2.0)
    except Exception as e:
        print(f"[路径规划] 路径插值失败: {e}")
        return []
    # 转成平面坐标列表
    waypoint_xy = []
    for transform_tuple in route:
        waypoint_xy.append([transform_tuple[0].location.x, transform_tuple[0].location.y])
    return waypoint_xy


def check_routes_is_possible(config):
    # 加载指定地图
    client = carla.Client("localhost", 2000)
    client.set_timeout(25.0)  # 超时防卡死
    world = client.load_world(config.map)

    routes_dir = os.path.join("scenario_origin/center", f"scenario_{config.scenario:02d}_routes")
    route_files = sorted(f for f in os.listdir(routes_dir) if f.startswith("route_") and f.endswith(".npy"))
    for route_file in route_files:
        route_path = os.path.join(routes_dir, route_file)
        try:
            waypoints_trajectory = []
            trajectory = np.load(route_path)
            for point in trajectory:
                x, y, z = point[:3]  # 获取前三列的坐标
                # 使用 z=0 与运行时 route_parser.py 的行为保持一致，
                # 确保 wmap.get_waypoint() snap 到相同的地面道路节点
                location = carla.Location(x=x, y=y, z=0)
                waypoints_trajectory.append(location)
            waypoint_xy = calculate_interpolate_trajectory(waypoints_trajectory, world)
            if len(waypoint_xy) < 2:  # 如果生成的路点太少，认为不可行
                os.remove(route_path)
                print(f"[清理] 删除不可行路线: {route_file}")
                continue
        except Exception as e:
            print(f"[清理] 加载或检测路线失败 {route_file}: {e}")
            os.remove(route_path)
            continue

    # 重新排序和命名
    remaining_routes = sorted(f for f in os.listdir(routes_dir) if f.startswith("route_") and f.endswith(".npy"))
    tmp_prefix = "._tmp_rename_"
    for idx, route_file in enumerate(remaining_routes):
        old_route_path = os.path.join(routes_dir, route_file)
        tmp_route_path = os.path.join(routes_dir, f"{tmp_prefix}{idx:02d}.npy")
        os.rename(old_route_path, tmp_route_path)

    for idx in range(len(remaining_routes)):
        tmp_route_path = os.path.join(routes_dir, f"{tmp_prefix}{idx:02d}.npy")
        final_route_path = os.path.join(routes_dir, f"route_{idx:02d}.npy")
        if os.path.exists(tmp_route_path):
            os.rename(tmp_route_path, final_route_path)
    print("[重编号] 完成，序号已重新排列从 00 开始")


def main(config):
    global center, dist, ax, fig, waypoints_sparse, road_waypoints, selected_waypoints_idx
    # 不使用 plt.ion()，改用 plt.show(block=True)，避免与 TkAgg 事件循环冲突

    waypoints_sparse = np.load(f"map_waypoints/{config.map}/sparse.npy")
    selected_waypoints_idx = []

    if config.road == 'auto':
        if config.scenario + 2 in [1, 2, 3, 5, 6]:
            config.road = 'straight'
        elif config.scenario + 2 in [4, 7, 8, 9, 10]:
            config.road = 'intersection'
        else:
            raise ValueError("scenario not found.")

    center = get_map_centers(config.map)[0]
    dist = 1000
    road_waypoints = select_waypoints(waypoints_sparse, center, dist)

    save_dir = os.path.join("scenario_origin", config.map, f"scenario_{config.scenario:02d}_routes")
    new_center, new_road_waypoints, selected_waypoints_idx = load_route(config, dist, waypoints_sparse, save_dir)
    if len(selected_waypoints_idx) > 0:
        center = new_center
        road_waypoints = new_road_waypoints

    def onclick(event):
        # 点击到图形边框外时 xdata/ydata 为 None，直接忽略
        if event.xdata is None or event.ydata is None:
            return
        if int(event.button) == 1:  # 左键
            waypoints_dist = np.linalg.norm(road_waypoints[:, :2] - [event.xdata, -event.ydata], axis=1)
            if waypoints_dist.min() < 5:
                idx = waypoints_dist.argmin()
                if idx in selected_waypoints_idx:
                    selected_waypoints_idx.remove(idx)
                else:
                    selected_waypoints_idx.append(idx)
            draw(ax, center, dist, road_waypoints, selected_waypoints_idx)
            set_title(ax)
            fig.canvas.draw_idle()
        elif int(event.button) == 3:  # 右键
            if len(selected_waypoints_idx) < 2:
                set_title(ax, "Need at least 2 waypoints to create a route.")
                fig.canvas.draw_idle()
            else:
                selected_waypoints = np.take(road_waypoints, selected_waypoints_idx, axis=0)
                save_waypoints(config, save_dir, selected_waypoints)
                if config.route < 0:
                    selected_waypoints_idx.clear()
                    draw(ax, center, dist, road_waypoints, selected_waypoints_idx)
                    set_title(ax, "Route created! Click to create more.")
                else:
                    set_title(ax, "Route updated! Keep modifying if needed.")
                fig.canvas.draw_idle()

    fig, ax = plt.subplots(figsize=(16, 16))  # 32x32 过大，改为 16x16 以降低渲染开销
    draw(ax, center, dist, road_waypoints, selected_waypoints_idx)
    set_title(ax)

    # 绑定事件
    fig.canvas.mpl_connect('button_press_event', onclick)
    fig.canvas.mpl_connect('key_press_event', on_key_press)
    fig.canvas.mpl_connect('scroll_event', on_scroll)
    fig.canvas.mpl_connect('button_press_event', on_middle_press)
    fig.canvas.mpl_connect('motion_notify_event', on_motion)
    fig.canvas.mpl_connect('button_release_event', on_middle_release)

    # 使用阻塞模式，直接交给 Tk 事件循环管理（ESC 键关闭窗口）
    plt.show(block=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--map', type=str, default='center')
    parser.add_argument('--save_dir', type=str, default="center")
    parser.add_argument('--scenario', type=int, default=8)
    parser.add_argument('--route', type=int, default=-1)
    parser.add_argument('--road', type=str, default='auto', choices=['auto', 'intersection', 'straight'])
    args = parser.parse_args()

    main(args)

    # 检查路径是否可行
    check_routes_is_possible(args)


