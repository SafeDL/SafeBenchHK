"""
创建和编辑自动驾驶仿真场景中交通物体的生成点和场景运行的触发点：

(1) 读取预先绘制的测试路线,并将其显示在图形界面上
(2) 左键选取对抗actor的位置和场景执行的触发位置
(3) 右键保存选取的路径点到scenario_origin的对应文件夹中

"""
import numpy as np
import os
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from utilities import select_waypoints, get_nearist_waypoints, get_map_centers
import argparse

# ==== 全局交互状态 ====
exit_flag = False
is_dragging = False
last_mouse_pos = None

# 全局绘制用数据
center = None
dist = None
waypoints_sparse = None
road_waypoints = None
selected_route_idx = []
selected_trigger_idx = []

# 路线批量
route_files = []
current_idx = 0
map_name = ""
scenario_id = 0
scenarios_dir = ""


def on_key_press(event):
    global exit_flag, current_idx, route_files
    if event.key == 'escape':
        exit_flag = True  # ESC退出
    elif event.key == 'delete':  # 按 Delete 键跳过本路线
        skip_route()


def skip_route():
    global current_idx, route_files

    current_idx += 1
    if current_idx >= len(route_files):
        set_title(ax, "All routes processed! Press ESC.")
        plt.draw()
        return

    load_current_route()
    draw(ax)
    set_title(ax, f"Skipped to route {current_idx:02d}")
    plt.draw()


def draw(ax):
    """绘制道路、路线和触发点（优化：只绘制视图范围内的数据）"""
    global center, dist, road_waypoints, selected_route_idx, selected_trigger_idx

    # 过滤可见范围的点
    xmin, xmax = center[0] - dist, center[0] + dist
    ymin, ymax = center[1] - dist, center[1] + dist
    mask = (road_waypoints[:,0] >= xmin) & (road_waypoints[:,0] <= xmax) & \
           (road_waypoints[:,1] >= ymin) & (road_waypoints[:,1] <= ymax)
    visible_wps = road_waypoints[mask]

    ax.clear()  # 清除数据，但保留标题由 set_title 设置

    # 绘制道路点
    ax.plot(visible_wps[:, 0], -visible_wps[:, 1], 'o', color='y', markersize=2)
    for x, y, z, pitch, yaw, roll in visible_wps:
        length = 4
        yaw_rad = yaw / 180 * np.pi
        dx, dy = length * np.cos(yaw_rad), length * np.sin(yaw_rad)
        ax.arrow(x, -y, dx, -dy, color='y', head_width=0.5, alpha=0.5)

    # 绘制路线点
    if selected_route_idx:
        wps = np.take(road_waypoints, selected_route_idx, axis=0)
        ax.plot(wps[:, 0], -wps[:, 1], '-o', color='b', markersize=3)
        ax.plot(wps[0, 0], -wps[0, 1], 'o', color='g')
        ax.text(wps[0, 0] + 8, -wps[0, 1] + 8, "Start", bbox=dict(facecolor='green', alpha=0.7))
        ax.plot(wps[-1, 0], -wps[-1, 1], 'o', color='r')
        ax.text(wps[-1, 0] + 8, -wps[-1, 1] + 8, "End", bbox=dict(facecolor='red', alpha=0.7))

    # 绘制触发点
    if selected_trigger_idx:
        wps = np.take(road_waypoints, selected_trigger_idx, axis=0)
        trigger_wp = wps[0]
        ax.plot(trigger_wp[0], -trigger_wp[1], 'o', color='red')
        ax.text(trigger_wp[0] + 10, -trigger_wp[1] + 10, "Trigger", bbox=dict(facecolor='red', alpha=0.7))
        for actor_wp in wps[1:]:
            ax.plot(actor_wp[0], -actor_wp[1], 'o', color='green')
            ax.text(actor_wp[0] + 8, -actor_wp[1] + 8, "Actor", bbox=dict(facecolor='green', alpha=0.7))
            ax.plot([trigger_wp[0], actor_wp[0]], [-trigger_wp[1], -actor_wp[1]], '--', color='b')

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(-ymax, -ymin)

def set_title(ax, title=None):
    if title is None:
        title = "Left click: select/remove trigger/actor.\nRight click: save scenario & next route.\nScroll: zoom | Middle drag: move."
    ax.set_title(title, fontsize=16, loc='left')


def select_route_waypoints(route_file):
    """加载 route 文件并返回以起点为中心的道路点集和路线索引"""
    global waypoints_sparse, dist
    route_wps = np.load(route_file)
    # 路线起点作为中心
    start_point = route_wps[0, :2]
    center_sel = start_point  # 直接用起点坐标
    # 选取指定范围内的道路点
    road_wps = select_waypoints(waypoints_sparse, center_sel, dist)
    # 找出路线对应的索引
    route_idx = []
    for wp in route_wps:
        idx, _ = get_nearist_waypoints(wp, road_wps)
        route_idx.append(idx)
    return center_sel, road_wps, route_idx


def load_current_route():
    global current_idx, route_files, center, road_waypoints, selected_route_idx, selected_trigger_idx, dist
    center_new, road_wps_new, route_idx = select_route_waypoints(route_files[current_idx])
    center = center_new
    road_waypoints = road_wps_new
    selected_route_idx = route_idx
    selected_trigger_idx = []

    # 加载路线后自动放大（缩小 dist）
    dist *= 0.2   # 比例可调，比如 0.5 表示放大 2 倍视角


def save_scenario(trigger_wps, side_marks):
    """保存当前路线的场景信息"""
    global scenarios_dir, current_idx
    os.makedirs(scenarios_dir, exist_ok=True)
    np.save(os.path.join(scenarios_dir, f"scenario_{current_idx:02d}.npy"), trigger_wps)
    np.save(os.path.join(scenarios_dir, f"scenario_{current_idx:02d}_sides.npy"), side_marks)

def on_click(event):
    """鼠标点击事件"""
    global selected_trigger_idx, current_idx

    if event.xdata is None or event.ydata is None:
        return

    # 左键选择/取消触发点和actor
    if event.button == 1:
        wps_dist = np.linalg.norm(road_waypoints[:, :2] - [event.xdata, -event.ydata], axis=1)
        if wps_dist.min() < 5:
            idx = wps_dist.argmin()
            if idx in selected_trigger_idx:
                selected_trigger_idx.remove(idx)
            else:
                selected_trigger_idx.append(idx)
        draw(ax)
        set_title(ax)
        plt.draw()

    # 右键保存场景并切换到下一条路线
    elif event.button == 3:
        if not selected_trigger_idx:
            set_title(ax, "Need at least 1 trigger point.")
            plt.draw()
            return

        trigger_wps = np.take(road_waypoints, selected_trigger_idx, axis=0)
        # 计算左右侧标记
        start_wp = np.take(road_waypoints, selected_route_idx[0], axis=0)
        end_wp = np.take(road_waypoints, selected_route_idx[-1], axis=0)
        start_pos, end_pos = start_wp[:2], end_wp[:2]
        side_marks = []
        for wp in trigger_wps:
            pos = wp[:2]
            cross = (pos[0]-start_pos[0])*(end_pos[1]-start_pos[1]) - (pos[1]-start_pos[1])*(end_pos[0]-start_pos[0])
            # 不必旋转朝向
            if scenario_id in [3, 4, 5, 6, 7]:
                side_marks.append("center")
            elif cross > 0:
                side_marks.append("left")
                wp[4] = (wp[4] + 90) % 360
            elif cross < 0:
                side_marks.append("right")
                wp[4] = (wp[4] - 90) % 360
            else:
                side_marks.append("center")

        save_scenario(trigger_wps, np.array(side_marks))
        current_idx += 1
        if current_idx >= len(route_files):
            set_title(ax, "All routes processed! Press ESC.")
            plt.draw()
            return
        load_current_route()
        draw(ax)
        set_title(ax, f"Now editing route {current_idx:02d}")
        plt.draw()


# ==== 缩放和拖拽 ====
def on_scroll(event):
    """滚轮缩放（只改变显示范围，不重新筛选waypoints）"""
    global dist
    zoom_factor = 0.9
    if event.button == 'up':
        dist *= zoom_factor
    elif event.button == 'down':
        dist /= zoom_factor
    dist = max(20, min(dist, 2000))
    draw(ax)
    set_title(ax, f"Zoom dist={dist:.1f}")
    plt.draw()


def on_middle_press(event):
    """中键按下开始拖拽"""
    global is_dragging, last_mouse_pos
    if event.button == 2:
        is_dragging = True
        last_mouse_pos = (event.xdata, event.ydata)


def on_motion(event):
    """拖拽地图（只更新显示中心）"""
    global center, last_mouse_pos
    if is_dragging and event.xdata is not None and event.ydata is not None and last_mouse_pos:
        dx = event.xdata - last_mouse_pos[0]
        dy = event.ydata - last_mouse_pos[1]
        center = (center[0] - dx, center[1] + dy)
        last_mouse_pos = (event.xdata, event.ydata)
        draw(ax)
        plt.draw()


def on_middle_release(event):
    """中键释放结束拖拽"""
    global is_dragging
    if event.button == 2:
        is_dragging = False


def main(config):
    global waypoints_sparse, dist, route_files, map_name, scenario_id, scenarios_dir, ax
    map_name = config.map
    scenario_id = config.scenario

    waypoints_sparse = np.load(f"map_waypoints/{map_name}/sparse.npy")
    dist = 1000

    routes_dir = os.path.join("scenario_origin", map_name, f"scenario_{scenario_id:02d}_routes")
    scenarios_dir = os.path.join("scenario_origin", map_name, f"scenario_{scenario_id:02d}_scenarios")
    route_files = sorted([os.path.join(routes_dir, f) for f in os.listdir(routes_dir) if f.startswith("route_")])

    if not route_files:
        print("No route files found!")
        return

    load_current_route()

    plt.ion()  # 开启交互模式
    fig, ax = plt.subplots(figsize=(32, 32))
    draw(ax)
    set_title(ax)

    fig.canvas.mpl_connect('key_press_event', on_key_press)
    fig.canvas.mpl_connect('button_press_event', on_click)
    fig.canvas.mpl_connect('scroll_event', on_scroll)
    fig.canvas.mpl_connect('button_press_event', on_middle_press)
    fig.canvas.mpl_connect('motion_notify_event', on_motion)
    fig.canvas.mpl_connect('button_release_event', on_middle_release)
    fig.canvas.mpl_connect('key_press_event', on_key_press)

    while not exit_flag:
        plt.pause(0.05)  # 频率更高
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--map', type=str, default='1201-ShaTin12D')
    parser.add_argument('--scenario', type=int, default=8)
    args = parser.parse_args()
    main(args)

