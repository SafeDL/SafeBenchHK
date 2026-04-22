"""
根据xml文件中定义的起始位置,从预先解析的map中的waypoints中获得从起点到终点的路径
"""

from safebench.carla_agents.navigation.local_planner import RoadOption
import matplotlib
matplotlib.use('TkAgg')
from safebench.carla_agents.navigation.global_route_planner import GlobalRoutePlanner


def interpolate_trajectory(world, waypoints_trajectory, hop_resolution=1.0, grp=None):
    """
        Given some raw keypoints interpolate a full dense trajectory to be used by the user.
            :param world: an reference to the CARLA world so we can use the planner
            :param waypoints_trajectory: the current coarse trajectory
            :param hop_resolution: is the resolution, how dense is the provided trajectory going to be made
            :param grp: optional pre-built GlobalRoutePlanner instance to reuse
            :return: the full interpolated route both in GPS coordinates and also in its original form.
            :raises RouteInterpolationError: if route cannot be interpolated between waypoints
    """
    import networkx as nx

    if grp is None:
        grp = GlobalRoutePlanner(world.get_map(), hop_resolution)
    route = []

    if len(waypoints_trajectory) == 1:
        route.append((waypoints_trajectory[0], RoadOption.VOID))

    for i in range(len(waypoints_trajectory) - 1):   # Goes until the one before the last.
        waypoint = waypoints_trajectory[i]
        waypoint_next = waypoints_trajectory[i + 1]
        try:
            interpolated_trace = grp.trace_route(waypoint, waypoint_next)
            for wp_tuple in interpolated_trace:
                route.append((wp_tuple[0].transform, wp_tuple[1]))
        except nx.NetworkXNoPath as e:
            from safebench.scenario.tools.route_manipulation import RouteInterpolationError
            raise RouteInterpolationError(
                waypoint_idx=i,
                waypoint_1=waypoint,
                waypoint_2=waypoint_next,
                error=str(e)
            ) from e

    return route


class RouteInterpolationError(Exception):
    """异常：无法在轨迹点之间插值路径"""
    def __init__(self, waypoint_idx, waypoint_1, waypoint_2, error):
        self.waypoint_idx = waypoint_idx
        self.waypoint_1 = waypoint_1
        self.waypoint_2 = waypoint_2
        self.error = error
        super().__init__(
            f"无法在轨迹点 {waypoint_idx} 和 {waypoint_idx+1} 之间规划路径\n"
            f"  点 {waypoint_idx}: ({waypoint_1.x:.2f}, {waypoint_1.y:.2f}, {waypoint_1.z:.2f})\n"
            f"  点 {waypoint_idx+1}: ({waypoint_2.x:.2f}, {waypoint_2.y:.2f}, {waypoint_2.z:.2f})\n"
            f"  原始错误: {error}"
        )
