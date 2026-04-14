import json
import argparse
import numpy as np
import os
import shutil
import carla
from utilities import build_route, get_map_centers


def create_route_hongkong(config, selected_waypoints):
    scenario_id = config.scenario
    save_dir = os.path.join(config.save_dir, f"scenario_{scenario_id:02d}_routes")
    os.makedirs(save_dir, exist_ok=True)
    all_routes_real_waypoints = []
    real_route_waypoints = []

    for route_waypoint in selected_waypoints:
        real_route_waypoints.append(np.array(route_waypoint))

    all_routes_real_waypoints.append(np.array(real_route_waypoints))

    return all_routes_real_waypoints


def get_static_public_attributes(cls):
    return {
        key: getattr(cls, key)
        for key in dir(cls)
        if not key.startswith('_')  # 过滤私有属性
        and not callable(getattr(cls, key))  # 过滤方法
        and key in cls.__dict__  # 确保是类级别的静态属性
        and not isinstance(cls.__dict__.get(key), property)  # 过滤掉property属性
    }


def object_to_numeric_dict(obj):
    return {
        key: str(getattr(obj, key))
        for key in dir(obj)
        if not key.startswith('_')  # 过滤私有属性
        and not callable(getattr(obj, key))  # 过滤方法
        and isinstance(getattr(obj, key), (int, float))  # 仅保留数值类型数据
    }


def save_routes(config, save_dir, route, weather_id, weather):
    route_id = 0
    save_file = os.path.join(save_dir, f"scenario_{config.scenario:02d}_route_{route_id:02d}_weather_{weather_id:02d}.xml")
    while os.path.isfile(save_file):
        route_id += 1
        save_file = os.path.join(save_dir, f"scenario_{config.scenario:02d}_route_{route_id:02d}_weather_{weather_id:02d}.xml")

    build_route(route, route_id, config.map, save_file, weathers=weather)
    return route_id


def main(config):
    # 强制NumPy以普通的浮点数格式来显示数值,而不是科学计数法
    np.set_printoptions(suppress=True)

    # scenario route datas
    scenario_route_datas = []
    data_id = 0

    # get scenario that need to export
    if config.scenario < 0:
        map_dir = os.path.join("scenario_origin", config.map)
        scenarios = list(filter(lambda x: x.endswith("_routes"), os.listdir(map_dir)))
        scenarios = list(map(lambda x: int(x[9: 11]), scenarios))  # 解析出功能场景的type
        scenarios.sort()
    else:
        scenarios = [config.scenario]

    # get centers of the map
    centers = get_map_centers(config.map)

    for scenario in scenarios:
        config.scenario = scenario
        # load scenarios
        scenarios_dir = os.path.join("scenario_origin", config.map, f"scenario_{config.scenario:02d}_routes")
        scenario_file_names = list(filter(lambda x: x.endswith('.npy'), os.listdir(scenarios_dir)))
        scenario_file_names.sort()

        # 指定将原始npy文件转存为目标行使路径xml的位置
        save_dir = os.path.join(config.save_dir, f"scenario_{config.scenario:02d}_routes")
        if os.path.isdir(save_dir):
            shutil.rmtree(save_dir)
        os.makedirs(save_dir)

        export_route_num = 0

        for scenario_file_name in scenario_file_names:
            scenario_file = os.path.join(scenarios_dir, scenario_file_name)
            selected_waypoints = np.load(scenario_file)
            # copy waypoints to each center
            for _ in centers:
                # 路径转成xml文件
                scenarios_routes = create_route_hongkong(config, selected_waypoints)

                # Setup weathers refer to the official design
                # https://carla.org/Doxygen/html/db/ddb/classcarla_1_1rpc_1_1WeatherParameters.html#a8286c371c1f7897842a358739cf54abf
                allWeatherParameters = get_static_public_attributes(carla.WeatherParameters)

                # save routes
                for scenarios_route in scenarios_routes:
                    for weather_id, item in enumerate(allWeatherParameters.values()):
                        weather = object_to_numeric_dict(item)
                        route_id = save_routes(config, save_dir, scenarios_route, weather_id=weather_id, weather=weather)

                        # add data to scenario_route_datas
                        data = {
                            "data_id": data_id,
                            "scenario_folder": "standard",
                            "scenario_id": config.scenario,
                            "route_id": route_id,
                            "weather_id": weather_id,
                            "risk_level": None,
                            "parameters": None
                        }

                        scenario_route_datas.append(data)
                        data_id += 1
                        export_route_num += 1

        print(f"{export_route_num} routes of scenario {scenario} is exported to {scenarios_dir}")

    # 保存场景的json配置数据
    os.makedirs(config.save_dir, exist_ok=True)
    scenario_id = config.scenario
    with open(os.path.join(config.save_dir, f'standard_scenario_{scenario_id:02d}.json'), 'w') as f:
        json.dump(scenario_route_datas, f, indent=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--map', type=str, default='center')
    parser.add_argument('--save_dir', type=str, default="scenario_data/center")
    parser.add_argument('--scenario', type=int, default=1)

    args = parser.parse_args()

    main(args)