import json
import argparse
import numpy as np
import os
from utilities import build_scenarios


def create_scenario_hongkong(selected_waypoints, side_marks):
    """
    根据给定的配置、中心点和选定的航路点创建场景。它会计算新的航路点位置，并找到最接近的密集航路点
    """

    all_scenarios_configs = []
    # 直接使用选定的航路点来生成场景配置
    scenario_config = build_scenarios(selected_waypoints, side_marks)
    all_scenarios_configs.append(scenario_config)

    return all_scenarios_configs


def save_scenarios(config, scenarios_configs):
    """
    将场景配置保存为 JSON 文件
    """
    scenario_id = config.scenario
    save_dir = os.path.join(config.save_dir, f"scenarios")
    os.makedirs(save_dir, exist_ok=True)
    save_file = os.path.join(save_dir, f"scenario_{scenario_id:02d}.json")
    scenario_name_list = ["", "DynamicObjectCrossing", "VehicleTurningRoute", "OtherLeadingVehicle",
                          "LaneChange", "OppositeVehicleRunningRedLight", "SignalizedJunctionLeftTurn",
                          "SignalizedJunctionRightTurn", "NoSignalJunctionCrossingRoute"]
    scenario_json = {
        "available_scenarios": [
            {
                config.map: [
                    {
                        "available_event_configurations": scenarios_configs,
                        "scenario_name": scenario_name_list[scenario_id]
                    }
                ]
            }
        ]
    }
    with open(save_file, 'w') as f:
        json.dump(scenario_json, f, indent=2)


def main(config):
    """
    加载地图的密集航路点和中心点，获取需要导出的场景，调用 create_scenario 函数生成场景配置，并调用 save_scenarios 函数保存这些配置
    """
    np.set_printoptions(suppress=True)

    # get scenario that need to export
    if config.scenario < 0:
        map_dir = os.path.join("scenario_origin", config.map)
        # 使用filter函数和lambda表达式筛选出以"_scenarios"结尾的文件或子目录
        scenarios = list(filter(lambda x: x.endswith("_scenarios"), os.listdir(map_dir)))
        # 使用map函数和lambda表达式将列表中的每个元素的第9个到第11个字符转换为整数
        scenarios = list(map(lambda x: int(x[9: 11]), scenarios))
        scenarios.sort()
    else:
        scenarios = [config.scenario]

    for scenario in scenarios:
        config.scenario = scenario
        # load scenarios
        all_scenario_configs = []
        save_dir = os.path.join("scenario_origin", config.map, f"scenario_{config.scenario:02d}_scenarios")
        scenario_file_names = [f for f in os.listdir(save_dir) if f.endswith(".npy") and "sides" not in f]
        scenario_file_names.sort()
        for scenario_file_name in scenario_file_names:
            scenario_file = os.path.join(save_dir, scenario_file_name)
            selected_waypoints = np.load(scenario_file)
            scenario_actor_sides_file = scenario_file.replace(".npy", "_sides.npy")
            side_marks = np.load(scenario_actor_sides_file)
            scenarios_configs = create_scenario_hongkong(selected_waypoints, side_marks)
            all_scenario_configs += scenarios_configs

        # save scenarios
        save_scenarios(config, all_scenario_configs)
        print(f"{len(all_scenario_configs)} scenarios of scenario {scenario} is exported to {save_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--map', type=str, default='center')
    parser.add_argument('--save_dir', type=str, default="scenario_data/center")
    parser.add_argument('--scenario', type=int, default=1)

    args = parser.parse_args()

    main(args)


