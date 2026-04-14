# SafeBench Tools 设计文档

## 概述

`tools` 文件夹包含了SafeBenchHK自动驾驶安全测试平台的**场景创建和管理工具集**。这些工具支持从地图数据获取、交互式路线/场景编辑、数据导出到可视化验证的完整工作流。

## 核心设计理念

1. **分离关注点**：将数据获取、编辑、导出、可视化分离为独立模块
2. **交互式编辑**：提供GUI工具让用户直观地绘制和编辑测试场景
3. **数据格式转换**：支持多种格式（.npy、.xml、.json）的相互转换
4. **数据一致性**：确保路线和场景数据的对应关系和编号连续性

## 工具架构

```
tools/
├── utilities.py                   # 核心工具函数库
├── get_map_data.py                # 第1步：获取地图数据
├── create_routes.py               # 第2步：交互式创建路线
├── create_scenarios.py            # 第3步：交互式创建场景
├── export_routes.py               # 第4步：导出路线为XML
├── export_scenarios.py            # 第4步：导出场景为JSON
├── export.py                      # 第4步：统一导出管理
├── visualize_routes_scenarios.py  # 第5步：可视化验证
└── map_waypoints/                 # 地图路点数据存储
    └── {map_name}/
        ├── sparse.npy             # 稀疏路点（8m间隔）
        └── dense.npy              # 密集路点（1m间隔）
```

## 工作流程

### 第1步：获取地图数据 (`get_map_data.py`)

**功能**：从CARLA仿真服务器获取地图的路点数据

**输入**：
- CARLA服务器地址和端口（默认localhost:2000）
- 地图名称（如 `1201-ShaTin12D`）

**输出**：
- `map_waypoints/{map_name}/sparse.npy` - 稀疏路点（8m间隔）
- `map_waypoints/{map_name}/dense.npy` - 密集路点（1m间隔）

**路点格式**：`[x, y, z, pitch, yaw, roll]`

**使用示例**：
```bash
python tools/get_map_data.py --map "1201-ShaTin12D" --port 2000
```

**关键函数**：
- `generate_waypoints(world, dist)` - 从CARLA世界生成指定间隔的路点

---

### 第2步：交互式创建路线 (`create_routes.py`)

**功能**：提供交互式GUI让用户在地图上绘制测试路线

**输入**：
- 地图名称
- 场景ID
- 路线ID（可选，-1表示自动编号）

**输出**：
- `scenario_origin/{map_name}/scenario_{id:02d}_routes/route_{id:02d}.npy` - 路线路点

**交互操作**：
| 操作 | 功能 |
|------|------|
| 左键点击 | 选择/取消选择路点 |
| 右键点击 | 保存当前选中的路点为路线 |
| 鼠标滚轮 | 缩放地图视图 |
| 中键拖拽 | 平移地图视图 |
| ESC键 | 退出程序 |

**使用示例**：
```bash
python tools/create_routes.py \
    --map "1201-ShaTin12D" \
    --scenario 8 \
    --route -1 \
    --road auto
```

**关键特性**：
- 支持加载已有路线进行编辑
- 自动检查路线可行性（通过CARLA路径插值）
- 自动重编号路线文件（保证连续性）
- 支持直线和十字路口两种道路类型

**关键函数**：
- `select_waypoints(waypoints, center, distance)` - 选择指定范围内的路点
- `check_routes_is_possible(config)` - 验证路线可行性并清理无效路线

---

### 第3步：交互式创建场景 (`create_scenarios.py`)

**功能**：在已有路线基础上，标记场景的触发点和交通参与者位置

**输入**：
- 地图名称
- 场景ID
- 已创建的路线文件

**输出**：
- `scenario_origin/{map_name}/scenario_{id:02d}_scenarios/scenario_{id:02d}.npy` - 场景路点
- `scenario_origin/{map_name}/scenario_{id:02d}_scenarios/scenario_{id:02d}_sides.npy` - 参与者侧向标记

**交互操作**：
| 操作 | 功能 |
|------|------|
| 左键点击 | 选择/取消选择触发点或actor位置 |
| 右键点击 | 保存当前场景，切换到下一条路线 |
| Delete键 | 跳过当前路线 |
| 鼠标滚轮 | 缩放地图视图 |
| 中键拖拽 | 平移地图视图 |
| ESC键 | 退出程序 |

**使用示例**：
```bash
python tools/create_scenarios.py \
    --map "1201-ShaTin12D" \
    --scenario 8
```

**场景结构**：
- 第一个选中点：**触发点**（红色）- 场景开始执行的位置
- 后续选中点：**Actor位置**（绿色）- 交通参与者的生成位置

**侧向标记**：
- `left` - 参与者在路线左侧
- `right` - 参与者在路线右侧
- `center` - 参与者在路线中心

**关键函数**：
- `select_route_waypoints(route_file)` - 加载路线并获取对应的道路路点
- `save_scenario(trigger_wps, side_marks)` - 保存场景配置

---

### 第4步：导出为标准格式

#### 4a. 导出路线为XML (`export_routes.py`)

**功能**：将.npy格式的路线转换为CARLA标准的XML格式

**输入**：
- 地图名称
- 场景ID（-1表示导出所有场景）
- 源目录（scenario_origin）
- 目标目录（scenario_data）

**输出**：
- `scenario_data/{map_name}/scenario_{id:02d}_routes/scenario_{id:02d}_route_{id:02d}_weather_{id:02d}.xml`
- `scenario_data/{map_name}/standard_scenario_{id:02d}.json` - 元数据

**XML格式示例**：
```xml
<?xml version='1.0' encoding='utf-8'?>
<routes>
  <route id="0" town="1201-ShaTin12D">
    <weather cloudiness="0.0" fog_density="0.0" ... />
    <waypoint x="100.00" y="200.00" z="2.00" pitch="0.00" yaw="90.00" roll="0.00" />
    ...
  </route>
</routes>
```

**使用示例**：
```bash
python tools/export_routes.py \
    --map "1201-ShaTin12D" \
    --save_dir "scenario_data/1201-ShaTin12D" \
    --scenario 8
```

**关键函数**：
- `build_route(waypoints, route_id, town, save_file, weathers)` - 构建XML路线文件
- `create_route_hongkong(config, selected_waypoints)` - 创建香港地图格式的路线

#### 4b. 导出场景为JSON (`export_scenarios.py`)

**功能**：将.npy格式的场景转换为JSON格式的场景配置

**输入**：
- 地图名称
- 场景ID（-1表示导出所有场景）
- 源目录（scenario_origin）
- 目标目录（scenario_data）

**输出**：
- `scenario_data/{map_name}/scenarios/scenario_{id:02d}.json`

**JSON格式示例**：
```json
{
  "available_scenarios": [
    {
      "1201-ShaTin12D": [
        {
          "available_event_configurations": [
            {
              "transform": {
                "x": "100.00",
                "y": "200.00",
                "z": "2.00",
                "pitch": "0.00",
                "yaw": "90.00"
              },
              "other_actors": {
                "left": [...],
                "right": [...],
                "center": [...]
              }
            }
          ],
          "scenario_name": "DynamicObjectCrossing"
        }
      ]
    }
  ]
}
```

**使用示例**：
```bash
python tools/export_scenarios.py \
    --map "1201-ShaTin12D" \
    --save_dir "scenario_data/1201-ShaTin12D" \
    --scenario 8
```

**关键函数**：
- `build_scenarios(waypoints, side_marks)` - 构建场景配置
- `create_scenario_hongkong(selected_waypoints, side_marks)` - 创建香港地图格式的场景

#### 4c. 统一导出管理 (`export.py`)

**功能**：统一管理导出流程，包括数据清理和重编号

**输入**：
- 地图名称
- 源目录（scenario_origin）
- 目标目录（scenario_data）
- 场景ID

**处理步骤**：
1. 清理无效路线（删除没有对应场景的路线）
2. 重新编号路线和场景（保证连续性）
3. 导出路线为XML
4. 导出场景为JSON

**使用示例**：
```bash
python tools/export.py \
    --map "1201-ShaTin12D" \
    --origin_dir "scenario_origin/1201-ShaTin12D" \
    --save_dir "scenario_data/1201-ShaTin12D" \
    --scenario 8
```

**关键函数**：
- `cleanup_unused_routes(scenario_id, origin_dir)` - 清理无效路线并重编号
- `try_remove(path)` - 安全删除文件

---

### 第5步：可视化验证 (`visualize_routes_scenarios.py`)

**功能**：可视化已创建的路线和场景，支持全局和局部视图

**输入**：
- 地图名称
- 场景ID
- 目标目录（scenario_data）
- 可视化模式（route/scenario/both）

**交互操作**：
| 操作 | 功能 |
|------|------|
| 左键点击 | 切换到下一条路线 |
| 右键点击 | 切换到上一条路线 |
| 中键点击 | 切换全局/局部放大视图 |
| ESC键 | 退出程序 |

**使用示例**：
```bash
python tools/visualize_routes_scenarios.py \
    --map "1201-ShaTin12D" \
    --save_dir "scenario_data/1201-ShaTin12D" \
    --scenario 8 \
    --mode both
```

**可视化元素**：
- **黄色点**：道路路点
- **蓝色线**：测试路线
- **绿色点**：路线起点
- **红色点**：路线终点或触发点
- **绿色点**：交通参与者位置
- **蓝色虚线**：触发点到参与者的连接线

---

## 核心工具函数库 (`utilities.py`)

### 路线处理函数

| 函数 | 功能 |
|------|------|
| `build_route(waypoints, route_id, town, save_file, weathers)` | 构建XML格式的路线文件 |
| `parse_route(route_file)` | 解析XML路线文件 |
| `select_waypoints(waypoints, center, distance)` | 选择指定范围内的路点 |
| `rotate_waypoints(origin_waypoints, center, theta)` | 旋转路点 |
| `get_nearist_waypoints(waypoint, waypoints)` | 找到最近的路点 |

### 场景处理函数

| 函数 | 功能 |
|------|------|
| `build_scenarios(waypoints, side_marks)` | 构建场景配置 |
| `parse_scenarios(scenario_config)` | 解析场景配置 |

### 地图处理函数

| 函数 | 功能 |
|------|------|
| `get_map_centers(map_name)` | 获取地图中心点 |
| `get_view_centers(map_name)` | 获取地图视图中心点 |
| `copy_routes_and_scenarios(old_map_name, new_map_name)` | 复制路线和场景到新地图 |

### 几何计算函数

| 函数 | 功能 |
|------|------|
| `compute_magnitude_angle(target_location, current_location, orientation)` | 计算相对角度和距离 |
| `compute_yaw_angle(yaw1, yaw2)` | 计算两个yaw角之间的夹角 |

---

## 数据格式规范

### 路点格式
```
[x, y, z, pitch, yaw, roll]
- x, y, z: 世界坐标（米）
- pitch, yaw, roll: 欧拉角（度）
```

### 天气参数
遵循CARLA官方天气参数规范：
- cloudiness: 云量（0-100）
- precipitation: 降水量（0-100）
- fog_density: 雾密度（0-100）
- wind_intensity: 风强度（0-100）
- wetness: 湿度（0-100）
- 等其他参数

### 场景ID映射
| ID | 场景名称 |
|----|---------|
| 1 | DynamicObjectCrossing |
| 2 | VehicleTurningRoute |
| 3 | OtherLeadingVehicle |
| 4 | LaneChange |
| 5 | OppositeVehicleRunningRedLight |
| 6 | SignalizedJunctionLeftTurn |
| 7 | SignalizedJunctionRightTurn |
| 8 | NoSignalJunctionCrossingRoute |

---

## 完整工作流示例

### 场景创建完整流程

```bash
# 1. 启动CARLA服务器
./CarlaUE4.sh -world-port=2000

# 2. 获取地图数据
python tools/get_map_data.py --map "1201-ShaTin12D" --port 2000

# 3. 交互式创建路线
python tools/create_routes.py \
    --map "1201-ShaTin12D" \
    --scenario 8 \
    --route -1 \
    --road auto

# 4. 交互式创建场景
python tools/create_scenarios.py \
    --map "1201-ShaTin12D" \
    --scenario 8

# 5. 导出为标准格式
python tools/export.py \
    --map "1201-ShaTin12D" \
    --origin_dir "scenario_origin/1201-ShaTin12D" \
    --save_dir "scenario_data/1201-ShaTin12D" \
    --scenario 8

# 6. 可视化验证
python tools/visualize_routes_scenarios.py \
    --map "1201-ShaTin12D" \
    --save_dir "scenario_data/1201-ShaTin12D" \
    --scenario 8 \
    --mode both
```

---

## 目录结构

### 输入数据结构
```
scenario_origin/{map_name}/
├── scenario_{id:02d}_routes/
│   ├── route_00.npy
│   ├── route_01.npy
│   └── ...
└── scenario_{id:02d}_scenarios/
    ├── scenario_00.npy
    ├── scenario_00_sides.npy
    ├── scenario_01.npy
    ├── scenario_01_sides.npy
    └── ...
```

### 输出数据结构
```
scenario_data/{map_name}/
├── scenario_{id:02d}_routes/
│   ├── scenario_{id:02d}_route_00_weather_00.xml
│   ├── scenario_{id:02d}_route_00_weather_01.xml
│   └── ...
├── scenarios/
│   └── scenario_{id:02d}.json
└── standard_scenario_{id:02d}.json
```

---

## 关键设计特性

### 1. 交互式编辑
- 使用Matplotlib提供直观的GUI界面
- 支持缩放、平移、选择等交互操作
- 实时反馈用户操作

### 2. 数据验证
- 自动检查路线可行性（通过CARLA路径插值）
- 验证路线和场景的对应关系
- 清理无效数据

### 3. 数据一致性
- 自动重编号确保文件编号连续
- 维护路线和场景的一一对应关系
- 支持批量导出多个场景

### 4. 灵活的地图支持
- 支持多个地图（香港地图、CARLA标准地图等）
- 可配置的地图中心点
- 支持坐标变换和旋转

### 5. 多格式支持
- .npy格式：高效的二进制存储
- .xml格式：CARLA标准格式
- .json格式：易于解析和配置

---

## 扩展指南

### 添加新地图支持

在 `utilities.py` 的 `get_map_centers()` 函数中添加新地图：

```python
def get_map_centers(map_name):
    if map_name == "new_map":
        centers = np.asarray([[x, y]])
    # ...
    return centers
```

### 添加新场景类型

在 `export_scenarios.py` 的 `scenario_name_list` 中添加新场景名称：

```python
scenario_name_list = [
    "",
    "DynamicObjectCrossing",
    # ... 其他场景
    "NewScenarioType"  # 新场景
]
```

### 自定义天气参数

在 `export_routes.py` 中修改天气参数生成逻辑，或在调用 `build_route()` 时传入自定义天气参数。

---

## 常见问题

### Q: 路线检查失败，提示"路径插值失败"
**A**: 确保CARLA服务器正在运行，且地图已正确加载。检查选中的路点是否在可行驶区域内。

### Q: 导出后XML文件为空
**A**: 检查 `scenario_origin` 目录中是否存在对应的.npy文件，确保路线和场景已正确创建。

### Q: 可视化时看不到路线或场景
**A**: 确保 `scenario_data` 目录中存在导出的XML和JSON文件，检查地图名称是否正确。

### Q: 重编号后文件编号不连续
**A**: 运行 `export.py` 的 `cleanup_unused_routes()` 函数会自动清理和重编号，确保连续性。

---

## 性能考虑

- **稀疏路点**（8m间隔）：用于快速交互和可视化
- **密集路点**（1m间隔）：用于精确路径规划
- **批量导出**：支持一次导出多个场景，提高效率
- **增量编辑**：支持加载已有路线进行编辑，无需重新创建

---

## 总结

tools工具集提供了一个完整、灵活、易用的场景创建和管理系统。通过分离关注点、提供交互式编辑、支持多种格式转换，使得SafeBench用户能够高效地创建和管理自动驾驶测试场景。
