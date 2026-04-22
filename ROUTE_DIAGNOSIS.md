# 路由规划错误诊断和修复指南

## 问题描述
错误 `NetworkXNoPath: Node 888 not reachable from 619` 表示在CARLA的全局路由规划器中，无法找到两个连续轨迹点之间的路径。

## 根本原因
这通常由以下原因引起：
1. **轨迹点坐标无效** - 点不在CARLA地图的道路网络上
2. **轨迹点距离过远** - 两个连续点之间的距离超过了可连接的范围
3. **地图断裂** - 道路网络中存在断裂或孤立的区域
4. **z值不正确** - 在多层结构（高架/隧道）中z值过高导致投影到错误的层

## 诊断步骤

### 1. 使用诊断脚本
```bash
# 确保CARLA服务器正在运行
./CarlaUE4.sh -world-port=2000

# 在另一个终端运行诊断脚本
python tools/diagnose_route.py \
    --route_file safebench/scenario/scenario_data/new_central/scenario_05_routes/scenario_05_route_00_weather_00.xml \
    --town center \
    --port 2000
```

这个脚本会：
- 检查每个轨迹点是否能投影到道路
- 检查连续点之间是否可连接
- 显示具体的失败点和坐标

### 2. 检查轨迹点坐标
查看XML文件中的waypoint坐标，确保：
- 坐标在地图范围内
- 相邻点之间的距离合理（通常 < 50米）
- z值设置正确（通常为0或接近地面高度）

### 3. 修复方法

#### 方法A: 修改轨迹点坐标
编辑XML文件，调整无效的轨迹点坐标，确保它们在有效的道路上。

#### 方法B: 使用交互式路由创建工具
```bash
python tools/create_routes.py --map center --town center
```
这个工具允许你在地图上交互式地选择有效的轨迹点。

#### 方法C: 验证现有路由
```bash
python tools/check_route_overlap.py --route_file <your_route_file>
```

## 代码改进
已在 `safebench/scenario/tools/route_manipulation.py` 中添加了更详细的错误信息，现在会显示：
- 失败的轨迹点索引
- 具体的坐标值
- 更清晰的错误消息

## 常见问题

**Q: 为什么z值设置为0？**
A: 在 `route_parser.py` 中，z值被设置为0以确保轨迹点总是投影到地面道路层，避免在多层结构中投影到错误的层。

**Q: 如何验证路由是否有效？**
A: 使用诊断脚本检查所有轨迹点和连接。如果所有点都显示 ✓，则路由有效。

**Q: 轨迹点之间的最大距离是多少？**
A: 通常建议不超过50-100米，具体取决于地图的道路网络密度。
