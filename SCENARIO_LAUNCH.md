# 统一场景启动配置

`launch_scenario.py` 使用一份 JSON 配置启动离线仿真或车端直接控制。原有命令行入口仍保留，不影响现有脚本。

## 快速使用

使用默认离线配置：

```bash
python launch_scenario.py
```

指定配置：

```bash
python launch_scenario.py --config configs/launch.json
```

只校验配置并查看实际命令：

```bash
python launch_scenario.py --config configs/launch.json --dry-run
```

配置中的相对路径默认相对项目根目录解析。用户正式场景测试的图片、GIF 和轨迹日志可输出到项目 `outputs/`；开发过程中的临时验证产物应输出到 `/tmp`。`outputs/` 已由 Git 忽略。

## 离线仿真

`launch_mode` 设为 `offline_simulation`。`runtime` 控制地图、步长、最大帧数、可视化和输出；`vehicles` 是不限车辆数量的列表。

每辆车必须有唯一 `id`：

- `kind: "algorithm"`：运行场景算法，需要 `algorithm`、`main_ptc_id`、起终点和车辆参数。
- `kind: "route"`：脚本背景车，`motion` 可选 `idm`、`constant`、`static` 或 `scripted_lane_change`。
- `environment`：只对该辆算法车生效的 `ROS_SCENE_*` 参数，多车之间互不污染。

增加车辆时，直接在 `vehicles` 数组中增加一个对象。多辆伴行车就配置多个 `algorithm` 对象，每辆车使用独立 `id`、起终点、伴行方位和 `rel_s` 窗口。

### 多车顶层字段

- `vehicles`：车辆配置列表，`id` 必须唯一。
- `primary_vehicle_id`：用于结果展示的主场景算法车 ID。
- `stop_on_terminal_vehicle_ids`：这些算法车进入终态时允许结束仿真。
- `stop_on_terminal_mode`：`all` 表示列表内全部车进入终态，`any` 表示任一车进入终态。
- `terminal_scene_statuses`：可选，覆盖默认场景终态集合。

未配置 `stop_on_terminal_vehicle_ids` 时，必须通过 `runtime.max_steps` 限制仿真时长。

Cut-in 完成换道并稳定后会闭锁输出 `cutin_success`，同时继续沿当前
车道巡航，不会因为场景完成自动停车。如果需要在 cut-in 成功时结束整轮离线
仿真，将该车 ID 加入 `stop_on_terminal_vehicle_ids`；保持空列表则可继续观察伴行车恢复。
旧的单算法车离线命令没有多车终止列表，检测到 `cutin_success` 后会直接结束仿真循环。

### Algorithm 车

```json
{
  "id": "companion_gvt",
  "kind": "algorithm",
  "algorithm": "scenarios.companion_algorithm.companion_algorithm",
  "main_ptc_id": "vut",
  "start_xy": [0.0, 0.0],
  "goal_xy": [100.0, 0.0],
  "start_speed": 5.0,
  "max_speed_kph": 25.0,
  "environment": {
    "ROS_SCENE_COMPANION_POSITION": "left_front"
  }
}
```

每辆 Algorithm 车都有独立的算法模块、`CFG`、`SCENE_MEMORY`、`SceneTargetProvider` 和轨迹规划器。`main_ptc_id` 必须指向同一配置中的另一辆车。

### Route 车

```json
{
  "id": "vut",
  "kind": "route",
  "motion": "constant",
  "start_xy": [0.0, 0.0],
  "goal_xy": [100.0, 0.0],
  "start_speed": 5.0
}
```

`motion` 支持 `idm`、`constant`、`static` 和 `scripted_lane_change`。`static` 车只需配置 `static_xy`。

### 同步规划语义

每个仿真步先冻结全部车辆状态，再为每辆 Algorithm 车构造包含其他全部车辆的障碍物/OBU 输入。所有车辆完成本帧规划后才统一推进，避免后规划车辆读到前一辆车下一帧状态的顺序偏差。

结果 JSON 中：

- `vehicle_traces` 按车辆 ID 保存轨迹。
- `scene_targets` 按算法车 ID 保存状态机和 ExternalTarget 记录。
- `final_scene_statuses` 保存每辆算法车结束时的状态。
- `obstacle_input_history` 保存每帧全车冻结状态。
- `timing_frames[].vehicles` 分车记录场景算法和局部规划耗时。

## 车端直控

`launch_mode` 设为 `direct_control`。直控配置按职责分为：

- `runtime`：地图、起终点、规划频率、到达阈值和轨迹日志。
- `ros`：定位、底盘、OBU、控制和轨迹话题。
- `vehicle`：本车 ID、尺寸和运动约束。
- `scene`：场景算法、伴行主车 ID/路线和场景环境参数。

[直控示例](configs/direct_control.json) 默认配置了 `dry_run: true` 和 `confirm_real_vehicle: false`，因此只会校验并打印命令。确认车辆、地图、起终点、ROS 话题和场景参数后，必须同时改为：

```json
{
  "dry_run": false,
  "confirm_real_vehicle": true
}
```

否则启动器拒绝实际启动 `direct_control_node.py`。

## 配置原则
- "vehicle": {
    "self_ptc_id": "companion_gvt（示例需要自行根据车端配置，用途是在车辆重叠时防止把自己当成其他车辆）",
    "length_mm": 4640.0,
    "width_mm": 1780.0,
    "wheelbase_mm": 2700.0,
    "max_speed_kph": 72.0,
    "max_acc": 3.0,
    "min_acc": 2.0
  },
  "scene": 
    "algorithm": "scenarios.companion_algorithm.companion_algorithm",
    "main_ptc_id": "vut（主车id在测试的时候显式配置，留空默认选择距离最近的OBU车辆）",
    "main_start_xy": [
      305.985199,
      241.739304
    ],
    "main_goal_xy": [
      158.323736,
      251.74311
    ],
- JSON 布尔值必须写为 `true`/`false`，不能写成字符串。
- 离线多车中，每辆 `algorithm` 车的 `main_ptc_id` 必须指向 `vehicles` 中已定义的其他车辆。
- `stop_on_terminal_vehicle_ids` 只填需要决定整轮仿真何时结束的主场景算法车，伴行车通常不填。
- 车端直控一份配置只启动一辆算法车；多辆实车由各车使用各自的 JSON 和启动器实例。
