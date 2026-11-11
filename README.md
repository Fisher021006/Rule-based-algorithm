# 车载直控部署包

## 当前项目结构

- `scenarios/`：切入、切出、紧急制动、盲区、交叉口和伴行六套规则场景算法。
- `planning_runtime/scene/`：统一场景上下文、车辆关系几何、路线意图和共享算法加载器。
- `traj_planner/`：局部轨迹规划、时间参数化及外部目标动态限值检查。
- `offline_sim/`：单车和多车共用的离线仿真、车辆推进、可视化与结果保存。

交叉口采用车辆几何中心进入固定圆形冲突区的实际时差评分，默认半径2.5米、
容差0.5秒。同步成绩与后续通行观测分别记录；入区后沿既定路线巡航，双方车身
清空或观察超时才结束执行。过点时差和区域占用间隔作为独立观测。

常用命令（在项目根目录运行）：

```bash
python3 simulate_scene_offline.py --help
```

统一离线入口通过 `--scene-algorithm-symbol` 指定原始场景函数，或通过
`--multi-vehicle-config` 加载多车配置。运行结果保留实际场景环境参数以便重演。
`docs/`和`outputs/`作为本地说明、审查记录及运行产物目录继续被Git忽略。

这个部署包用于在车端直接运行当前算法，不依赖 Kafka，不依赖平台任务接口。

## 包内内容

- `direct_control_node.py`
  车载直控主节点，读取定位、底盘反馈和可选的 OBU 障碍物输入，实时规划并下发 `controlcmd.msg`
- `simulate_rollout.py`
  离线滚动仿真脚本，用于验证起终点和控制量时序
- `simulator/`
- `traj_planner/`
- `drivable_lanes_planner/`
- `global_planner/`
- `utils/`
- `xodr_parser/`
  以上目录为当前算法运行所需的最小代码与地图资源
- `msgs/`
  当前接车用到的消息定义副本
- `requirements-runtime.txt`
  车端运行所需的 Python 依赖
- `create_venv.sh`
  创建 Python 虚拟环境并安装依赖
- `launch_direct_control.sh`
  车载直控节点启动脚本
- `runtime.env.example`
  环境变量模板

## 车上部署流程

1. 将整个部署目录拷贝到车端，例如 `/opt/direct_control_bundle`
2. source 车上的 ROS 环境
3. 按实际 ROS 消息包名修改 `runtime.env`
4. 创建 Python 虚拟环境并安装依赖
5. 启动车载直控节点

## 1. source ROS 环境

示例：

```bash
source /opt/ros/<你的ROS版本>/setup.bash
source <你的catkin_ws>/devel/setup.bash
```

## 2. 配置环境变量

复制模板：

```bash
cp runtime.env.example runtime.env
```

重点修改：

- `ROS_CONTROLCMD_MSG`
- `ROS_SENSORGPS_MSG`
- `ROS_ACTUATOR_MSG`
- `ROS_OBU_MSG`（若要接 OBU）
- `ROS_SENSORGPS_TOPIC`
- `ROS_ACTUATOR_TOPIC`
- `ROS_OBU_TOPIC`
- `ROS_CONTROLCMD_TOPIC`
- `ROS_SCENE_MAIN_PTC_ID`（若需要固定指定某辆 OBU 车作为场景主车）

## 3. 创建虚拟环境

```bash
bash create_venv.sh
```

说明：
- 该脚本会用 `--system-site-packages` 创建虚拟环境，复用车端已安装的 ROS Python 依赖
- 同时补装 `rospkg`、`catkin_pkg` 等运行时依赖，避免 `rospy` 导入失败

## 4. 启动车载直控节点

用当前位置作为起点、终点用投影坐标：

```bash
bash launch_direct_control.sh --use-current-start --goal-xy -429 129
```

固定起终点：

```bash
bash launch_direct_control.sh --start-xy -418 59 --goal-xy -429 129
```

## 5. 离线验证

```bash
source .venv/bin/activate
python simulate_rollout.py --start-xy -418 59 --goal-xy -429 129 --output-prefix test_run
```

## 注意

- 这个包不会替你安装 ROS，本包只处理 Python 运行环境和算法代码
- `controlcmd.msg / sensorgps.msg / actuator.msg` 必须已经在车端 ROS 工作空间中编译成功
- 若要将 OBU 目标送入当前规划器，`obupants.msg / obupant.msg` 也必须在车端 ROS 工作空间中编译成功
- 若车端 `sensorgps` 的真实定义是 `lon=经度`、`lat=纬度`，保持 `ROS_SENSORGPS_SWAP_LATLON=0`
- 只有在车端消息字段实际反着用时，才把 `ROS_SENSORGPS_SWAP_LATLON` 改成 `1`
- 当前默认假设 `sensorgps.velocity` 单位就是 `m/s`，因此 `ROS_SENSORGPS_SPEED_SCALE=1.0`
- 若车端 `sensorgps.velocity` 实际单位是 `km/h`，再把 `ROS_SENSORGPS_SPEED_SCALE` 改成 `0.2777777778`
- `common_msgs/controlcmd.msg` 中 `acceleration` 字段在该协议下承载的是速度指令，单位 `km/h`
- 当前 OBU 输入会优先读取 `pos_lon/pos_lat`，回退 `lon/lat`，并带上 `speed / heading / accel / size / timestamp` 转成内部障碍物
- 默认地图为 `xodr_parser/data/同济测试场.xodr`，并已附带 `xodr_parser/cache/同济测试场.pkl`

## 6. 平台任务接入（新）

`direct_control_node.py` 保持原样，平台接入由 `vehicle_platform_bridge.py` 提供，
用 `--role` 区分五个角色：`mux / task-manager / adapter / algorithm / probe`。

一键启动四个角色（mux、task-manager、adapter、algorithm）：

```bash
source /opt/ros/noetic/setup.bash
source ~/redis_stu/devel/setup.bash
source .venv/bin/activate

# 默认 cut_in，mux 限速 0，车不会动，用于联调
./start_platform_bridge.sh

# 真正控车
./start_platform_bridge.sh --max-speed-kph 25

# 换场景
./start_platform_bridge.sh -a scenarios.blind_spot_algorithm.blind_spot_algorithm

# 查看/停止
./start_platform_bridge.sh status
./start_platform_bridge.sh stop
```

脚本会自动选择终端模式：图形界面用 gnome-terminal 的四个标签页，
SSH 下用 tmux 会话 `bridge`，都没有就后台运行并输出日志到 `logs/`。
`--dry-run` 只打印命令不启动。

注意：

- `mux` 是 `/controlcmd` 的唯一发布者，**不要**同时运行 `launch_direct_control.sh`，
  否则两个节点会抢控制权。
- 任务停止或算法异常时 `mux` 自动回到 HOLD（`acceleration=0, brkreq=0`）。
- 首次上车先用 `--role probe` 打印 `/start_task`、`/taskstatus`、
  `/test_trajectory_req` 的真实内容，再启动四个角色。
