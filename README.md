# 车载直控部署包

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
