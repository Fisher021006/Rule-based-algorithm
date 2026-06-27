# ROS Noetic Docker 环境说明

这个目录定义了本项目在 macOS Docker Desktop 上运行 ROS Noetic 的基础环境。
它的目标是让算法、fake 车辆、Redis-ROS Adapter 都在同一个 Ubuntu/ROS
容器里运行，避免 macOS 原生安装 ROS 带来的兼容和网络问题。

## 1. 构建镜像

在项目根目录执行：

```bash
platform_integration/scripts/docker_ros_build.sh
```

该命令会构建镜像：

```text
direct-control-ros:noetic
```

镜像里包含：

- ROS Noetic 基础运行环境；
- 由 `msgs/*.msg` 编译出来的 `common_msgs` ROS 消息包；
- `requirements-runtime.txt` 中的 Python 运行时依赖；
- Redis Python 客户端和 PyYAML，供平台 Adapter 连接 Redis、读取配置。

## 2. 进入容器

在项目根目录执行：

```bash
platform_integration/scripts/docker_ros_shell.sh
```

进入后，Mac 项目目录会挂载到容器内：

```text
/workspace/direct_control_bundle
```

这意味着你在 Mac 上改的文件，容器里能立即看到；容器里写到项目目录下的日志，
Mac Finder 或编辑器也能直接打开。

## 3. ROS 环境加载

容器入口脚本是：

```text
platform_integration/docker/ros-noetic/entrypoint.sh
```

它会自动执行：

```bash
source /opt/ros/noetic/setup.bash
source /root/catkin_ws/devel/setup.bash
```

第一句加载 ROS Noetic 自带命令，第二句加载镜像里编译好的 `common_msgs`。
如果没有第二句，`rosmsg show common_msgs/sensorgps`、`rostopic echo` 和 Python
节点导入消息类型都可能失败。

## 4. 基础检查

容器内可以检查自定义消息是否存在：

```bash
rosmsg show common_msgs/sensorgps
rosmsg show common_msgs/actuator
rosmsg show common_msgs/controlcmd
rosmsg show common_msgs/obupants
```

启动 ROS master：

```bash
roscore
```

另开一个 Mac 终端进入同一个容器：

```bash
docker exec -it direct-control-ros bash
```

然后查看当前 ROS topic：

```bash
rostopic list
```

## 5. 只验证 ROS 算法闭环

该测试不连接 Redis，只证明算法能在 Docker ROS 中收 fake 车辆输入并输出
`/controlcmd`：

```text
fake_vehicle_ros_node.py -> /sensorgps, /actuator
direct_control_node.py   -> /controlcmd
fake_vehicle_ros_node.py <- /controlcmd
```

容器内执行：

```bash
platform_integration/scripts/run_ros_fake_smoke.sh
```

默认路线：

```text
start_xy = -418.0, 59.0
goal_xy  = -429.0, 129.0
mode     = local_sampling
```

需要改起终点时可以用环境变量：

```bash
START_X=-418 START_Y=59 GOAL_X=-429 GOAL_Y=129 platform_integration/scripts/run_ros_fake_smoke.sh
```

日志和临时地图缓存默认写到：

```text
/tmp/direct_control_fake_smoke
```

## 6. 本地 Redis-ROS Adapter smoke 测试

该测试使用临时 Redis 容器和 mock 平台，不连接真实平台：

```text
mock 平台 -> Redis 控制通道
Adapter   -> /sensorgps, /actuator, /obu/source
算法      -> /controlcmd
Adapter   -> Redis 上传通道
```

先保持 `direct-control-ros` 容器运行，再从 Mac 项目根目录执行：

```bash
platform_integration/scripts/run_virtual_redis_adapter_smoke.sh
```

看到下面输出代表本地闭环跑通：

```text
[mock] PASS
[smoke] PASS
```

## 7. 真实平台 Adapter 启动

真实平台参数写在：

```text
platform_integration/config/platform_real.yaml
```

从 Mac 项目根目录执行：

```bash
platform_integration/scripts/run_platform_adapter.sh
```

该脚本会进入正在运行的 ROS 容器，确认 `roscore` 可用，然后启动
`platform_integration/scripts/run_virtual_redis_adapter.py --config platform_integration/config/platform_real.yaml`。
平台下发任务后，Adapter 才会启动 `direct_control_node.py`。

## 8. 文件职责

- `Dockerfile`：构建 Ubuntu 20.04 + ROS Noetic 镜像，安装依赖并编译消息包。
- `entrypoint.sh`：容器启动时自动加载 ROS 和 `common_msgs` 环境。
- `common_msgs/CMakeLists.txt`：声明需要编译哪些 `.msg` 文件。
- `common_msgs/package.xml`：声明 ROS 消息包元信息和依赖。
- `README.md`：说明 Docker 环境、构建、进入容器、测试和真实平台启动流程。
