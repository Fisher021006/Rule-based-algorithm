# Redis-ROS Adapter 说明

这个包是“平台 Redis 通信”和“本地 ROS 算法”之间的桥接层。当前实现面向
本地虚拟平台推演：平台下发 A-B 任务，Adapter 启动现有直控算法，虚拟车
根据算法 `/controlcmd` 更新状态，并把轨迹回传给平台显示。

## 1. 当前闭环

当前 MVP 的数据流如下：

```text
平台 Redis 控制通道
  -> Adapter 解析 type=1 准备任务和 type=2 开始任务
  -> Adapter 发布 /sensorgps、/actuator、/obu/source
  -> direct_control_node.py 运行原 ROS 算法
  -> 算法发布 /controlcmd
  -> Adapter 推进虚拟主车
  -> Adapter 回传平台 trajectory / mergeDataChannel
```

核心原则是：不直接改算法主体，先让 Adapter 适配平台协议和 ROS topic。

## 2. 真实平台启动

真实平台固定配置在：

```text
platform_integration/config/platform_real.yaml
```

先启动并保持 ROS Docker 容器运行：

```bash
platform_integration/scripts/docker_ros_shell.sh
```

再另开一个 Mac 终端执行：

```bash
platform_integration/scripts/run_platform_adapter.sh
```

脚本启动后会持续在线。平台下发准备任务时，Adapter 解析起终点并回报准备
状态；平台下发开始任务后，Adapter 才会启动算法。

## 3. 本地 smoke 测试

不接真实平台、只验证本地闭环：

```bash
platform_integration/scripts/run_virtual_redis_adapter_smoke.sh
```

该测试会启动临时 Redis 容器，用 `platform_integration/scripts/mock_platform_ab_task.py` 模拟平台
下发任务，并等待 Adapter 回传非零速度轨迹。

只验证 ROS 算法是否能在 Docker 中跑起来：

```bash
platform_integration/scripts/run_ros_fake_smoke.sh
```

该测试不会连接 Redis，只使用 `platform_integration/scripts/fake_vehicle_ros_node.py` 造
`/sensorgps` 和 `/actuator`。

## 4. 核心文件

- `core/config.py`：读取 YAML/JSON 配置，并转换成 Adapter 命令行参数。
- `core/coordinates.py`：处理平台经纬度、XODR 投影坐标和航向角转换。
- `core/types.py`：定义任务、通道、控制指令、回传车辆状态等中间结构。
- `core/platform_protocol.py`：解析平台 Redis JSON，封装状态和轨迹回传消息。
- `core/direct_control_launcher.py`：启动、停止和监控 `direct_control_node.py`。
- `core/virtual_vehicle.py`：根据 `/controlcmd` 推进本地虚拟主车状态。
- `virtual/virtual_adapter.py`：完整 Adapter 主流程，连接 Redis、ROS 和算法。

## 5. 重要边界

当前 `virtual_vehicle.py` 是本地推演用的虚拟主车，不是实车底盘接口。
后续迁移到车端时，可以复用 `core/config.py`、`core/coordinates.py`、
`core/types.py` 和 `core/platform_protocol.py`，但需要把虚拟车状态更新替换
为真实车端 ROS 定位、底盘和控制回传链路。

## 6. 日志位置

真实平台运行时，算法日志默认写到：

```text
platform_integration/logs/platform_redis_ros_adapter/direct_control_node_task_<task>_case_<case>_<时间戳>.log
```

平台下发的原始控制消息会写到：

```text
platform_integration/logs/platform_redis_ros_adapter/redis_control_payloads/
```

这两个目录都在项目目录下，因此 Mac 侧可以直接查看。
