# 平台接入目录说明

这个目录集中放置“平台 Redis 接入 + Docker ROS 联调 + 本地虚拟域控推演”相关文件。
算法主体仍然保留在项目根目录，例如 `direct_control_node.py`、`planning_runtime/`、
`scenarios/`、`traj_planner/` 等都不放进这里。

## 目录职责

- `config/`：真实平台 Redis、设备 ID、ROS topic、算法启动参数配置。
- `redis_ros_adapter/`：平台 Redis 和本地 ROS 算法之间的 Adapter 源码。
- `scripts/`：构建 Docker、进入容器、启动真实平台 Adapter、运行 smoke 测试的脚本。
- `docker/`：ROS Noetic Docker 镜像、common_msgs 编译配置和容器入口脚本。
- `logs/`：真实平台任务和 Adapter 运行过程中生成的日志。

## 为什么 `msgs/` 没放进来

`msgs/` 是项目级 ROS 消息定义目录，Docker 构建时会从项目根目录复制它并编译成
`common_msgs`。这些消息也可能被算法和后续车端部署复用，所以保留在项目根目录更合适。

## 为什么 `.dockerignore` 还在根目录

Docker build context 是项目根目录，Docker 只读取 build context 根目录下的
`.dockerignore`。如果把它移进 `platform_integration/`，Docker 构建时不会生效。

## 常用命令

构建 ROS Docker 镜像：

```bash
platform_integration/scripts/docker_ros_build.sh
```

进入 ROS Docker 容器：

```bash
platform_integration/scripts/docker_ros_shell.sh
```

检查真实平台 Redis 是否能连上：

```bash
platform_integration/scripts/check_platform_redis.sh
```

启动真实平台 Adapter：

```bash
platform_integration/scripts/run_platform_adapter.sh
```

只验证 ROS 算法闭环：

```bash
platform_integration/scripts/run_ros_fake_smoke.sh
```

验证本地 Redis-ROS Adapter 闭环：

```bash
platform_integration/scripts/run_virtual_redis_adapter_smoke.sh
```

