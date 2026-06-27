#!/usr/bin/env bash
set -e

# Docker 容器启动入口。
# 每次进入容器或执行脚本前，先加载 ROS Noetic 和 common_msgs 工作区环境，
# 再设置默认 ROS master 地址。这样 rostopic、rosmsg 和 Python ROS 节点
# 都能直接找到消息定义。
source /opt/ros/noetic/setup.bash
source /root/catkin_ws/devel/setup.bash

export ROS_MASTER_URI="${ROS_MASTER_URI:-http://127.0.0.1:11311}"
export ROS_IP="${ROS_IP:-127.0.0.1}"

exec "$@"
