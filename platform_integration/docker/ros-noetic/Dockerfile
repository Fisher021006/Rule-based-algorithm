FROM ros:noetic-ros-base-focal

ENV DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/bash", "-c"]

# 安装构建 ROS 消息包和运行 Python 算法所需的基础工具。
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    python-is-python3 \
    python3-catkin-tools \
    python3-pip \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

# 更新 Python 打包工具，避免安装运行时依赖时遇到旧版本兼容问题。
RUN python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel

# 构建 common_msgs：把项目 msgs 目录里的自定义 ROS 消息编译进镜像。
WORKDIR /root/catkin_ws/src/common_msgs
COPY platform_integration/docker/ros-noetic/common_msgs/package.xml ./package.xml
COPY platform_integration/docker/ros-noetic/common_msgs/CMakeLists.txt ./CMakeLists.txt
COPY msgs ./msg

WORKDIR /root/catkin_ws
RUN source /opt/ros/noetic/setup.bash && catkin_make

# 安装算法和 Adapter 运行时依赖，包括 Redis 客户端和 YAML 配置解析。
COPY requirements-runtime.txt /tmp/requirements-runtime.txt
RUN python3 -m pip install --no-cache-dir -r /tmp/requirements-runtime.txt redis PyYAML

# 容器启动时自动加载 ROS 和 common_msgs 环境。
COPY platform_integration/docker/ros-noetic/entrypoint.sh /ros_entrypoint_direct_control.sh
RUN chmod +x /ros_entrypoint_direct_control.sh

# 项目目录由 docker_ros_shell.sh 挂载到这里。
WORKDIR /workspace/direct_control_bundle
ENTRYPOINT ["/ros_entrypoint_direct_control.sh"]
CMD ["bash"]
