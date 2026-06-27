cmake_minimum_required(VERSION 3.0.2)
project(common_msgs)

# Docker 镜像内的最小 ROS 消息包，只负责编译算法需要的自定义 msg。
find_package(catkin REQUIRED COMPONENTS
  message_generation
  std_msgs
)

# 这些 msg 来自项目根目录 msgs/，构建镜像时会复制到 common_msgs/msg。
add_message_files(
  FILES
  actuator.msg
  controlcmd.msg
  obupant.msg
  obupants.msg
  sensorgps.msg
)

# 生成 Python/C++ 可用的 ROS 消息代码。
generate_messages(
  DEPENDENCIES
  std_msgs
)

# 对外声明运行时需要 message_runtime，供 Python ROS 节点导入 common_msgs。
catkin_package(
  CATKIN_DEPENDS message_runtime std_msgs
)
