"""Redis-ROS Adapter 内部共享的数据结构。

这些 dataclass 是平台协议、虚拟车辆、算法启动器之间的中间表示。
它们把平台 JSON 里的字段先整理成稳定的 Python 对象，避免其它模块直接
到处读 `payload["params"]` 这类易变结构。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TaskPoint:
    """平台任务轨迹点。

    A-B 自由行驶最关心起点和终点；后续如果平台下发途经点，本结构也可以
    承接 `site_type=2` 的中间点。`raw` 保留原始字段，方便排查协议差异。
    """

    latitude: float
    longitude: float
    course_angle: float = 0.0
    speed_kph: float = 0.0
    site_type: int = -1
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class PreparedTask:
    """平台 type=1 准备任务解析后的结果。

    Adapter 收到该对象后会保存当前任务，并在任务开始时把 start/goal
    传给 `direct_control_node.py`。如果 `ready` 为假，说明平台消息里没有
    同时提供起点和终点，不能启动算法。
    """

    task_id: str = ""
    case_id: str = ""
    record_id: str = ""
    start: TaskPoint | None = None
    goal: TaskPoint | None = None
    waypoints: list[TaskPoint] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def ready(self) -> bool:
        """判断任务是否具备启动 A-B 算法所需的起点和终点。"""

        return self.start is not None and self.goal is not None


@dataclass
class ProtocolChannels:
    """平台 type=2 控制消息里携带的通道信息。

    背景车通道用于订阅平台实时车辆状态；上传通道用于回传本车轨迹；
    mergeDataChannel 用于平台需要融合显示时的合并数据回传。
    """

    background_channels: list[str] = field(default_factory=list)
    upload_channel: str = ""
    monitor_channel: str = ""
    merge_data_channel: str = ""
    task_id: str = ""
    case_id: str = ""
    record_id: str = ""
    raw_protocols: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class TaskControl:
    """平台任务开始/停止控制命令。

    `task_type=1` 表示开始推演，`task_type=0` 表示结束推演。该对象同时携带
    平台分配的数据通道，Adapter 必须先解析通道再决定往哪里回传轨迹。
    """

    task_type: int
    channels: ProtocolChannels
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class ControlCommandState:
    """算法通过 `/controlcmd` 输出的控制指令快照。

    当前项目里 `controlcmd.acceleration` 实际承载目标速度，单位是 km/h；
    虚拟车辆会根据这个目标速度和转角做简化运动学更新。
    """

    timestamp_ms: int = 0
    target_speed_kph: float = 0.0
    wheel_angle_deg: float = 0.0
    gear: int = 4
    turnlight: int = 0
    brkreq: int = 0


@dataclass
class VehicleReportState:
    """回传平台所需的车辆状态。

    该结构是虚拟车状态、算法控制命令和平台回传协议之间的统一出口。
    `platform_protocol.py` 会把它封装成平台要求的 trajectory/merge JSON。
    """

    timestamp_ms: int
    vehicle_id: str
    name: str
    latitude: float
    longitude: float
    course_angle: float
    speed_kph: float
    lon_acc: float
    lat_acc: float
    length_cm: int
    width_cm: int
    height_cm: int
    wheel_angle_deg: float
    acceleration_mps2: float
    command: ControlCommandState = field(default_factory=ControlCommandState)
