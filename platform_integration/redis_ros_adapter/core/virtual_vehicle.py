"""本地平台推演用的轻量虚拟车辆模型。

真实平台联调时，本地电脑没有真实底盘反馈。本模块模拟一个理想底盘：
接收算法发布的 `/controlcmd` 速度指令和前轮角指令，用运动学自行车模型
前向积分得到车辆状态，再把该状态反馈给 ROS 算法并回传平台 4.2。
它只用于本地平台推演，不等价于真实车辆动力学。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from redis_ros_adapter.core.coordinates import (
    MapProjector,
    heading_diff_deg,
    internal_rad_to_platform_heading_deg,
    platform_heading_deg_to_internal_rad,
)
from redis_ros_adapter.core.types import ControlCommandState, TaskPoint, VehicleReportState


GPS_TO_VEHICLE_CENTER_OFFSET_M = 0.5


@dataclass
class VirtualVehicleConfig:
    """虚拟车辆的尺寸和运动约束配置。"""

    vehicle_id: str = "virtual_ego"
    name: str = "virtual_ego"
    length_cm: int = 464
    width_cm: int = 178
    height_cm: int = 150
    wheelbase_m: float = 2.7
    max_accel_mps2: float = 1.5
    max_decel_mps2: float = 3.0
    max_speed_kph: float = 25.0
    trajectory_advance_index: int = 1


@dataclass
class VirtualVehicle:
    """根据算法输出推进的虚拟主车状态。

    Adapter 会周期性调用 `step()` 更新车辆，再调用 `gps_latlon()` 发布
    `/sensorgps`，调用 `to_report_state()` 回传平台 trajectory。平台推演
    只用 `/controlcmd` 作为车辆模型输入；`/planning/trajectory` 仅作为误差
    评估参考，不直接推动车辆。
    """

    projector: MapProjector
    config: VirtualVehicleConfig = field(default_factory=VirtualVehicleConfig)
    x: float = 0.0
    y: float = 0.0
    heading: float = 0.0
    v: float = 0.0
    a: float = 0.0
    kappa: float = 0.0
    command: ControlCommandState = field(default_factory=ControlCommandState)
    target_speed_mps: float = 0.0
    wheel_angle_rad: float = 0.0
    heading_source: str = "task_course_angle"

    def reset_from_task_point(self, point: TaskPoint) -> None:
        """按平台任务起点重置虚拟车状态。

        如果平台起点航向角与最近车道方向差异过大，会使用最近车道方向作为
        初始航向，避免一开始车头朝错方向导致算法立即打大转角或停车。
        """

        self.x, self.y = self.projector.latlon_to_xy(point.latitude, point.longitude)
        task_heading_deg = float(point.course_angle)
        lane_heading, lane_dist = self.projector.nearest_lane_heading(self.x, self.y)
        if lane_heading is not None and lane_dist <= 5.0:
            lane_heading_deg = internal_rad_to_platform_heading_deg(lane_heading)
            diff_deg = heading_diff_deg(task_heading_deg, lane_heading_deg)
            if diff_deg > 30.0:
                self.heading = lane_heading
                self.heading_source = "nearest_lane"
                print(
                    "[虚拟车辆] 使用最近车道方向覆盖平台起点航向："
                    f"task_courseAngle={task_heading_deg:.2f}deg lane={lane_heading_deg:.2f}deg "
                    f"diff={diff_deg:.2f}deg lane_dist={lane_dist:.2f}m",
                    flush=True,
                )
            else:
                self.heading = platform_heading_deg_to_internal_rad(task_heading_deg)
                self.heading_source = "task_course_angle"
        else:
            self.heading = platform_heading_deg_to_internal_rad(task_heading_deg)
            self.heading_source = "task_course_angle"
        self.v = max(0.0, float(point.speed_kph) / 3.6)
        self.a = 0.0
        self.kappa = 0.0
        self.target_speed_mps = self.v
        self.wheel_angle_rad = 0.0
        self.command = ControlCommandState()

    def apply_control_command(self, command: ControlCommandState) -> None:
        """接收算法 `/controlcmd` 指令并更新虚拟车目标速度和转角。"""

        self.command = command
        if int(command.brkreq):
            target_kph = 0.0
        else:
            target_kph = max(0.0, min(float(self.config.max_speed_kph), float(command.target_speed_kph)))
        self.target_speed_mps = target_kph / 3.6
        self.wheel_angle_rad = max(math.radians(-30.0), min(math.radians(30.0), math.radians(float(command.wheel_angle_deg))))

    def _step_control_integrator(self, dt: float) -> None:
        """用 `/controlcmd` 驱动理想运动学自行车模型。"""

        if dt <= 0.0:
            return

        old_v = self.v
        if int(self.command.brkreq) or int(self.command.gear) != 4:
            self.v = 0.0
        else:
            self.v = max(0.0, float(self.target_speed_mps))
        self.a = (self.v - old_v) / dt

        wheelbase = max(float(self.config.wheelbase_m), 1e-3)
        delta = max(math.radians(-30.0), min(math.radians(30.0), self.wheel_angle_rad))
        self.kappa = math.tan(delta) / wheelbase if abs(delta) > 1e-6 else 0.0
        delta_heading = self.v * self.kappa * dt
        next_heading = (self.heading + delta_heading + math.pi) % (2.0 * math.pi) - math.pi
        if abs(self.kappa) > 1e-6:
            self.x += (math.sin(self.heading + delta_heading) - math.sin(self.heading)) / self.kappa
            self.y += (-math.cos(self.heading + delta_heading) + math.cos(self.heading)) / self.kappa
        else:
            self.x += self.v * math.cos(self.heading) * dt
            self.y += self.v * math.sin(self.heading) * dt
        self.heading = next_heading

    def step(self, dt: float) -> None:
        """按时间步长推进虚拟车辆运动状态。

        平台推演始终使用算法实际下发的 `/controlcmd` 作为输入，前向积分出
        车辆状态。
        """

        self._step_control_integrator(dt)

    def gps_latlon(self) -> tuple[float, float]:
        """返回用于发布 `/sensorgps` 的 GPS 点经纬度。

        原车端 GPS 点与车辆几何中心存在约 0.5 米偏移，这里保持同样偏移关系，
        尽量贴近原算法对定位输入的假设。
        """

        gps_x = self.x + GPS_TO_VEHICLE_CENTER_OFFSET_M * math.cos(self.heading)
        gps_y = self.y + GPS_TO_VEHICLE_CENTER_OFFSET_M * math.sin(self.heading)
        return self.projector.xy_to_latlon(gps_x, gps_y)

    def center_latlon(self) -> tuple[float, float]:
        """返回车辆几何中心经纬度，用于平台轨迹回传。"""

        return self.projector.xy_to_latlon(self.x, self.y)

    def to_report_state(self, timestamp_ms: int) -> VehicleReportState:
        """把虚拟车当前状态转换成平台回传结构。"""

        lat, lon = self.center_latlon()
        return VehicleReportState(
            timestamp_ms=int(timestamp_ms),
            vehicle_id=str(self.config.vehicle_id),
            name=str(self.config.name),
            latitude=float(lat),
            longitude=float(lon),
            course_angle=float(internal_rad_to_platform_heading_deg(self.heading)),
            speed_kph=float(self.v * 3.6),
            lon_acc=float(self.a),
            lat_acc=float(self.v * self.v * self.kappa),
            length_cm=int(self.config.length_cm),
            width_cm=int(self.config.width_cm),
            height_cm=int(self.config.height_cm),
            wheel_angle_deg=float(math.degrees(self.wheel_angle_rad)),
            acceleration_mps2=float(self.a),
            command=self.command,
        )
