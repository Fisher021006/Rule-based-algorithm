#!/usr/bin/env python3
"""本地 Docker smoke 测试用的 fake ROS 车辆节点。

这个节点模拟 `direct_control_node.py` 启动所需的最小车端 ROS 环境：

- 持续发布有效定位状态的 `/sensorgps`
- 持续发布健康底盘状态的 `/actuator`
- 订阅 `/controlcmd`，用简化运动学自行车模型更新车辆位置

它只用于证明算法在 Docker ROS 中能跑通，不是物理仿真器，也不会连接平台
Redis。
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
from pyproj import CRS, Transformer


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.tools import clean_proj_string
from xodr_parser.xodr_parser import XODRParser


DEFAULT_MAP_PATH = PROJECT_ROOT / "xodr_parser" / "data" / "同济测试场.xodr"
GPS_TO_VEHICLE_CENTER_OFFSET_M = 0.5


def _get_env(*names: str, default: str = "") -> str:
    """按顺序读取多个环境变量，返回第一个非空值。"""

    for name in names:
        value = os.getenv(name)
        if value not in (None, ""):
            return value
    return default


def build_arg_parser() -> argparse.ArgumentParser:
    """创建 fake 车辆节点的命令行参数解析器。"""

    parser = argparse.ArgumentParser(description="发布 direct_control_node.py 所需的 fake 车端 ROS 输入")
    parser.add_argument("--map-path", default=str(DEFAULT_MAP_PATH), help="OpenDRIVE 地图路径")
    parser.add_argument("--cache-path", default="", help="可选的 XODR 解析缓存路径")
    parser.add_argument("--rebuild-cache", action="store_true", help="启动前重新构建地图缓存")
    parser.add_argument("--start-xy", nargs=2, type=float, default=(-418.0, 59.0), metavar=("X", "Y"))
    parser.add_argument("--start-latlon", nargs=2, type=float, metavar=("LAT", "LON"))
    parser.add_argument("--start-heading-deg", type=float, default=None, help="GPS 航向角，正北为 0 度，顺时针增加")
    parser.add_argument("--start-speed-mps", type=float, default=0.0)
    parser.add_argument("--hz", type=float, default=20.0)
    parser.add_argument("--wheelbase-m", type=float, default=2.7)
    parser.add_argument("--max-accel-mps2", type=float, default=1.5)
    parser.add_argument("--max-decel-mps2", type=float, default=3.0)
    parser.add_argument("--max-speed-kph", type=float, default=25.0)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument(
        "--sensorgps-topic",
        default=_get_env("ROS_SENSORGPS_TOPIC", "ROS_GVT_GPS_TOPIC", default="/sensorgps"),
    )
    parser.add_argument(
        "--actuator-topic",
        default=_get_env("ROS_ACTUATOR_TOPIC", "ROS_GVT_ACTUATOR_TOPIC", default="/actuator"),
    )
    parser.add_argument(
        "--controlcmd-topic",
        default=_get_env("ROS_CONTROLCMD_TOPIC", "ROS_CONTROL_TOPIC", default="/controlcmd"),
    )
    parser.add_argument("--node-name", default="fake_vehicle_ros_node")
    return parser


def resolve_cache_path(map_path: Path, cache_path: str) -> Path:
    """确定 fake 节点使用的 XODR 缓存路径。"""

    if cache_path:
        return Path(cache_path)
    return map_path.parent.parent / "cache" / f"{map_path.stem}.pkl"


def load_or_build_map(map_path_str: str, cache_path_str: str, rebuild_cache: bool) -> XODRParser:
    """加载或重新构建 fake 节点需要的 OpenDRIVE 地图解析结果。"""

    map_path = Path(map_path_str)
    cache_path = resolve_cache_path(map_path, cache_path_str)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists() and not rebuild_cache:
        try:
            return XODRParser.load_cache(str(cache_path))
        except (ModuleNotFoundError, AttributeError, ValueError, ImportError):
            pass
    parser = XODRParser(str(map_path))
    parser.save_cache(str(cache_path))
    return parser


def make_transformers(xodr_parser: XODRParser) -> tuple[Transformer, Transformer]:
    """根据 XODR 投影信息创建经纬度和平面坐标的双向转换器。"""

    proj_crs = CRS.from_proj4(clean_proj_string(xodr_parser.geo_reference["proj"]))
    wgs84_crs = CRS.from_epsg(4326)
    to_proj = Transformer.from_crs(wgs84_crs, proj_crs, always_xy=True)
    to_latlon = Transformer.from_crs(proj_crs, wgs84_crs, always_xy=True)
    return to_proj, to_latlon


def gps_heading_deg_to_internal_rad(heading_deg: float) -> float:
    """把 GPS 航向角转换成内部数学坐标系航向角。"""

    return math.radians(90.0 - float(heading_deg))


def internal_rad_to_gps_heading_deg(theta_rad: float) -> float:
    """把内部数学坐标系航向角转换成 GPS 航向角。"""

    return (90.0 - math.degrees(float(theta_rad))) % 360.0


def initial_heading_from_nearest_lane(xodr_parser: XODRParser, start_xy: np.ndarray) -> float:
    """根据起点最近车道中心线估计初始航向角。"""

    best_line = None
    best_dist = float("inf")
    point = np.asarray(start_xy, dtype=float)
    for line in xodr_parser.lane_centerlines.values():
        arr = np.asarray(line, dtype=float)
        if arr.ndim != 2 or len(arr) < 2:
            continue
        distances = np.linalg.norm(arr[:, :2] - point, axis=1)
        idx = int(np.argmin(distances))
        dist = float(distances[idx])
        if dist < best_dist:
            best_dist = dist
            best_line = arr
    if best_line is None:
        return 0.0
    distances = np.linalg.norm(best_line[:, :2] - point, axis=1)
    idx = int(np.argmin(distances))
    if idx >= len(best_line) - 1:
        idx = max(0, len(best_line) - 2)
    vec = best_line[idx + 1, :2] - best_line[idx, :2]
    return float(math.atan2(float(vec[1]), float(vec[0])))


class FakeVehicleNode:
    """最小 ROS 车辆环境。

    该类发布算法需要的定位和底盘状态，接收算法输出的 `/controlcmd`，并用
    简化模型更新下一帧位置。
    """

    def __init__(self, args: argparse.Namespace) -> None:
        """初始化地图、起点、ROS publisher/subscriber 和车辆状态。"""

        import rospy
        from common_msgs.msg import actuator, controlcmd, sensorgps

        self.rospy = rospy
        self.SensorGpsMsg = sensorgps
        self.ActuatorMsg = actuator
        self.ControlCmdMsg = controlcmd
        self.args = args

        self.xodr_parser = load_or_build_map(args.map_path, args.cache_path, args.rebuild_cache)
        self.transformer_to_proj, self.transformer_to_latlon = make_transformers(self.xodr_parser)

        if args.start_latlon is not None:
            lat = float(args.start_latlon[0])
            lon = float(args.start_latlon[1])
            x, y = self.transformer_to_proj.transform(lon, lat)
            self.x = float(x)
            self.y = float(y)
        else:
            self.x = float(args.start_xy[0])
            self.y = float(args.start_xy[1])

        if args.start_heading_deg is None:
            self.heading = initial_heading_from_nearest_lane(self.xodr_parser, np.array([self.x, self.y], dtype=float))
        else:
            self.heading = gps_heading_deg_to_internal_rad(float(args.start_heading_deg))

        self.v = max(0.0, float(args.start_speed_mps))
        self.a = 0.0
        self.target_speed_mps = self.v
        self.wheel_angle_rad = 0.0
        self.brake_requested = False
        self.frame_id = 0
        self.last_update_time = time.monotonic()

        rospy.init_node(args.node_name)
        self.gps_pub = rospy.Publisher(args.sensorgps_topic, self.SensorGpsMsg, queue_size=10)
        self.actuator_pub = rospy.Publisher(args.actuator_topic, self.ActuatorMsg, queue_size=10)
        rospy.Subscriber(args.controlcmd_topic, self.ControlCmdMsg, self._control_callback, queue_size=20)

    def _control_callback(self, msg) -> None:
        """接收算法控制指令，并转换成 fake 车辆的目标速度和转角。"""

        # 当前项目里，controlcmd.acceleration 字段实际承载目标速度，单位 km/h。
        target_kph = float(getattr(msg, "acceleration", 0.0) or 0.0)
        if getattr(msg, "brkreq", 0):
            target_kph = 0.0
            self.brake_requested = True
        else:
            self.brake_requested = False
        target_kph = max(0.0, min(float(self.args.max_speed_kph), target_kph))
        self.target_speed_mps = target_kph / 3.6
        self.wheel_angle_rad = math.radians(float(getattr(msg, "wheel_angle", 0.0) or 0.0))

    def _step(self, dt: float) -> None:
        """按时间步长推进 fake 车辆运动状态。"""

        if dt <= 0.0:
            return
        speed_error = self.target_speed_mps - self.v
        accel_limit = float(self.args.max_accel_mps2) if speed_error >= 0.0 else float(self.args.max_decel_mps2)
        if self.brake_requested:
            accel_limit = max(accel_limit, float(self.args.max_decel_mps2))
        dv = max(-accel_limit * dt, min(accel_limit * dt, speed_error))
        old_v = self.v
        self.v = max(0.0, self.v + dv)
        self.a = (self.v - old_v) / dt

        delta = max(math.radians(-30.0), min(math.radians(30.0), self.wheel_angle_rad))
        if abs(delta) > 1e-5 and self.v > 1e-4:
            self.heading += (self.v / max(float(self.args.wheelbase_m), 1e-3)) * math.tan(delta) * dt
            self.heading = (self.heading + math.pi) % (2.0 * math.pi) - math.pi

        self.x += self.v * math.cos(self.heading) * dt
        self.y += self.v * math.sin(self.heading) * dt

    def _publish_gps(self, now_ms: int) -> None:
        """发布一帧 `/sensorgps` 定位消息。"""

        msg = self.SensorGpsMsg()
        gps_x = self.x + GPS_TO_VEHICLE_CENTER_OFFSET_M * math.cos(self.heading)
        gps_y = self.y + GPS_TO_VEHICLE_CENTER_OFFSET_M * math.sin(self.heading)
        lon, lat = self.transformer_to_latlon.transform(gps_x, gps_y)
        msg.lon = float(lon)
        msg.lat = float(lat)
        msg.alt = 0.0
        msg.roadtype = 0
        msg.lanetype = 0
        msg.heading = float(internal_rad_to_gps_heading_deg(self.heading))
        msg.pitch = 0.0
        msg.roll = 0.0
        msg.pitchrate = 0.0
        msg.rollrate = 0.0
        msg.yawrate = 0.0
        msg.accx = float(self.a)
        msg.accy = 0.0
        msg.accz = 0.0
        msg.mile = 0.0
        msg.velocity = float(self.v)
        msg.status = 4
        msg.rawstatus = 50
        msg.satenum = 20
        msg.gpstime = int(now_ms)
        msg.isvalid = 1
        msg.timestamp = int(now_ms)
        msg.speedE = float(self.v * math.cos(self.heading))
        msg.speedN = float(self.v * math.sin(self.heading))
        msg.speedD = 0.0
        self.gps_pub.publish(msg)

    def _publish_actuator(self, now_ms: int) -> None:
        """发布一帧 `/actuator` 底盘状态消息。"""

        msg = self.ActuatorMsg()
        msg.speed = float(self.v * 3.6)
        msg.isvalid = 1
        msg.timestamp = int(now_ms)
        msg.gear = 4
        msg.epb = 1
        msg.wireControlStatus = 1
        msg.accx = float(self.a)
        msg.brkMode = 1 if self.brake_requested else 0
        msg.ehbWorkMode = 5
        msg.epdWorkMode = 5
        msg.epsWorkMode = 5
        msg.vehWorkMode = 5
        msg.error = 0
        msg.errorlevel = 0
        msg.epsangle = float(math.degrees(self.wheel_angle_rad))
        msg.SteerSpeed = float(self.v * 3.6)
        msg.velAngle = 1 if self.v > 0.05 else 0
        self.actuator_pub.publish(msg)

    def run(self) -> None:
        """进入 ROS 循环，持续更新车辆并发布 fake 输入。"""

        rate = self.rospy.Rate(max(float(self.args.hz), 1e-3))
        while not self.rospy.is_shutdown():
            now = time.monotonic()
            dt = now - self.last_update_time
            self.last_update_time = now
            self._step(dt)

            now_ms = int(time.time() * 1000)
            self._publish_gps(now_ms)
            self._publish_actuator(now_ms)

            self.frame_id += 1
            if self.args.log_every > 0 and self.frame_id % int(self.args.log_every) == 0:
                self.rospy.loginfo(
                    "fake_vehicle x=%.2f y=%.2f heading_deg=%.1f v=%.2f target_v=%.2f wheel_deg=%.2f",
                    self.x,
                    self.y,
                    internal_rad_to_gps_heading_deg(self.heading),
                    self.v,
                    self.target_speed_mps,
                    math.degrees(self.wheel_angle_rad),
                )
            rate.sleep()


def main() -> None:
    """脚本入口：解析参数并启动 fake 车辆节点。"""

    args = build_arg_parser().parse_args()
    node = FakeVehicleNode(args)
    node.run()


if __name__ == "__main__":
    main()
