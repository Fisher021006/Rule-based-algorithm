"""车载直控节点。

目标：
1. 不再依赖 Kafka、任务消息和预制轨迹。
2. 直接基于指定地图、起点和终点建立全局路径。
3. 实时读取车端 ROS 定位 / 底盘反馈。
4. 调用现有规划器生成局部轨迹。
5. 将轨迹 [x, y, heading, v, a, kappa] 转换成 controlcmd.msg 并直接下发。
"""

import argparse
import json
from dataclasses import dataclass
import importlib
import os
import threading
import time
from pathlib import Path

import numpy as np
from pyproj import CRS, Transformer

from simulator.vehicle import GVT
from utils.tools import clean_proj_string
from xodr_parser.xodr_parser import XODRParser


DEFAULT_MAP_PATH = str((Path(__file__).resolve().parent / "xodr_parser" / "data" / "同济测试场.xodr"))
OBU_SELF_FILTER_DIST_M = 1.0
GPS_TO_VEHICLE_CENTER_OFFSET_M = 0.5
PLANNING_FAILURE_GRACE_S = 0.5
CONTROL_PREVIEW_INDEX = 5


def import_symbol(symbol_path):
    module_name, symbol_name = symbol_path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, symbol_name)


def _get_env(*names, default=""):
    for name in names:
        value = os.getenv(name)
        if value not in (None, ""):
            return value
    return default


def _get_bool_env(*names, default=False):
    value = _get_env(*names, default="")
    if value == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def resolve_message_type(env_var, default_candidates, display_name):
    env_symbol = os.getenv(env_var)
    candidates = [env_symbol] if env_symbol else []
    candidates.extend(default_candidates)

    last_error = None
    for candidate in candidates:
        try:
            return import_symbol(candidate)
        except Exception as exc:  # pragma: no cover - 运行期依赖 ROS 环境
            last_error = exc

    raise ImportError(
        f"无法导入 {display_name} 的 Python 消息类，请设置环境变量 {env_var}。"
    ) from last_error


def resolve_optional_message_type(env_var, default_candidates):
    env_symbol = os.getenv(env_var)
    candidates = [env_symbol] if env_symbol else []
    candidates.extend(default_candidates)

    for candidate in candidates:
        try:
            return import_symbol(candidate)
        except Exception:  # pragma: no cover - 运行期依赖 ROS 环境
            continue
    return None


def resolve_ros_types():
    try:
        import rospy
    except ImportError as exc:
        raise ImportError("未找到 rospy，请先 source 对应 ROS 环境。") from exc

    control_msg_cls = resolve_message_type(
        "ROS_CONTROLCMD_MSG",
        ["common_msgs.msg.controlcmd", "controlcmd.msg.controlcmd", "vehicle_msgs.msg.controlcmd", "chassis_msgs.msg.controlcmd"],
        "controlcmd",
    )
    gps_msg_cls = resolve_message_type(
        "ROS_SENSORGPS_MSG",
        ["common_msgs.msg.sensorgps", "sensorgps.msg.sensorgps", "vehicle_msgs.msg.sensorgps", "chassis_msgs.msg.sensorgps"],
        "sensorgps",
    )
    actuator_msg_cls = resolve_message_type(
        "ROS_ACTUATOR_MSG",
        ["common_msgs.msg.actuator", "actuator.msg.actuator", "vehicle_msgs.msg.actuator", "chassis_msgs.msg.actuator"],
        "actuator",
    )
    obu_msg_cls = resolve_optional_message_type(
        "ROS_OBU_MSG",
        ["common_msgs.msg.obupants", "obupants.msg.obupants", "vehicle_msgs.msg.obupants"],
    )
    trajectory_msg_cls = resolve_message_type(
        "ROS_TRAJECTORY_MSG",
        ["std_msgs.msg.String"],
        "planned trajectory",
    )
    return rospy, control_msg_cls, gps_msg_cls, actuator_msg_cls, obu_msg_cls, trajectory_msg_cls


def build_arg_parser():
    parser = argparse.ArgumentParser(description="基于 OpenDRIVE 地图的车载直控节点")
    parser.add_argument("--map-path", default=DEFAULT_MAP_PATH, help="xodr 地图文件路径")
    parser.add_argument("--cache-path", default="", help="缓存 pkl 路径；为空时按地图名自动生成")
    parser.add_argument("--rebuild-cache", action="store_true", help="启动时强制重建地图缓存")

    parser.add_argument("--goal-xy", nargs=2, type=float, metavar=("X", "Y"), help="终点投影坐标")
    parser.add_argument("--goal-latlon", nargs=2, type=float, metavar=("LAT", "LON"), help="终点经纬度")

    parser.add_argument("--start-xy", nargs=2, type=float, metavar=("X", "Y"), help="起点投影坐标")
    parser.add_argument("--start-latlon", nargs=2, type=float, metavar=("LAT", "LON"), help="起点经纬度")
    parser.add_argument("--use-current-start", action="store_true", help="使用车辆当前位置作为起点")

    parser.add_argument(
        "--sensorgps-topic",
        "--gps-topic",
        dest="sensorgps_topic",
        default=_get_env("ROS_SENSORGPS_TOPIC", "ROS_GVT_GPS_TOPIC", default="/sensorgps"),
        help="自车定位话题",
    )
    parser.add_argument(
        "--actuator-topic",
        default=_get_env("ROS_ACTUATOR_TOPIC", "ROS_GVT_ACTUATOR_TOPIC", default="/actuator"),
        help="底盘反馈话题",
    )
    parser.add_argument(
        "--obu-topic",
        default=_get_env("ROS_OBU_TOPIC", default="/obu/source"),
        help="OBU 障碍物话题",
    )
    parser.add_argument(
        "--controlcmd-topic",
        "--control-topic",
        dest="controlcmd_topic",
        default=_get_env("ROS_CONTROLCMD_TOPIC", "ROS_CONTROL_TOPIC", default="/controlcmd"),
        help="控车指令话题",
    )
    parser.add_argument(
        "--trajectory-topic",
        default=_get_env("ROS_TRAJECTORY_TOPIC", default="/planning/trajectory"),
        help="规划轨迹话题；本地平台推演用 std_msgs/String(JSON) 发布完整轨迹",
    )
    parser.add_argument("--node-name", default=_get_env("ROS_NODE_NAME", default="direct_control_node"))

    parser.add_argument("--plan-hz", type=float, default=float(_get_env("ROS_PLAN_HZ", default="10.0")))
    parser.add_argument("--control-hz", type=float, default=float(_get_env("ROS_CONTROL_HZ", default="50.0")))
    parser.add_argument(
        "--planner-mode",
        choices=["scene_target", "local_sampling"],
        default=_get_env("ROS_PLANNER_MODE", default="scene_target"),
        help="车端规划入口模式：scene_target 先跑场景算法，local_sampling 直接跑局部采样",
    )
    parser.add_argument("--arrive-dist", type=float, default=1.5, help="判定到达终点的距离阈值，单位 m")
    parser.add_argument("--arrive-speed", type=float, default=0.3, help="判定到达终点的速度阈值，单位 m/s")
    parser.add_argument(
        "--obu-track-timeout-ms",
        type=int,
        default=int(_get_env("ROS_OBU_TRACK_TIMEOUT_MS", default="1000")),
        help="OBU 障碍物保留时长，超过该时长的历史目标将被丢弃",
    )
    parser.add_argument(
        "--scene-main-ptc-id",
        default=_get_env("ROS_SCENE_MAIN_PTC_ID", default=""),
        help="场景算法锁定的主车 ptc_id；为空时按默认规则选取 OBU 车辆",
    )
    parser.add_argument(
        "--scene-algorithm-symbol",
        default=_get_env("ROS_SCENE_ALGORITHM_SYMBOL", default="scenarios.scene_algorithm.scene_algorithm"),
        help="scene_target 模式下使用的场景算法符号路径，例如 scenarios.cut_in_algorithm.cut_in_algorithm",
    )

    parser.add_argument(
        "--swap-latlon",
        dest="swap_latlon",
        action="store_true",
        default=_get_bool_env("ROS_SENSORGPS_SWAP_LATLON", "ROS_GVT_SWAP_LATLON", default=False),
        help="若 sensorgps 的 lat/lon 字段定义反了则开启",
    )
    parser.add_argument(
        "--no-swap-latlon",
        dest="swap_latlon",
        action="store_false",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--speed-scale",
        type=float,
        default=float(_get_env("ROS_SENSORGPS_SPEED_SCALE", default="1.0")),
        help="定位速度换算到 m/s 的缩放系数；若 sensorgps.velocity 原始就是 m/s，则保持 1.0",
    )

    parser.add_argument("--vehicle-length-mm", type=float, default=4640.0)
    parser.add_argument("--vehicle-width-mm", type=float, default=1780.0)
    parser.add_argument("--wheelbase-mm", type=float, default=2700.0)
    parser.add_argument("--max-speed-kph", type=float, default=25.0)
    parser.add_argument("--max-acc", type=float, default=3.0)
    parser.add_argument("--min-acc", type=float, default=2.0)
    return parser


def resolve_cache_path(map_path_str, cache_path_str):
    map_path = Path(map_path_str)
    if cache_path_str:
        return Path(cache_path_str)
    return map_path.parent.parent / "cache" / f"{map_path.stem}.pkl"


def load_or_build_map(map_path_str, cache_path_str, rebuild_cache):
    map_path = Path(map_path_str)
    cache_path = resolve_cache_path(map_path_str, cache_path_str)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if cache_path.exists() and not rebuild_cache:
        parser = XODRParser.load_cache(str(cache_path))
    else:
        parser = XODRParser(str(map_path))
        parser.save_cache(str(cache_path))
    return parser, cache_path


def make_transformers(xodr_parser):
    proj_crs = CRS.from_proj4(clean_proj_string(xodr_parser.geo_reference["proj"]))
    wgs84_crs = CRS.from_epsg(4326)
    to_proj = Transformer.from_crs(wgs84_crs, proj_crs, always_xy=True)
    to_latlon = Transformer.from_crs(proj_crs, wgs84_crs, always_xy=True)
    return to_proj, to_latlon


def parse_point_args(name, xy_value, latlon_value, transformer_to_proj):
    if xy_value is not None:
        return np.array([float(xy_value[0]), float(xy_value[1])], dtype=float)
    if latlon_value is not None:
        lat = float(latlon_value[0])
        lon = float(latlon_value[1])
        x, y = transformer_to_proj.transform(lon, lat)
        return np.array([x, y], dtype=float)
    raise ValueError(f"缺少 {name} 坐标，请提供 --{name}-xy 或 --{name}-latlon。")


def _get_first_attr(obj, names, default=None):
    for name in names:
        if isinstance(obj, dict):
            value = obj.get(name, None)
            if value is not None:
                return value
            continue
        if hasattr(obj, name):
            value = getattr(obj, name)
            if value is not None:
                return value
    return default


def _safe_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value, default=0):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _set_first_attr(obj, names, value):
    for name in names:
        if hasattr(obj, name):
            setattr(obj, name, value)
            return True
    return False


@dataclass
class ObstacleState:
    ptc_id: str
    x: float
    y: float
    v: float
    heading: float = 0.0
    a: float = 0.0
    length: float = 0.0
    width: float = 0.0
    height: float = 0.0
    timestamp_ms: int = 0
    source: int = 0
    ptc_type: int = 0
    source_id: str = ""


class DirectControlNode:
    def __init__(self, args):
        self.args = args
        (
            self.rospy,
            self.ControlCmdMsg,
            self.SensorGpsMsg,
            self.ActuatorMsg,
            self.ObuMsg,
            self.TrajectoryMsg,
        ) = resolve_ros_types()
        self.rospy.init_node(self.args.node_name)

        self.xodr_parser, self.cache_path = load_or_build_map(args.map_path, args.cache_path, args.rebuild_cache)
        self.transformer_to_proj, self.transformer_to_latlon = make_transformers(self.xodr_parser)

        self.gps_lock = threading.Lock()
        self.actuator_lock = threading.Lock()
        self.obu_lock = threading.Lock()
        self.command_lock = threading.Lock()
        self.pose_ready_event = threading.Event()

        self.latest_gps = None
        self.latest_actuator = None
        self.latest_obu = None
        self.latest_obstacles = {}
        self.latest_command = self.build_hold_command()
        self._last_valid_plan_ts = 0.0
        self._last_plan_failure_log_ts = 0.0

        self.goal_xy = parse_point_args("goal", args.goal_xy, args.goal_latlon, self.transformer_to_proj)
        self.start_xy = None
        self.goal_reached = False
        self.route_initialized = False
        self.running = True

        outline = {
            "height": args.vehicle_length_mm,
            "width": args.vehicle_width_mm,
            "wheelbase": args.wheelbase_mm,
        }
        sport_params = {
            "maxSpeed": args.max_speed_kph,
            "maxAcc": args.max_acc,
            "minAcc": args.min_acc,
        }
        self.gvt = GVT("ego", outline, sport_params, self.xodr_parser)
        self.gvt.set_planner_mode(self.args.planner_mode)
        self.gvt.scene_target_provider.set_main_vehicle_ptc_id(self.args.scene_main_ptc_id)
        self.gvt.scene_target_provider.set_scene_algorithm_symbol(self.args.scene_algorithm_symbol)
        self.gvt.scene_target_provider.set_runtime_dt(1.0 / max(float(self.args.plan_hz), 1e-3))
        self.gvt.config["local_fsm"]["arrive_complete_dist"] = float(self.args.arrive_dist)
        self.gvt.config["local_fsm"]["arrive_complete_speed"] = float(self.args.arrive_speed)

        self.control_pub = self.rospy.Publisher(args.controlcmd_topic, self.ControlCmdMsg, queue_size=10)
        self.trajectory_pub = None
        if self.args.trajectory_topic:
            self.trajectory_pub = self.rospy.Publisher(args.trajectory_topic, self.TrajectoryMsg, queue_size=10)
        self.rospy.Subscriber(args.sensorgps_topic, self.SensorGpsMsg, self._gps_callback, queue_size=50)
        self.rospy.Subscriber(args.actuator_topic, self.ActuatorMsg, self._actuator_callback, queue_size=50)
        if self.ObuMsg is not None and self.args.obu_topic:
            self.rospy.Subscriber(args.obu_topic, self.ObuMsg, self._obu_callback, queue_size=50)
            print(f"📡 已启用 OBU 输入: topic={self.args.obu_topic}, msg={self.ObuMsg.__module__}.{self.ObuMsg.__name__}")
        else:
            print("ℹ️ 未启用 OBU 输入：未解析到 obupants 消息类或未配置 topic")

        self.plan_thread = threading.Thread(target=self.run_planning_loop, daemon=True)
        self.control_thread = threading.Thread(target=self.run_control_loop, daemon=True)

    def _build_gps_payload(self, msg):
        lat = _safe_float(_get_first_attr(msg, ["lon"], 0.0), 0.0) if self.args.swap_latlon else _safe_float(_get_first_attr(msg, ["lat"], 0.0), 0.0)
        lon = _safe_float(_get_first_attr(msg, ["lat"], 0.0), 0.0) if self.args.swap_latlon else _safe_float(_get_first_attr(msg, ["lon"], 0.0), 0.0)
        timestamp = _safe_int(_get_first_attr(msg, ["timestamp"], 0), 0)
        if timestamp <= 0:
            timestamp = _safe_int(_get_first_attr(msg, ["gpstime"], 0), 0)
        return {
            "timestamp": timestamp,
            "latitude": lat,
            "longitude": lon,
            "speed": _safe_float(_get_first_attr(msg, ["velocity"], 0.0), 0.0) * self.args.speed_scale,
            "heading": _safe_float(_get_first_attr(msg, ["heading"], 0.0), 0.0),
            "altitude": _safe_float(_get_first_attr(msg, ["alt"], 0.0), 0.0),
            "pitch": _safe_float(_get_first_attr(msg, ["pitch"], 0.0), 0.0),
            "roll": _safe_float(_get_first_attr(msg, ["roll"], 0.0), 0.0),
            "roadtype": _safe_int(_get_first_attr(msg, ["roadtype"], 0), 0),
            "lanetype": _safe_int(_get_first_attr(msg, ["lanetype"], 0), 0),
            "pitchrate": _safe_float(_get_first_attr(msg, ["pitchrate"], 0.0), 0.0),
            "rollrate": _safe_float(_get_first_attr(msg, ["rollrate"], 0.0), 0.0),
            "yawrate": _safe_float(_get_first_attr(msg, ["yawrate"], 0.0), 0.0),
            "accx": _safe_float(_get_first_attr(msg, ["accx"], 0.0), 0.0),
            "accy": _safe_float(_get_first_attr(msg, ["accy"], 0.0), 0.0),
            "accz": _safe_float(_get_first_attr(msg, ["accz"], 0.0), 0.0),
            "mile": _safe_float(_get_first_attr(msg, ["mile"], 0.0), 0.0),
            "status": _safe_int(_get_first_attr(msg, ["status"], 0), 0),
            "rawstatus": _safe_int(_get_first_attr(msg, ["rawstatus"], 0), 0),
            "satenum": _safe_int(_get_first_attr(msg, ["satenum"], 0), 0),
            "isvalid": _safe_int(_get_first_attr(msg, ["isvalid"], 0), 0),
        }

    def _gps_callback(self, msg):
        payload = self._build_gps_payload(msg)
        with self.gps_lock:
            if self.latest_gps is None or payload["timestamp"] >= self.latest_gps["timestamp"]:
                self.latest_gps = payload
                self.pose_ready_event.set()

    def _build_actuator_payload(self, msg):
        return {
            "timestamp": _safe_int(_get_first_attr(msg, ["timestamp"], 0), 0),
            "speed": _safe_float(_get_first_attr(msg, ["speed"], 0.0), 0.0),
            "accx": _safe_float(_get_first_attr(msg, ["accx"], 0.0), 0.0),
            "gaspedalcar": _safe_float(_get_first_attr(msg, ["gaspedalcar"], 0.0), 0.0),
            "brkMode": _safe_int(_get_first_attr(msg, ["brkMode"], 0), 0),
            "vehWorkMode": _safe_int(_get_first_attr(msg, ["vehWorkMode"], 0), 0),
            "ehbWorkMode": _safe_int(_get_first_attr(msg, ["ehbWorkMode"], 0), 0),
            "BrkPressure": _safe_int(_get_first_attr(msg, ["BrkPressure"], 0), 0),
            "epd": _safe_int(_get_first_attr(msg, ["epd"], 0), 0),
            "epdWorkMode": _safe_int(_get_first_attr(msg, ["epdWorkMode"], 0), 0),
            "epsWorkMode": _safe_int(_get_first_attr(msg, ["epsWorkMode"], 0), 0),
            "epsangle": _safe_float(_get_first_attr(msg, ["epsangle"], 0.0), 0.0),
            "SteerSpeed": _safe_float(_get_first_attr(msg, ["SteerSpeed"], 0.0), 0.0),
            "velAngle": _safe_float(_get_first_attr(msg, ["velAngle"], 0.0), 0.0),
            "error": _safe_int(_get_first_attr(msg, ["error"], 0), 0),
            "errorlevel": _safe_int(_get_first_attr(msg, ["errorlevel"], 0), 0),
        }

    def _actuator_callback(self, msg):
        payload = self._build_actuator_payload(msg)
        with self.actuator_lock:
            self.latest_actuator = payload

    def _build_obu_participant_payload(self, participant):
        ptc_id_raw = _get_first_attr(participant, ["ptc_id", "ptcid", "id"], None)
        source_id = str(_get_first_attr(participant, ["source_id", "sourceId"], "") or "")
        track_id = str(ptc_id_raw) if ptc_id_raw not in ("", None) else source_id
        return {
            "track_id": track_id,
            "ptc_id": _safe_int(ptc_id_raw, 0) if ptc_id_raw not in ("", None) else 0,
            "ptc_type": _safe_int(_get_first_attr(participant, ["ptc_type"], 0), 0),
            "source": _safe_int(_get_first_attr(participant, ["source"], 0), 0),
            "source_id": source_id,
            "sec_mark": _safe_int(_get_first_attr(participant, ["sec_mark"], 0), 0),
            "pos_lon": _safe_float(_get_first_attr(participant, ["pos_lon"], 0.0), 0.0),
            "pos_lat": _safe_float(_get_first_attr(participant, ["pos_lat"], 0.0), 0.0),
            "lon": _safe_float(_get_first_attr(participant, ["lon", "longitude"], 0.0), 0.0),
            "lat": _safe_float(_get_first_attr(participant, ["lat", "latitude"], 0.0), 0.0),
            "speed": max(0.0, _safe_float(_get_first_attr(participant, ["speed"], 0.0), 0.0)),
            "heading": _safe_float(_get_first_attr(participant, ["heading"], 0.0), 0.0),
            "accel": _safe_float(_get_first_attr(participant, ["accel"], 0.0), 0.0),
            "width": max(0.0, _safe_float(_get_first_attr(participant, ["width"], 0.0), 0.0)),
            "length": max(0.0, _safe_float(_get_first_attr(participant, ["length"], 0.0), 0.0)),
            "height": max(0.0, _safe_float(_get_first_attr(participant, ["height"], 0.0), 0.0)),
            "isvalid": _safe_int(_get_first_attr(participant, ["isvalid", "is_valid"], 1), 1),
        }

    def _build_obu_payload(self, msg):
        participants = _get_first_attr(msg, ["pants", "participants", "obupants", "objects", "tracks"], [])
        timestamp_ms = _safe_int(_get_first_attr(msg, ["timestamp"], 0), 0)
        if timestamp_ms <= 0:
            timestamp_ms = int(time.time() * 1000)
        payload = {
            "timestamp": timestamp_ms,
            "isvalid": _safe_int(_get_first_attr(msg, ["isvalid"], 1), 1),
            "msg_cnt": _safe_int(_get_first_attr(msg, ["msg_cnt"], len(participants)), len(participants)),
            "participants": [],
        }
        for participant in participants or []:
            payload["participants"].append(self._build_obu_participant_payload(participant))
        return payload

    def _normalize_obu_participant(self, participant, default_timestamp_ms):
        ptc_id = str(participant.get("track_id", "") or "")
        if not ptc_id:
            return None
        if participant.get("isvalid", 1) == 0:
            return None

        lon = participant.get("pos_lon", 0.0)
        lat = participant.get("pos_lat", 0.0)
        if abs(lon) <= 1e-6 or abs(lat) <= 1e-6:
            lon = participant.get("lon", 0.0)
            lat = participant.get("lat", 0.0)
        if abs(lon) <= 1e-6 and abs(lat) <= 1e-6:
            return None

        x, y = self.transformer_to_proj.transform(float(lon), float(lat))
        heading_deg = float(participant.get("heading", 0.0))
        return ObstacleState(
            ptc_id=ptc_id,
            x=float(x),
            y=float(y),
            v=float(participant.get("speed", 0.0)),
            heading=float(np.radians(90.0 - heading_deg)),
            a=float(participant.get("accel", 0.0)),
            length=float(participant.get("length", 0.0)),
            width=float(participant.get("width", 0.0)),
            height=float(participant.get("height", 0.0)),
            timestamp_ms=int(default_timestamp_ms),
            source=int(participant.get("source", 0)),
            ptc_type=int(participant.get("ptc_type", 0)),
            source_id=str(participant.get("source_id", "")),
        )

    def _obu_callback(self, msg):
        payload = self._build_obu_payload(msg)
        if payload["isvalid"] == 0:
            return

        updated = {}
        for participant in payload["participants"]:
            obstacle = self._normalize_obu_participant(participant, payload["timestamp"])
            if obstacle is None:
                continue
            updated[obstacle.ptc_id] = obstacle

        with self.obu_lock:
            self.latest_obu = payload
            for ptc_id, obstacle in updated.items():
                prev = self.latest_obstacles.get(ptc_id)
                if prev is None or obstacle.timestamp_ms >= prev.timestamp_ms:
                    self.latest_obstacles[ptc_id] = obstacle

    @staticmethod
    def build_hold_command():
        return {
            "timestamp": int(time.time() * 1000),
            "speed_kph": 0.0,
            "front_wheel_angle_deg": 0.0,
            "gear": 4,
            "turnlight": 0,
            "brkreq": 1,
        }

    def evaluate_actuator_ready(self):
        with self.actuator_lock:
            actuator = self.latest_actuator
        if actuator is None:
            return True, "no_feedback"
        if actuator["error"] != 0:
            return False, f"actuator_error_{actuator['error']}"
        if actuator["vehWorkMode"] not in (4, 5):
            return False, f"veh_mode_{actuator['vehWorkMode']}"
        if actuator["ehbWorkMode"] not in (4, 5):
            return False, f"ehb_mode_{actuator['ehbWorkMode']}"
        if actuator["epsWorkMode"] not in (4, 5):
            return False, f"eps_mode_{actuator['epsWorkMode']}"
        return True, "ok"

    def current_pose_to_xy_heading(self):
        with self.gps_lock:
            gps = dict(self.latest_gps) if self.latest_gps else None
        if gps is None:
            return None
        # 车端实测存在 isvalid=0、但 status=4 / rawstatus=50 依然有效的情况。
        if gps["rawstatus"] != 50 and gps["status"] != 4:
            return None
        heading = np.radians(90.0 - gps["heading"])
        x, y = self.transformer_to_proj.transform(gps["longitude"], gps["latitude"])
        # GPS 安装点位于车辆几何中心前方 0.5m，沿车头方向回退到车体中心。
        x -= GPS_TO_VEHICLE_CENTER_OFFSET_M * np.cos(heading)
        y -= GPS_TO_VEHICLE_CENTER_OFFSET_M * np.sin(heading)
        return {
            "x": float(x),
            "y": float(y),
            "heading": float(heading),
            "speed": float(gps["speed"]),
            "acc": float(gps["accx"]),
            "timestamp": int(gps["timestamp"]),
        }

    def initialize_route_if_needed(self):
        if self.route_initialized:
            return

        if self.args.use_current_start:
            if not self.pose_ready_event.wait(timeout=10.0):
                raise RuntimeError("等待车辆当前定位超时，无法使用当前位置作为起点。")
            pose = self.current_pose_to_xy_heading()
            if pose is None:
                raise RuntimeError("未获取到有效车辆定位。")
            self.start_xy = np.array([pose["x"], pose["y"]], dtype=float)
        else:
            self.start_xy = parse_point_args(
                "start", self.args.start_xy, self.args.start_latlon, self.transformer_to_proj
            )

        self.gvt.global_planning(self.start_xy, self.goal_xy)
        self.route_initialized = True
        print(f"✅ 路由初始化完成：start={self.start_xy.tolist()} goal={self.goal_xy.tolist()}")

    def update_vehicle_state(self):
        pose = self.current_pose_to_xy_heading()
        if pose is None:
            return False
        self.gvt.x = pose["x"]
        self.gvt.y = pose["y"]
        self.gvt.heading = pose["heading"]
        self.gvt.v = pose["speed"]
        self.gvt.a = pose["acc"]
        return True

    def build_obstacle_list(self):
        with self.obu_lock:
            obstacle_snapshot = list(self.latest_obstacles.values())

        if not obstacle_snapshot:
            return []

        now_ms = int(time.time() * 1000)
        fresh_obstacles = []
        retained = {}
        ego_x = self.gvt.x
        ego_y = self.gvt.y

        for obstacle in obstacle_snapshot:
            if obstacle.timestamp_ms > 0 and now_ms - obstacle.timestamp_ms > self.args.obu_track_timeout_ms:
                continue
            if ego_x is not None and ego_y is not None:
                if np.hypot(obstacle.x - ego_x, obstacle.y - ego_y) < OBU_SELF_FILTER_DIST_M:
                    continue
            fresh_obstacles.append(obstacle)
            retained[obstacle.ptc_id] = obstacle

        if len(retained) != len(obstacle_snapshot):
            with self.obu_lock:
                self.latest_obstacles = retained

        return fresh_obstacles

    def infer_turnlight(self, trajectory):
        traj = np.asarray(trajectory, dtype=float)
        future_idx = min(10, len(traj) - 1)
        current_pt = traj[0]
        future_pt = traj[future_idx]
        dx = future_pt[0] - current_pt[0]
        dy = future_pt[1] - current_pt[1]
        future_heading = np.arctan2(dy, dx)
        heading_delta = (future_heading - current_pt[2] + np.pi) % (2 * np.pi) - np.pi
        if heading_delta > np.radians(5.0):
            return 2
        if heading_delta < -np.radians(5.0):
            return 1
        return 0

    def trajectory_to_control_cmd(self, trajectory):
        traj = np.asarray(trajectory, dtype=float)
        if traj.ndim != 2 or len(traj) < 2:
            return self.build_hold_command()

        current = traj[0]
        plan_hz = max(float(self.args.plan_hz), 1e-3)
        target_index = min(CONTROL_PREVIEW_INDEX, len(traj) - 1)
        next_point = traj[target_index]
        segment_dt = target_index / plan_hz
        dx = float(next_point[0] - current[0])
        dy = float(next_point[1] - current[1])
        ds = float(np.hypot(dx, dy))
        heading_delta = float((next_point[2] - current[2] + np.pi) % (2 * np.pi) - np.pi)
        speed_mps = float(max(next_point[3], 0.0))
        kappa_cmd = heading_delta / max(ds, 1e-3)

        speed_kph = float(np.clip(speed_mps * 3.6, 0.0, self.args.max_speed_kph))
        wheel_angle_rad = float(np.arctan(self.gvt.wheelbase * kappa_cmd))
        wheel_angle_deg = float(np.clip(np.degrees(wheel_angle_rad), -26.0, 26.0))
        brkreq = 1 if speed_kph < 0.1 else 0

        return {
            "timestamp": int(time.time() * 1000),
            "speed_kph": speed_kph,
            "front_wheel_angle_deg": wheel_angle_deg,
            "gear": 4,
            "turnlight": self.infer_turnlight(traj),
            "brkreq": brkreq,
        }

    def publish_planned_trajectory(self, trajectory, command=None):
        """发布完整规划轨迹，供本地平台推演的轨迹一致车辆模型使用。"""

        if self.trajectory_pub is None:
            return
        traj = np.asarray(trajectory, dtype=float)
        if traj.ndim != 2 or traj.shape[0] == 0 or traj.shape[1] < 6:
            return

        payload = {
            "timestamp": int(time.time() * 1000),
            "frame": "map",
            "point_fields": ["x", "y", "heading", "v", "a", "kappa"],
            "points": traj[:, :6].tolist(),
        }
        if command is not None:
            payload["control_command"] = {
                "timestamp": int(command["timestamp"]),
                "speed_kph": float(command["speed_kph"]),
                "front_wheel_angle_deg": float(command["front_wheel_angle_deg"]),
                "gear": int(command["gear"]),
                "turnlight": int(command["turnlight"]),
                "brkreq": int(command["brkreq"]),
            }
        msg = self.TrajectoryMsg()
        msg.data = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        self.trajectory_pub.publish(msg)

    def is_arrived(self):
        if self.gvt.x is None or self.gvt.y is None:
            return False
        dist = float(np.linalg.norm(np.array([self.gvt.x, self.gvt.y]) - self.goal_xy))
        return dist <= self.args.arrive_dist and self.gvt.v <= self.args.arrive_speed

    def run_planning_loop(self):
        interval = 1.0 / max(self.args.plan_hz, 1e-3)
        print(f"▶️  规划线程启动 (Hz={self.args.plan_hz})")
        self.initialize_route_if_needed()

        while self.running and not self.rospy.is_shutdown():
            start_time = time.time()
            if not self.pose_ready_event.wait(timeout=0.2):
                time.sleep(interval)
                continue

            if not self.update_vehicle_state():
                time.sleep(interval)
                continue

            if self.is_arrived():
                if not self.goal_reached:
                    print("✅ 已到达终点，切换为安全停车命令。")
                self.goal_reached = True
                with self.command_lock:
                    self.latest_command = self.build_hold_command()
                time.sleep(interval)
                continue

            ready, reason = self.evaluate_actuator_ready()
            if not ready:
                with self.command_lock:
                    self.latest_command = self.build_hold_command()
                print(f"⚠ 底盘当前不允许接管，发送保守刹车命令：{reason}")
                time.sleep(interval)
                continue

            obstacles = self.build_obstacle_list()
            trajectory = self.gvt.traj_planning(obstacles)
            if trajectory is None:
                if self.gvt.traj_planner.attribute.get("plan_mode") == "arrived":
                    self.goal_reached = True
                    with self.command_lock:
                        self.latest_command = self.build_hold_command()
                    print("✅ 状态机判定已到达终点，切换为安全停车命令。")
                    time.sleep(interval)
                    continue
                if start_time - self._last_plan_failure_log_ts >= 1.0:
                    attr = self.gvt.traj_planner.attribute
                    obstacle_ids = ",".join(str(getattr(obstacle, "ptc_id", "")) for obstacle in obstacles[:8])
                    if len(obstacles) > 8:
                        obstacle_ids = f"{obstacle_ids},..."
                    print(
                        "⚠ 规划未输出轨迹："
                        f"obstacle_count={len(obstacles)} "
                        f"obstacle_ids=[{obstacle_ids or '-'}] "
                        f"plan_mode={attr.get('plan_mode', '')} "
                        f"candidate_count={attr.get('candidate_count', '')} "
                        f"feasible_candidate_count={attr.get('feasible_candidate_count', '')} "
                        f"failure_reason={attr.get('failure_reason', '')} "
                        f"decision_reason={attr.get('reason', '')}",
                        flush=True,
                    )
                    self._last_plan_failure_log_ts = start_time
                if self._last_valid_plan_ts > 0.0:
                    time_since_last_valid = time.time() - self._last_valid_plan_ts
                    if time_since_last_valid < PLANNING_FAILURE_GRACE_S:
                        time.sleep(interval)
                        continue
                with self.command_lock:
                    self.latest_command = self.build_hold_command()
                time.sleep(interval)
                continue

            command = self.trajectory_to_control_cmd(trajectory)
            self.publish_planned_trajectory(trajectory, command)
            with self.command_lock:
                self.latest_command = command
            self._last_valid_plan_ts = time.time()

            elapsed = time.time() - start_time
            time.sleep(max(0.0, interval - elapsed))

    def run_control_loop(self):
        interval = 1.0 / max(self.args.control_hz, 1e-3)
        print(f"▶️  控制线程启动 (Hz={self.args.control_hz})")
        while self.running and not self.rospy.is_shutdown():
            with self.command_lock:
                command = dict(self.latest_command)

            command["timestamp"] = int(time.time() * 1000)
            msg = self.ControlCmdMsg()
            self._apply_control_command(msg, command)
            try:
                self.control_pub.publish(msg)
            except Exception as exc:
                if self.rospy.is_shutdown() or "closed topic" in str(exc):
                    break
                print(f"⚠ 控制命令发布失败：{exc}", flush=True)
            time.sleep(interval)

    def stop(self):
        self.running = False
        hold = self.build_hold_command()
        msg = self.ControlCmdMsg()
        self._apply_control_command(msg, hold)
        try:
            self.control_pub.publish(msg)
        except Exception as exc:
            if not (self.rospy.is_shutdown() or "closed topic" in str(exc)):
                print(f"⚠ 停车命令发布失败：{exc}", flush=True)
        print("⏹️  节点退出，已下发一次安全停车命令。")

    def _apply_control_command(self, msg, command):
        _set_first_attr(msg, ["timestamp"], int(command["timestamp"]))
        # 协议以 acceleration 字段承载速度指令；若消息类保留 speed_kph 字段名，也一并兼容。
        _set_first_attr(msg, ["acceleration", "speed_kph"], float(command["speed_kph"]))
        _set_first_attr(msg, ["wheel_angle"], float(command["front_wheel_angle_deg"]))
        _set_first_attr(msg, ["gear"], int(command["gear"]))
        _set_first_attr(msg, ["turnlight", "turnLight"], int(command["turnlight"]))
        _set_first_attr(msg, ["brkreq"], int(command["brkreq"]))

    def run(self):
        print(f"🚀 直控节点启动，地图缓存：{self.cache_path}")
        print(f"🚀 目标点：{self.goal_xy.tolist()}")
        self.plan_thread.start()
        self.control_thread.start()

        try:
            while not self.rospy.is_shutdown():
                time.sleep(0.2)
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()


def main():
    args = build_arg_parser().parse_args()
    if args.goal_xy is None and args.goal_latlon is None:
        raise SystemExit("必须提供终点：--goal-xy 或 --goal-latlon")
    if not args.use_current_start and args.start_xy is None and args.start_latlon is None:
        raise SystemExit("必须提供起点：--start-xy / --start-latlon，或使用 --use-current-start")

    node = DirectControlNode(args)
    node.run()


if __name__ == "__main__":
    main()
