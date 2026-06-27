"""本地平台推演版 Redis-ROS Adapter。

这个入口把平台 Redis 协议、本地 ROS 算法和虚拟车辆模型串成闭环：
平台下发 A-B 任务后，Adapter 启动 `direct_control_node.py`，持续发布
`/sensorgps`、`/actuator` 和背景车 `/obu/source`，订阅算法输出的
`/controlcmd`，再把虚拟主车轨迹回传给平台。它用于本地推演联调，
不是最终实车域控程序。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

from redis_ros_adapter.core.config import flatten_adapter_config, load_adapter_config
from redis_ros_adapter.core.coordinates import DEFAULT_MAP_PATH, MapProjector, load_or_build_map
from redis_ros_adapter.core.direct_control_launcher import DirectControlLauncher
from redis_ros_adapter.core.platform_protocol import (
    build_merge_report,
    build_status_message,
    build_trajectory_report,
    dumps_json_message,
    extract_vehicle_values,
    loads_json_message,
    now_ms,
    parse_prepare_message,
    parse_task_control_message,
)
from redis_ros_adapter.core.types import ControlCommandState, PreparedTask, ProtocolChannels
from redis_ros_adapter.core.virtual_vehicle import VirtualVehicle, VirtualVehicleConfig


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _env_bool(name: str, default: bool = False) -> bool:
    """从环境变量读取布尔开关。"""

    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _default(defaults: dict[str, Any], key: str, fallback: Any) -> Any:
    """优先使用配置文件默认值，不存在时使用命令行/环境变量兜底值。"""

    return defaults.get(key, fallback)


def build_arg_parser(defaults: dict[str, Any] | None = None) -> argparse.ArgumentParser:
    """创建 Adapter 命令行参数解析器。

    参数来源优先级是：命令行显式传入 > 配置文件 > 环境变量 > 代码兜底值。
    真实平台运行时通常由 `platform_integration/config/platform_real.yaml` 固化这些参数。
    """

    defaults = defaults or {}
    parser = argparse.ArgumentParser(description="本地平台推演用 Redis-ROS Adapter")
    parser.add_argument("--config", default=str(_default(defaults, "config", "")))
    parser.add_argument(
        "--mode",
        choices=["online_only", "virtual_simulation"],
        default=str(_default(defaults, "mode", os.getenv("ADAPTER_MODE", "virtual_simulation"))),
    )
    parser.add_argument("--online-only", action="store_const", const="online_only", dest="mode")
    parser.add_argument("--redis-host", default=_default(defaults, "redis_host", os.getenv("REDIS_HOST", "host.docker.internal")))
    parser.add_argument("--redis-port", type=int, default=int(_default(defaults, "redis_port", os.getenv("REDIS_PORT", "6379"))))
    parser.add_argument("--redis-db", type=int, default=int(_default(defaults, "redis_db", os.getenv("REDIS_DB", "0"))))
    parser.add_argument("--redis-password", default=_default(defaults, "redis_password", os.getenv("REDIS_PASSWORD", "")))
    parser.add_argument("--redis-timeout-s", type=float, default=float(_default(defaults, "redis_timeout_s", os.getenv("REDIS_TIMEOUT_S", "5"))))
    parser.add_argument("--device-id", type=int, default=int(_default(defaults, "device_id", os.getenv("PLATFORM_DEVICE_ID", "4"))))
    parser.add_argument("--device-name", default=_default(defaults, "device_name", os.getenv("PLATFORM_DEVICE_NAME", "")))
    parser.add_argument("--vehicle-id", default=_default(defaults, "vehicle_id", os.getenv("PLATFORM_VEHICLE_ID", "")))
    parser.add_argument("--vehicle-name", default=_default(defaults, "vehicle_name", os.getenv("PLATFORM_VEHICLE_NAME", "")))
    parser.add_argument(
        "--vehicle-id-from-device-id",
        action="store_true",
        default=bool(_default(defaults, "vehicle_id_from_device_id", _env_bool("VEHICLE_ID_FROM_DEVICE_ID", True))),
    )
    parser.add_argument("--vehicle-length-cm", type=int, default=int(_default(defaults, "vehicle_length_cm", 464)))
    parser.add_argument("--vehicle-width-cm", type=int, default=int(_default(defaults, "vehicle_width_cm", 178)))
    parser.add_argument("--vehicle-height-cm", type=int, default=int(_default(defaults, "vehicle_height_cm", 150)))
    parser.add_argument("--control-channel", default=_default(defaults, "control_channel", os.getenv("PLATFORM_CONTROL_CHANNEL", "control1734685724338channel")))
    parser.add_argument("--status-channel", default=_default(defaults, "status_channel", os.getenv("PLATFORM_STATUS_CHANNEL", "STATUSResult")))
    parser.add_argument("--data-channel-hint", default=_default(defaults, "data_channel_hint", os.getenv("PLATFORM_DATA_CHANNEL_HINT", "")))
    parser.add_argument("--map-path", default=_default(defaults, "map_path", str(DEFAULT_MAP_PATH)))
    parser.add_argument("--cache-path", default=_default(defaults, "cache_path", os.getenv("ADAPTER_CACHE_PATH", "/tmp/virtual_adapter_xodr_cache.pkl")))
    parser.add_argument("--direct-cache-path", default=_default(defaults, "direct_cache_path", os.getenv("DIRECT_CONTROL_CACHE_PATH", "/tmp/direct_control_adapter_cache.pkl")))
    parser.add_argument("--rebuild-cache", action="store_true", default=bool(_default(defaults, "rebuild_cache", _env_bool("REBUILD_CACHE", False))))
    parser.add_argument("--planner-mode", choices=["local_sampling", "scene_target"], default=_default(defaults, "planner_mode", os.getenv("PLANNER_MODE", "local_sampling")))
    parser.add_argument("--scene-algorithm-symbol", default=_default(defaults, "scene_algorithm_symbol", os.getenv("SCENE_ALGORITHM_SYMBOL", "")))
    parser.add_argument("--plan-hz", type=float, default=float(_default(defaults, "plan_hz", os.getenv("ROS_PLAN_HZ", "10"))))
    parser.add_argument("--control-hz", type=float, default=float(_default(defaults, "control_hz", os.getenv("ROS_CONTROL_HZ", "20"))))
    parser.add_argument("--publish-hz", type=float, default=float(_default(defaults, "publish_hz", os.getenv("ADAPTER_PUBLISH_HZ", "20"))))
    parser.add_argument("--report-hz", type=float, default=float(_default(defaults, "report_hz", os.getenv("ADAPTER_REPORT_HZ", "10"))))
    parser.add_argument("--heartbeat-hz", type=float, default=float(_default(defaults, "heartbeat_hz", os.getenv("ADAPTER_HEARTBEAT_HZ", "1"))))
    parser.add_argument("--duration-s", type=float, default=float(_default(defaults, "duration_s", os.getenv("ADAPTER_DURATION_S", "0"))))
    parser.add_argument("--trajectory-advance-index", type=int, default=int(_default(defaults, "trajectory_advance_index", os.getenv("ADAPTER_TRAJECTORY_ADVANCE_INDEX", "1"))))
    parser.add_argument("--max-speed-kph", type=float, default=float(_default(defaults, "max_speed_kph", os.getenv("ADAPTER_MAX_SPEED_KPH", "25"))))
    parser.add_argument("--log-dir", default=_default(defaults, "log_dir", os.getenv("ADAPTER_LOG_DIR", "/tmp/virtual_redis_ros_adapter")))
    parser.add_argument("--node-name", default=_default(defaults, "node_name", "virtual_redis_ros_adapter"))
    parser.add_argument("--sensorgps-topic", default=_default(defaults, "sensorgps_topic", os.getenv("ROS_SENSORGPS_TOPIC", "/sensorgps")))
    parser.add_argument("--actuator-topic", default=_default(defaults, "actuator_topic", os.getenv("ROS_ACTUATOR_TOPIC", "/actuator")))
    parser.add_argument("--controlcmd-topic", default=_default(defaults, "controlcmd_topic", os.getenv("ROS_CONTROLCMD_TOPIC", "/controlcmd")))
    parser.add_argument("--trajectory-topic", default=_default(defaults, "trajectory_topic", os.getenv("ROS_TRAJECTORY_TOPIC", "/planning/trajectory")))
    parser.add_argument("--obu-topic", default=_default(defaults, "obu_topic", os.getenv("ROS_OBU_TOPIC", "/obu/source")))
    parser.add_argument("--scene-main-ptc-id", default=_default(defaults, "scene_main_ptc_id", os.getenv("ROS_SCENE_MAIN_PTC_ID", "")))
    parser.add_argument("--scene-exposed-ptc-id", default=_default(defaults, "scene_exposed_ptc_id", os.getenv("ROS_SCENE_EXPOSED_PTC_ID", "")))
    return parser


def parse_args() -> argparse.Namespace:
    """解析命令行参数，并把配置文件字段合并进去。"""

    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", default=os.getenv("PLATFORM_ADAPTER_CONFIG", ""))
    known, _ = pre_parser.parse_known_args()
    defaults: dict[str, Any] = {}
    if known.config:
        defaults = flatten_adapter_config(load_adapter_config(known.config))
        defaults["config"] = known.config
    args = build_arg_parser(defaults).parse_args()
    if args.vehicle_id_from_device_id and not str(args.vehicle_id or "").strip():
        args.vehicle_id = str(args.device_id)
    if not str(args.vehicle_name or "").strip():
        args.vehicle_name = str(args.device_name or args.vehicle_id)
    return args


def make_redis_client(args: argparse.Namespace) -> Any:
    """根据配置连接平台 Redis，并在失败时给出排查提示。

    Adapter 必须先 `PING` 成功，否则后续心跳、任务接收和轨迹回传都没有意义。
    """

    import redis

    client = redis.Redis(
        host=args.redis_host,
        port=args.redis_port,
        db=args.redis_db,
        password=args.redis_password or None,
        decode_responses=True,
        socket_connect_timeout=float(args.redis_timeout_s),
        socket_timeout=float(args.redis_timeout_s),
    )
    try:
        client.ping()
    except Exception as exc:
        password_state = "set" if str(args.redis_password or "") else "empty"
        raise SystemExit(
            "\n".join(
                [
                    "[redis] failed to connect to platform Redis.",
                    f"[redis] host={args.redis_host} port={args.redis_port} db={args.redis_db} password={password_state}",
                    f"[redis] error={type(exc).__name__}: {exc}",
                    "[redis] check these with the platform side:",
                    "  1. Redis password is empty or not.",
                    "  2. The service is plain Redis, not TLS Redis or a proxy protocol.",
                    "  3. The host/port is the Redis service for this test environment.",
                    "  4. Your Mac/Docker IP is allowed by the platform network or whitelist.",
                    "  5. VPN/LAN access to 192.168.1.66:6379 is active.",
                ]
            )
        ) from exc
    return client


class VirtualRedisRosAdapter:
    """本地推演 Adapter 主类。

    该类负责四件事：监听平台 Redis 控制通道、把平台任务转换成 ROS 输入、
    启动并监控算法进程、把算法驱动的虚拟车辆状态回传平台。
    """

    def __init__(self, args: argparse.Namespace) -> None:
        """初始化 Redis、ROS publisher/subscriber、地图投影和虚拟车辆。"""

        import rospy
        from common_msgs.msg import actuator, controlcmd, sensorgps
        from std_msgs.msg import String as RosString

        self.args = args
        self.rospy = rospy
        self.SensorGpsMsg = sensorgps
        self.ActuatorMsg = actuator
        self.ControlCmdMsg = controlcmd
        self.TrajectoryMsg = RosString
        try:
            from common_msgs.msg import obupant, obupants

            self.ObuPantMsg = obupant
            self.ObuPantsMsg = obupants
        except Exception:
            self.ObuPantMsg = None
            self.ObuPantsMsg = None
        self.redis = make_redis_client(args)
        self.pubsub = self.redis.pubsub(ignore_subscribe_messages=True)
        self.pubsub.subscribe(args.control_channel)
        self.subscribed_background_channels: set[str] = set()

        self.xodr_parser = load_or_build_map(args.map_path, args.cache_path, args.rebuild_cache)
        self.projector = MapProjector(self.xodr_parser)
        self.vehicle = VirtualVehicle(
            projector=self.projector,
            config=VirtualVehicleConfig(
                vehicle_id=str(args.vehicle_id),
                name=str(args.vehicle_name),
                length_cm=int(args.vehicle_length_cm),
                width_cm=int(args.vehicle_width_cm),
                height_cm=int(args.vehicle_height_cm),
                max_speed_kph=float(args.max_speed_kph),
                trajectory_advance_index=int(args.trajectory_advance_index),
            ),
        )
        self.vehicle_ready = False
        self.task: PreparedTask | None = None
        self.channels = ProtocolChannels()
        self.running_task = False
        self.prepare_status_active = False
        self.prepare_status_state = 2
        self.frame_id = 0
        self.last_step_time = time.monotonic()
        self.last_ros_publish_time = 0.0
        self.last_report_time = 0.0
        self.last_heartbeat_time = 0.0
        self.last_prepare_status_time = 0.0
        self.latest_background_payloads: dict[str, dict[str, Any]] = {}

        self.log_dir = Path(args.log_dir)
        self.control_payload_log_dir = self.log_dir / "redis_control_payloads"
        self.launcher = DirectControlLauncher(PROJECT_ROOT, self.log_dir)

        rospy.init_node(args.node_name)
        self.gps_pub = rospy.Publisher(args.sensorgps_topic, self.SensorGpsMsg, queue_size=10)
        self.actuator_pub = rospy.Publisher(args.actuator_topic, self.ActuatorMsg, queue_size=10)
        self.obu_pub = None
        if self.ObuPantsMsg is not None:
            self.obu_pub = rospy.Publisher(args.obu_topic, self.ObuPantsMsg, queue_size=10)
        else:
            rospy.logwarn("common_msgs/obupants is unavailable; platform background vehicles will not feed scene algorithms")
        rospy.Subscriber(args.controlcmd_topic, self.ControlCmdMsg, self._control_callback, queue_size=50)
        if args.trajectory_topic:
            rospy.Subscriber(args.trajectory_topic, self.TrajectoryMsg, self._trajectory_callback, queue_size=20)

    def _control_callback(self, msg: Any) -> None:
        """接收算法 `/controlcmd` 输出，并写入虚拟车辆控制目标。"""

        command = ControlCommandState(
            timestamp_ms=int(getattr(msg, "timestamp", 0) or 0),
            target_speed_kph=float(getattr(msg, "acceleration", 0.0) or 0.0),
            wheel_angle_deg=float(getattr(msg, "wheel_angle", 0.0) or 0.0),
            gear=int(getattr(msg, "gear", 4) or 4),
            turnlight=int(getattr(msg, "turnlight", 0) or 0),
            brkreq=int(getattr(msg, "brkreq", 0) or 0),
        )
        self.vehicle.apply_control_command(command)

    def _trajectory_callback(self, msg: Any) -> None:
        """接收算法 `/planning/trajectory` 输出，仅用于控制量闭环误差观察。"""

        raw = str(getattr(msg, "data", "") or "")
        if not raw:
            return
        try:
            payload = json.loads(raw)
        except Exception as exc:
            self.rospy.logwarn_throttle(2.0, "invalid trajectory json: %s", exc)
            return

        points = payload.get("points")
        if not isinstance(points, list):
            return
        advance_index = max(0, int(self.args.trajectory_advance_index))
        if points:
            try:
                start_point = points[0]
                target_point = points[min(advance_index, len(points) - 1)]
                control_command = payload.get("control_command") if isinstance(payload.get("control_command"), dict) else {}
                control_speed_kph = float(control_command.get("speed_kph", 0.0) or 0.0)
                control_brake = int(control_command.get("brkreq", 0) or 0)
                trajectory_dx = float(target_point[0]) - float(start_point[0])
                trajectory_dy = float(target_point[1]) - float(start_point[1])
                trajectory_speed = float(target_point[3])
                pos_err = math.hypot(self.vehicle.x - float(start_point[0]), self.vehicle.y - float(start_point[1]))
                heading_err = (
                    self.vehicle.heading - float(start_point[2]) + math.pi
                ) % (2.0 * math.pi) - math.pi
                speed_err = self.vehicle.v - float(start_point[3])
                self.rospy.loginfo_throttle(
                    1.0,
                    "received trajectory reference points=%d advance_index=%d dx=%.6f dy=%.6f target_speed_mps=%.6f control_speed_kph=%.3f control_brkreq=%d model_speed_mps=%.6f model_ref_err_m=%.6f heading_err_rad=%.6f speed_err_mps=%.6f",
                    len(points),
                    advance_index,
                    trajectory_dx,
                    trajectory_dy,
                    trajectory_speed,
                    control_speed_kph,
                    control_brake,
                    self.vehicle.v,
                    pos_err,
                    heading_err,
                    speed_err,
                )
            except (TypeError, ValueError, IndexError):
                self.rospy.logwarn_throttle(2.0, "failed to summarize trajectory payload")

    def _publish_status(self, status_type: int, state: int) -> None:
        """向平台状态通道发布设备在线或任务准备状态。"""

        payload = build_status_message(self.args.device_id, status_type, state)
        try:
            self.redis.publish(self.args.status_channel, dumps_json_message(payload))
        except Exception as exc:
            self.rospy.logwarn_throttle(2.0, "failed to publish status to Redis: %s", exc)

    def _dump_control_payload(self, prefix: str, payload: dict[str, Any], suffix: str = "") -> None:
        """把平台下发的控制 JSON 落盘，方便复盘每次真实任务输入。"""

        try:
            self.control_payload_log_dir.mkdir(parents=True, exist_ok=True)
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            suffix_part = f"_{suffix}" if suffix else ""
            path = self.control_payload_log_dir / f"{timestamp}_{prefix}{suffix_part}.json"
            path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as exc:
            self.rospy.logwarn_throttle(5.0, "failed to dump control payload: %s", exc)

    def _clear_prepared_task(self) -> None:
        """清空当前 3.1 准备窗口，防止旧轨迹继续触发 3.2 ready。"""

        self.task = None
        self.vehicle_ready = False
        self.prepare_status_active = False
        self.prepare_status_state = 2
        self.latest_background_payloads.clear()

    def _handle_prepare(self, payload: dict[str, Any]) -> None:
        """处理平台 type=1 准备任务消息。

        成功解析起终点后，先把虚拟车放到起点并向平台回报“任务已准备”；
        此时还不启动算法，真正启动发生在平台后续 type=2 start 消息。
        """

        task = parse_prepare_message(payload)
        if task is None:
            return
        self._dump_control_payload("prepare", payload, str(task.task_id or task.case_id or "unknown"))
        self.task = task
        self.prepare_status_active = True
        self.latest_background_payloads.clear()
        if task.ready and task.start is not None:
            self.vehicle.reset_from_task_point(task.start)
            self.vehicle_ready = True
            self.prepare_status_state = 1
            self._publish_status(status_type=1, state=1)
            self.last_prepare_status_time = time.monotonic()
            self.rospy.loginfo(
                "prepared task task_id=%s case_id=%s start=(%.8f, %.8f) goal=(%.8f, %.8f)",
                task.task_id,
                task.case_id,
                task.start.latitude,
                task.start.longitude,
                task.goal.latitude if task.goal else 0.0,
                task.goal.longitude if task.goal else 0.0,
            )
        else:
            self.vehicle_ready = False
            self.prepare_status_state = 2
            self._publish_status(status_type=1, state=2)
            self.last_prepare_status_time = time.monotonic()
            self.rospy.logwarn("prepare message missing start or goal")

    def _handle_task_control(self, payload: dict[str, Any]) -> None:
        """处理平台 type=2 任务开始/停止消息。"""

        control = parse_task_control_message(payload)
        if control is None:
            return
        self.channels = control.channels
        self._dump_control_payload("task_control", payload, str(self.channels.task_id or "unknown"))
        for channel in self.channels.background_channels:
            if channel and channel not in self.subscribed_background_channels:
                self.pubsub.subscribe(channel)
                self.subscribed_background_channels.add(channel)

        if control.task_type == 1:
            self._start_task()
        elif control.task_type == 0:
            self._stop_task()

    def _start_task(self) -> None:
        """按最近一次准备任务启动算法，并重置本地虚拟车辆闭环状态。"""

        if self.task is None or not self.task.ready:
            self.rospy.logwarn("cannot start task before valid prepare message")
            self._clear_prepared_task()
            return
        if self.channels.task_id and self.task.task_id and self.channels.task_id != self.task.task_id:
            self._publish_status(status_type=1, state=2)
            self.rospy.logwarn(
                "refuse to start task_id=%s with stale prepared task_id=%s; wait for new prepare",
                self.channels.task_id,
                self.task.task_id,
            )
            self._clear_prepared_task()
            return
        if self.task.task_id == "" and self.channels.task_id:
            self.task.task_id = self.channels.task_id
        if self.channels.case_id:
            self.task.case_id = self.channels.case_id
        if self.channels.record_id:
            self.task.record_id = self.channels.record_id

        # 4.1 start 到达后，3.2 准备状态停止，后续进入 4.2 数据上报阶段。
        self.vehicle.reset_from_task_point(self.task.start)
        self.vehicle_ready = True
        self.prepare_status_active = False
        self.prepare_status_state = 2
        self.frame_id = 0
        self.last_step_time = time.monotonic()
        self.last_ros_publish_time = 0.0
        self.last_report_time = 0.0
        self._publish_ros_inputs(now_ms())
        self.rospy.loginfo(
            "reset virtual vehicle to task start before launch task_id=%s heading_source=%s",
            self.task.task_id,
            getattr(self.vehicle, "heading_source", ""),
        )

        self.launcher.start(
            self.task,
            map_path=self.args.map_path,
            cache_path=self.args.direct_cache_path,
            rebuild_cache=bool(self.args.rebuild_cache),
            planner_mode=self.args.planner_mode,
            scene_algorithm_symbol=self.args.scene_algorithm_symbol,
            obu_topic=self.args.obu_topic,
            scene_main_ptc_id=self.args.scene_main_ptc_id,
            scene_exposed_ptc_id=self.args.scene_exposed_ptc_id,
            plan_hz=self.args.plan_hz,
            control_hz=self.args.control_hz,
            trajectory_topic=self.args.trajectory_topic,
            max_speed_kph=self.args.max_speed_kph,
        )
        self.running_task = True
        self.rospy.loginfo("started direct_control_node.py for task_id=%s case_id=%s", self.task.task_id, self.task.case_id)

    def _stop_task(self) -> None:
        """停止当前算法任务，并保持 Adapter 继续在线等待下一次下发。"""

        self.launcher.stop()
        self.running_task = False
        self.channels = ProtocolChannels()
        self._clear_prepared_task()
        self.rospy.loginfo("stopped task")

    def _process_control_payload(self, payload: dict[str, Any]) -> None:
        """按平台消息 type 分发准备任务或任务控制消息。"""

        msg_type = int(payload.get("type", -1))
        if msg_type == 1:
            self._handle_prepare(payload)
        elif msg_type == 2:
            self._handle_task_control(payload)
        else:
            self.rospy.logdebug("ignored control payload type=%s", msg_type)

    def _process_redis_messages(self) -> None:
        """非阻塞读取 Redis pub/sub 消息。

        控制通道消息进入任务状态机；背景车通道消息会暂存并立即尝试转换成
        ROS OBU 消息发布给场景算法。
        """

        for _ in range(20):
            try:
                message = self.pubsub.get_message(timeout=0.0)
            except Exception as exc:
                self.rospy.logwarn_throttle(2.0, "failed to read Redis pubsub message: %s", exc)
                return
            if message is None:
                break
            if message.get("type") != "message":
                continue
            channel = str(message.get("channel", "") or "")
            try:
                payload = loads_json_message(message.get("data"))
            except Exception as exc:
                self.rospy.logwarn("invalid redis payload on %s: %s", channel, exc)
                continue
            if channel == self.args.control_channel:
                self._process_control_payload(payload)
            else:
                self.latest_background_payloads[channel] = payload
                self._publish_obu_from_background_payload(payload, channel)

    @staticmethod
    def _read_vehicle_id(vehicle: dict[str, Any]) -> str:
        """兼容不同平台字段名读取车辆 ID。"""

        for key in ("id", "ptc_id", "ptcid", "trackId", "track_id", "sourceId", "source_id"):
            value = vehicle.get(key)
            if value not in (None, ""):
                return str(value)
        return ""

    @staticmethod
    def _read_float_from_vehicle(vehicle: dict[str, Any], keys: tuple[str, ...], default: float = 0.0) -> float:
        """按候选字段名读取车辆浮点字段。"""

        for key in keys:
            value = vehicle.get(key)
            if value not in (None, ""):
                try:
                    return float(value)
                except (TypeError, ValueError):
                    return float(default)
        return float(default)

    @staticmethod
    def _platform_speed_to_mps(speed_value: float) -> float:
        """把平台速度 km/h 转为 ROS OBU 使用的 m/s。"""

        return max(0.0, float(speed_value)) / 3.6

    @staticmethod
    def _platform_size_to_m(value: float) -> float:
        """把平台车辆尺寸转换为米。

        平台有时以厘米回传车辆长宽高，数值大于 20 时按厘米处理；已经是米时
        原样保留。
        """

        value = max(0.0, float(value))
        if value > 20.0:
            return value / 100.0
        return value

    def _build_obu_pant_msg(self, vehicle: dict[str, Any], source_channel: str) -> Any | None:
        """把单个平台注册车辆状态转换成 `common_msgs/obupant`。

        与当前主车 ID 相同的记录会被过滤，避免算法把自己当作障碍物。
        """

        if self.ObuPantMsg is None:
            return None

        vehicle_id = self._read_vehicle_id(vehicle)
        if not vehicle_id or vehicle_id == str(self.args.vehicle_id) or vehicle_id == str(self.args.device_id):
            return None

        lon = self._read_float_from_vehicle(vehicle, ("longitude", "lon", "lng", "pos_lon"))
        lat = self._read_float_from_vehicle(vehicle, ("latitude", "lat", "pos_lat"))
        if abs(lon) <= 1e-9 or abs(lat) <= 1e-9:
            return None

        msg = self.ObuPantMsg()
        msg.ptc_id = str(vehicle_id)
        msg.ptc_type = int(self._read_float_from_vehicle(vehicle, ("ptc_type", "vehicleType"), 1.0))
        msg.source = 1
        msg.source_id = str(source_channel)
        msg.sec_mark = 0
        msg.pos_lon = float(lon)
        msg.pos_lat = float(lat)
        msg.lon = float(lon)
        msg.lat = float(lat)
        msg.speed = self._platform_speed_to_mps(self._read_float_from_vehicle(vehicle, ("speed",), 0.0))
        msg.heading = self._read_float_from_vehicle(vehicle, ("courseAngle", "heading"), 0.0)
        msg.accel = self._read_float_from_vehicle(vehicle, ("acc", "accel", "acceleration", "lonAcc"), 0.0)
        msg.width = self._platform_size_to_m(self._read_float_from_vehicle(vehicle, ("width",), 0.0))
        msg.length = self._platform_size_to_m(self._read_float_from_vehicle(vehicle, ("length",), 0.0))
        msg.height = self._platform_size_to_m(self._read_float_from_vehicle(vehicle, ("height",), 0.0))
        msg.isvalid = 1
        return msg

    def _publish_obu_from_background_payload(self, payload: dict[str, Any], source_channel: str) -> None:
        """把平台背景车消息转换成 `/obu/source` 并发布给 ROS 算法。"""

        if self.obu_pub is None or self.ObuPantsMsg is None:
            return

        vehicles = extract_vehicle_values(payload)
        pants = []
        for vehicle in vehicles:
            pant = self._build_obu_pant_msg(vehicle, source_channel)
            if pant is not None:
                pants.append(pant)
        if not pants:
            return

        msg = self.ObuPantsMsg()
        msg.timestamp = now_ms()
        msg.isvalid = 1
        msg.msg_cnt = len(pants)
        msg.pants = pants
        self.obu_pub.publish(msg)
        ids = ",".join(str(getattr(pant, "ptc_id", "")) for pant in pants[:8])
        self.rospy.loginfo_throttle(
            1.0,
            "published %d platform background vehicles to %s ids=[%s]",
            len(pants),
            self.args.obu_topic,
            ids,
        )

    def _publish_ros_inputs(self, timestamp_ms: int) -> None:
        """发布算法所需的主车 ROS 输入。

        `/sensorgps` 提供定位和速度，`/actuator` 提供健康的底盘状态；这两类
        输入替代了原离线仿真器或真实车端传感器反馈。
        """

        lat, lon = self.vehicle.gps_latlon()
        gps = self.SensorGpsMsg()
        gps.lon = float(lon)
        gps.lat = float(lat)
        gps.alt = 0.0
        gps.heading = float((90.0 - math.degrees(self.vehicle.heading)) % 360.0)
        gps.velocity = float(self.vehicle.v)
        gps.accx = float(self.vehicle.a)
        gps.status = 4
        gps.rawstatus = 50
        gps.satenum = 20
        gps.gpstime = int(timestamp_ms)
        gps.isvalid = 1
        gps.timestamp = int(timestamp_ms)
        gps.speedE = float(self.vehicle.v * math.cos(self.vehicle.heading))
        gps.speedN = float(self.vehicle.v * math.sin(self.vehicle.heading))
        self.gps_pub.publish(gps)

        actuator = self.ActuatorMsg()
        actuator.speed = float(self.vehicle.v * 3.6)
        actuator.isvalid = 1
        actuator.timestamp = int(timestamp_ms)
        actuator.gear = 4
        actuator.epb = 1
        actuator.wireControlStatus = 1
        actuator.accx = float(self.vehicle.a)
        actuator.brkMode = 1 if int(self.vehicle.command.brkreq) else 0
        actuator.ehbWorkMode = 5
        actuator.epdWorkMode = 5
        actuator.epsWorkMode = 5
        actuator.vehWorkMode = 5
        actuator.error = 0
        actuator.errorlevel = 0
        actuator.epsangle = float(self.vehicle.command.wheel_angle_deg)
        actuator.SteerSpeed = float(self.vehicle.v * 3.6)
        actuator.velAngle = 1 if self.vehicle.v > 0.05 else 0
        self.actuator_pub.publish(actuator)

    def _publish_reports(self, timestamp_ms: int) -> None:
        """把虚拟主车状态回传到平台数据通道和 mergeDataChannel。"""

        if not self.running_task:
            return
        state = self.vehicle.to_report_state(timestamp_ms)
        upload_channel = self.channels.upload_channel or str(self.args.data_channel_hint or "")
        upload_receivers = None
        merge_receivers = None
        if upload_channel:
            try:
                upload_receivers = self.redis.publish(
                    upload_channel,
                    dumps_json_message(build_trajectory_report(state, self.frame_id, timestamp_ms)),
                )
            except Exception as exc:
                self.rospy.logwarn_throttle(2.0, "failed to publish trajectory to Redis: %s", exc)
        if self.channels.merge_data_channel:
            try:
                merge_receivers = self.redis.publish(
                    self.channels.merge_data_channel,
                    dumps_json_message(
                        build_merge_report(
                            state,
                            self.frame_id,
                            case_id=self.channels.case_id or (self.task.case_id if self.task else ""),
                            task_id=self.channels.task_id or (self.task.task_id if self.task else ""),
                            source_channel=upload_channel,
                            timestamp_ms=timestamp_ms,
                            background_payloads=dict(self.latest_background_payloads),
                            exclude_vehicle_ids={str(self.args.vehicle_id), str(self.args.device_id)},
                        )
                    ),
                )
            except Exception as exc:
                self.rospy.logwarn_throttle(2.0, "failed to publish merge data to Redis: %s", exc)
        self.rospy.loginfo_throttle(
            1.0,
            "published platform 4.2 report frame=%d upload=%s receivers=%s merge=%s merge_receivers=%s lat=%.8f lon=%.8f speed_kph=%.2f",
            self.frame_id,
            upload_channel or "-",
            "-" if upload_receivers is None else str(upload_receivers),
            self.channels.merge_data_channel or "-",
            "-" if merge_receivers is None else str(merge_receivers),
            state.latitude,
            state.longitude,
            state.speed_kph,
        )

    def _check_direct_control_process(self) -> None:
        """检查算法进程是否异常退出，并向平台报告任务异常状态。"""

        if self.running_task and not self.launcher.running:
            code = self.launcher.poll()
            self.running_task = False
            self._clear_prepared_task()
            self._publish_status(status_type=1, state=2)
            self.rospy.logwarn("direct_control_node.py exited unexpectedly code=%s", code)

    def run(self) -> None:
        """运行 Adapter 主循环。

        主循环周期性处理 Redis 消息、推进虚拟车、发布 ROS 输入、回传平台轨迹、
        发送心跳，并监控算法进程状态。
        """

        rate = self.rospy.Rate(max(float(self.args.publish_hz), 1e-3))
        try:
            while not self.rospy.is_shutdown():
                self._process_redis_messages()
                monotonic_now = time.monotonic()
                dt = monotonic_now - self.last_step_time
                self.last_step_time = monotonic_now
                if self.vehicle_ready:
                    self.vehicle.step(dt)
                    timestamp_ms = now_ms()
                    if monotonic_now - self.last_ros_publish_time >= 1.0 / max(float(self.args.publish_hz), 1e-3):
                        self._publish_ros_inputs(timestamp_ms)
                        self.last_ros_publish_time = monotonic_now
                    if monotonic_now - self.last_report_time >= 1.0 / max(float(self.args.report_hz), 1e-3):
                        self._publish_reports(timestamp_ms)
                        self.last_report_time = monotonic_now
                        self.frame_id += 1

                if monotonic_now - self.last_heartbeat_time >= 1.0 / max(float(self.args.heartbeat_hz), 1e-3):
                    self._publish_status(status_type=0, state=1)
                    self.last_heartbeat_time = monotonic_now

                if self.prepare_status_active and not self.running_task and monotonic_now - self.last_prepare_status_time >= 1.0:
                    self._publish_status(status_type=1, state=self.prepare_status_state)
                    self.last_prepare_status_time = monotonic_now

                self._check_direct_control_process()
                rate.sleep()
        finally:
            self.launcher.stop()
            self.pubsub.close()


def run_online_heartbeat(args: argparse.Namespace) -> None:
    """只发送平台在线心跳，不启动 ROS 和算法。

    这个模式用于验证设备配置、Redis 地址、密码、网络白名单和状态通道是否
    正确，适合在正式跑算法前先确认平台能看到设备在线。
    """

    client = make_redis_client(args)

    interval_s = 1.0 / max(float(args.heartbeat_hz), 1e-3)
    deadline = None if float(args.duration_s) <= 0.0 else time.monotonic() + float(args.duration_s)
    print(
        "[online] redis=%s:%s db=%s status_channel=%s device_id=%s heartbeat_hz=%.3f"
        % (args.redis_host, args.redis_port, args.redis_db, args.status_channel, args.device_id, float(args.heartbeat_hz)),
        flush=True,
    )
    try:
        while True:
            timestamp_ms = now_ms()
            payload = build_status_message(args.device_id, status_type=0, state=1, timestamp_ms=timestamp_ms)
            client.publish(args.status_channel, dumps_json_message(payload))
            print("[online] published %s" % dumps_json_message(payload), flush=True)
            if deadline is not None and time.monotonic() >= deadline:
                break
            time.sleep(interval_s)
    except KeyboardInterrupt:
        print("[online] stopped by user", flush=True)


def main() -> None:
    """命令行入口：根据模式启动在线心跳或完整虚拟推演 Adapter。"""

    args = parse_args()
    if args.mode == "online_only":
        run_online_heartbeat(args)
        return
    adapter = VirtualRedisRosAdapter(args)
    adapter.run()


if __name__ == "__main__":
    main()
