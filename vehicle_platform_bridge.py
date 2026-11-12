"""实车平台任务接入桥（新增文件，不修改 direct_control_node.py）。

同一个文件提供五个可独立启动的进程，用 `--role` 选择：

    task-manager  订阅 /start_task、/taskstatus、/test_trajectory_req、
                  /hd_route_points，输出 /student/task_state 和 /student/route
    adapter       订阅 /objectTrack/track_results8CornerForRviz，输出
                  /student/obstacles（JSON）
    algorithm     direct_control_node.DirectControlNode 的子类，启动即 STANDBY，
                  收到 ACTIVE 后才规划控车，输出 /student/controlcmd
    mux           唯一发布 /controlcmd 的节点，默认 20 Hz HOLD；只有任务 ACTIVE、
                  算法命令新鲜时才转发

另有一个 `probe` 角色，用 `rostopic` 同源的消息类把车端真实字段打印出来，
用于第一次上车时校准 /start_task、/taskstatus、/test_trajectory_req 的字段。

设计约束：

* `direct_control_node.py` 原样保留，直接控车测试方式不受影响；
* 内部话题一律 `std_msgs/String` + JSON，避免 device_msgs / common_msgs
  两套工作空间 MD5 不一致导致节点收不到消息；
* 对外只使用车端 `~/redis_stu` 里的 common_msgs（controlcmd / sensorgps /
  actuator / obupants / sensorobjects）；
* HOLD 是 acceleration=0、wheel_angle=0、gear=4、turnlight=0、brkreq=0；
  brkreq=1 只留给 EMERGENCY。
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np


# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

DEFAULT_STATE_TOPIC = "/student/task_state"
DEFAULT_ROUTE_TOPIC = "/student/route"
DEFAULT_STUDENT_CMD_TOPIC = "/student/controlcmd"
DEFAULT_STUDENT_OBSTACLES_TOPIC = "/student/obstacles"
DEFAULT_CONTROLCMD_TOPIC = "/controlcmd"
DEFAULT_START_TASK_TOPIC = "/start_task"
DEFAULT_TASK_STATUS_TOPIC = "/taskstatus"
DEFAULT_TEST_TRAJECTORY_TOPIC = "/test_trajectory_req"
DEFAULT_HD_ROUTE_TOPIC = "/hd_route_points"
DEFAULT_OBJECT_TRACK_TOPIC = "/objectTrack/track_results8CornerForRviz"
DEFAULT_SENSORGPS_TOPIC = "/sensorgps"
DEFAULT_ACTUATOR_TOPIC = "/actuator"
DEFAULT_OBU_TOPIC = "/obu/source"

HOLD_GEAR = 4
DEFAULT_MAX_SPEED_KPH = 25.0
DEFAULT_MAX_WHEEL_DEG = 26.0

STATE_STANDBY = "STANDBY"
STATE_PREPARE = "PREPARE"
STATE_ACTIVE = "ACTIVE"
STATE_EMERGENCY = "EMERGENCY"
STATE_NOOP = "NOOP"

_VALID_STATES = {STATE_STANDBY, STATE_PREPARE, STATE_ACTIVE, STATE_EMERGENCY}


# ---------------------------------------------------------------------------
# 基础工具
# ---------------------------------------------------------------------------

def now_ms() -> int:
    return int(time.time() * 1000)


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    return str(value).strip().lower() in {"1", "true", "yes", "on", "start", "active"}


def first_value(obj: Any, names: Iterable[str], default: Any = None) -> Any:
    """从 dict 或 ROS 消息对象里按候选字段名取第一个非 None 值。"""
    if isinstance(obj, dict):
        for name in names:
            if name in obj and obj[name] is not None:
                return obj[name]
        return default
    for name in names:
        if hasattr(obj, name):
            value = getattr(obj, name)
            if value is not None:
                return value
    return default


def resolve_msg_class(env_var: str, candidates: Iterable[str], display_name: str):
    """按环境变量 / 候选符号路径解析 ROS 消息类。"""
    symbols = []
    env_symbol = os.getenv(env_var, "").strip()
    if env_symbol:
        symbols.append(env_symbol)
    symbols.extend(candidates)

    last_error = None
    for symbol in symbols:
        try:
            module_name, class_name = symbol.rsplit(".", 1)
            module = importlib.import_module(module_name)
            return getattr(module, class_name)
        except Exception as exc:  # pragma: no cover - 运行期依赖 ROS 环境
            last_error = exc

    log(f"⚠ 无法解析 {display_name} 消息类，请设置环境变量 {env_var}；最后错误：{last_error}")
    return None


# ---------------------------------------------------------------------------
# ROS 消息 -> 普通 dict
# ---------------------------------------------------------------------------

_PRIMITIVES = (str, bytes, bool, int, float, type(None))


def ros_msg_to_dict(value: Any, depth: int = 0, seen: set[int] | None = None) -> Any:
    """把 ROS 消息递归转成普通 dict/list，便于不依赖具体字段名地解析。"""
    if depth > 8:
        return None
    if seen is None:
        seen = set()

    if isinstance(value, _PRIMITIVES):
        if isinstance(value, bytes):
            return value.decode("utf-8", "replace")
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return value

    if isinstance(value, (list, tuple)):
        return [ros_msg_to_dict(item, depth + 1, seen) for item in value]

    if isinstance(value, dict):
        return {
            str(key): ros_msg_to_dict(item, depth + 1, seen)
            for key, item in value.items()
        }

    object_id = id(value)
    if object_id in seen:
        return None
    seen.add(object_id)

    slots = getattr(value, "__slots__", None)
    if slots:
        result = {}
        for slot in slots:
            if str(slot).startswith("_"):
                continue
            try:
                result[str(slot)] = ros_msg_to_dict(getattr(value, slot), depth + 1, seen)
            except Exception:
                continue
        return result

    if hasattr(value, "__dict__"):
        return {
            str(key): ros_msg_to_dict(item, depth + 1, seen)
            for key, item in vars(value).items()
            if not str(key).startswith("_")
        }

    return str(value)


def parse_json_strings(value: Any, depth: int = 0) -> Any:
    """平台经常把任务 JSON 塞进 string 字段，这里递归展开。"""
    if depth > 8:
        return value
    if isinstance(value, str):
        text = value.strip()
        if (text.startswith("{") and text.endswith("}")) or (
            text.startswith("[") and text.endswith("]")
        ):
            try:
                return parse_json_strings(json.loads(text), depth + 1)
            except Exception:
                return value
        return value
    if isinstance(value, dict):
        return {key: parse_json_strings(item, depth + 1) for key, item in value.items()}
    if isinstance(value, list):
        return [parse_json_strings(item, depth + 1) for item in value]
    return value


def walk_values(value: Any, path: str = "") -> Iterable[tuple[str, Any]]:
    """深度优先遍历所有 value，产出 (path, value)。"""
    yield path, value
    if isinstance(value, dict):
        for key, item in value.items():
            child = f"{path}.{key}" if path else str(key)
            yield from walk_values(item, child)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from walk_values(item, f"{path}[{index}]")


def walk_mappings(value: Any, path: str = "") -> Iterable[tuple[str, dict]]:
    for child_path, item in walk_values(value, path):
        if isinstance(item, dict):
            yield child_path, item


# ---------------------------------------------------------------------------
# 路线点解析
# ---------------------------------------------------------------------------

_LAT_KEYS = ("lat", "latitude", "pos_lat", "posLat", "latitude_deg", "Latitude")
_LON_KEYS = ("lon", "lng", "longitude", "longtitude", "pos_lon", "posLon", "Longitude")
_X_KEYS = ("x", "X", "pos_x", "posX")
_Y_KEYS = ("y", "Y", "pos_y", "posY")
_SITE_TYPE_KEYS = ("siteType", "site_type", "sitetype", "pointType", "point_type")
_START_KEYS = ("start", "startPoint", "start_point", "startPos", "start_pos", "from", "origin")
_GOAL_KEYS = ("goal", "goalPoint", "goal_point", "goalPos", "goal_pos", "end", "endPoint", "end_point", "destination")


def point_from_mapping(value: Any) -> dict[str, float] | None:
    """把一个 dict 识别成路线点；经纬度优先，其次 x/y。"""
    if not isinstance(value, dict):
        return None

    lat = first_value(value, _LAT_KEYS)
    lon = first_value(value, _LON_KEYS)
    x = first_value(value, _X_KEYS)
    y = first_value(value, _Y_KEYS)
    site_type = first_value(value, _SITE_TYPE_KEYS)

    has_latlon = (
        lat is not None
        and lon is not None
        and abs(as_float(lat)) > 1e-9
        and abs(as_float(lon)) > 1e-9
    )
    has_xy = x is not None and y is not None
    if not has_latlon and not has_xy:
        return None

    point: dict[str, float] = {}
    if has_latlon:
        point["lat"] = as_float(lat)
        point["lon"] = as_float(lon)
    else:
        point["x"] = as_float(x)
        point["y"] = as_float(y)
    if site_type is not None:
        point["site_type"] = as_int(site_type, -1)
    return point


def _as_point_list(value: Any) -> list[dict[str, float]] | None:
    """列表里每一项都能识别成路线点时，返回该列表。"""
    if not isinstance(value, list) or len(value) < 2:
        return None

    points: list[dict[str, float]] = []
    for item in value:
        if not isinstance(item, dict):
            return None
        point = point_from_mapping(item)
        if point is None:
            inner = first_value(item, ("value", "point", "pos", "position", "pose"))
            if isinstance(inner, dict):
                point = point_from_mapping(inner)
        if point is None:
            return None
        points.append(point)
    return points


def iter_point_lists(value: Any, path: str = "") -> Iterable[tuple[str, list[dict[str, float]]]]:
    if isinstance(value, dict):
        for key, item in value.items():
            child = f"{path}.{key}" if path else str(key)
            if isinstance(item, list):
                points = _as_point_list(item)
                if points:
                    yield child, points
            yield from iter_point_lists(item, child)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from iter_point_lists(item, f"{path}[{index}]")


def _first_point_list(value: Any) -> list[dict[str, float]] | None:
    for _, points in iter_point_lists(value):
        return points
    return None


def _find_point_by_keys(value: Any, keys: Iterable[str]) -> dict[str, float] | None:
    if isinstance(value, dict):
        for key, item in value.items():
            if key in keys:
                if isinstance(item, dict):
                    point = point_from_mapping(item)
                    if point:
                        return point
                elif isinstance(item, list) and item:
                    for candidate in item:
                        if isinstance(candidate, dict):
                            point = point_from_mapping(candidate)
                            if point:
                                return point
        for item in value.values():
            point = _find_point_by_keys(item, keys)
            if point:
                return point
    elif isinstance(value, list):
        for item in value:
            point = _find_point_by_keys(item, keys)
            if point:
                return point
    return None


def _participant_route(payload: Any, ego_id: str) -> dict[str, Any] | None:
    if not ego_id:
        return None
    for _, value in walk_values(payload):
        if not isinstance(value, list):
            continue
        for item in value:
            if not isinstance(item, dict):
                continue
            participant_id = first_value(
                item, ("ptcId", "ptc_id", "id", "vehicleId", "objId", "objectId")
            )
            if participant_id is None or str(participant_id) != str(ego_id):
                continue
            points = _first_point_list(item)
            if points:
                return _make_route(
                    points[0], points[-1], f"participant:{participant_id}"
                )
    return None


def _make_route(start: dict[str, float], goal: dict[str, float], source: str) -> dict[str, Any]:
    return {
        "start": start,
        "goal": goal,
        "source": source,
        "frame": "latlon" if "lat" in start else "xy",
        "stamp_ms": now_ms(),
    }


def extract_route(
    payload: Any,
    *,
    ego_id: str = "",
    allow_flat: bool = True,
    require_site_type: bool = False,
) -> dict[str, Any] | None:
    """从任意任务/轨迹消息 dict 中提取起终点。

    优先级：
    1. 任意位置出现 siteType=0（起点）和 siteType=1（终点）；
    2. 顶层/嵌套的 start/goal 字段；
    3. 指定 ego_id 的参与者轨迹；
    4. 最长/最像路线的点列表的首尾点（require_site_type 时禁用）。
    """
    if not isinstance(payload, dict):
        return None

    site_points: dict[int, dict[str, float]] = {}
    for _, mapping in walk_mappings(payload):
        point = point_from_mapping(mapping)
        if point and "site_type" in point and point["site_type"] in (0, 1):
            site_points.setdefault(int(point["site_type"]), point)
    if 0 in site_points and 1 in site_points:
        return _make_route(site_points[0], site_points[1], "site_type")

    start_point = _find_point_by_keys(payload, _START_KEYS)
    goal_point = _find_point_by_keys(payload, _GOAL_KEYS)
    if start_point and goal_point:
        return _make_route(start_point, goal_point, "start_goal_keys")

    participant_route = _participant_route(payload, ego_id)
    if participant_route:
        return participant_route

    if require_site_type:
        return None

    best: tuple[int, str, list[dict[str, float]]] | None = None
    for path, points in iter_point_lists(payload):
        score = len(points)
        lowered = path.lower()
        if any(
            keyword in lowered
            for keyword in ("route", "points", "posarray", "trajectory", "path", "waypoint")
        ):
            score += 1000
        if best is None or score > best[0]:
            best = (score, path, points)
    if best:
        _, path, points = best
        return _make_route(points[0], points[-1], f"point_list:{path}")

    return None


def manual_route_from_args(args: argparse.Namespace) -> dict[str, Any] | None:
    start = None
    goal = None
    if args.manual_start_latlon:
        start = {"lat": float(args.manual_start_latlon[0]), "lon": float(args.manual_start_latlon[1])}
    elif args.manual_start_xy:
        start = {"x": float(args.manual_start_xy[0]), "y": float(args.manual_start_xy[1])}

    if args.manual_goal_latlon:
        goal = {"lat": float(args.manual_goal_latlon[0]), "lon": float(args.manual_goal_latlon[1])}
    elif args.manual_goal_xy:
        goal = {"x": float(args.manual_goal_xy[0]), "y": float(args.manual_goal_xy[1])}

    if start and goal:
        return _make_route(start, goal, "manual")
    return None


def _normalize_enum(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value)
    text = str(value).strip().lower()
    if text in {"start", "begin", "run", "running", "play", "active", "go", "resume"}:
        return 1
    if text in {"stop", "end", "finish", "pause", "idle", "halt", "abort"}:
        return 0
    try:
        return int(float(text))
    except (TypeError, ValueError):
        return None


def _int_set(text: str, default: Iterable[int]) -> set[int]:
    values: set[int] = set()
    for part in str(text or "").split(","):
        part = part.strip()
        if not part:
            continue
        normalized = _normalize_enum(part)
        if normalized is not None:
            values.add(normalized)
    return values or set(default)


# ---------------------------------------------------------------------------
# 任务状态机
# ---------------------------------------------------------------------------

@dataclass
class TaskMapping:
    prepare_status: set[int] = field(default_factory=lambda: {1})
    running_status: set[int] = field(default_factory=lambda: {2})
    stop_status: set[int] = field(default_factory=lambda: {0, 3})
    start_cmd: set[int] = field(default_factory=lambda: {1})
    stop_cmd: set[int] = field(default_factory=lambda: {0})
    emergency: set[int] = field(default_factory=set)
    use_taskstatus_transitions: bool = False
    start_task_status_active: bool = False

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "TaskMapping":
        return cls(
            prepare_status=_int_set(args.prepare_task_status_values, (1,)),
            running_status=_int_set(args.running_task_status_values, (2,)),
            stop_status=_int_set(args.stop_task_status_values, (0, 3)),
            start_cmd=_int_set(args.start_task_cmd_values, (1,)),
            stop_cmd=_int_set(args.stop_task_cmd_values, (0,)),
            emergency=_int_set(args.emergency_values, ()),
            use_taskstatus_transitions=bool(args.use_taskstatus_transitions),
            start_task_status_active=bool(args.start_task_status_active),
        )


@dataclass
class TaskDecision:
    state: str
    reason: str
    task_id: str = ""
    case_id: str = ""
    scene_type: str = ""
    raw_type: int | None = None
    raw_task_type: int | None = None
    raw_task_status: int | None = None


def _find_nested_value(payload: Any, names: Iterable[str]) -> Any:
    if isinstance(payload, dict):
        for key, value in payload.items():
            if key in names and value not in (None, ""):
                return value
        for value in payload.values():
            found = _find_nested_value(value, names)
            if found is not None:
                return found
    elif isinstance(payload, list):
        for value in payload:
            found = _find_nested_value(value, names)
            if found is not None:
                return found
    return None


def interpret_task(
    payload: Any,
    mapping: TaskMapping,
    *,
    source: str,
) -> TaskDecision:
    """把一条平台任务消息解释成 PREPARE / ACTIVE / STANDBY / EMERGENCY。"""
    if not isinstance(payload, dict):
        return TaskDecision(STATE_NOOP, "payload_not_dict")

    raw_type = _normalize_enum(
        first_value(payload, ("type", "msgType", "msg_type", "protocolType", "protocol_type"))
    )
    raw_task_type = _normalize_enum(
        first_value(
            payload,
            ("taskType", "task_type", "taskCmd", "task_cmd", "cmd", "command", "action", "operate", "operation"),
        )
    )
    raw_task_status = _normalize_enum(
        first_value(payload, ("taskStatus", "task_status", "status", "state"))
    )
    emergency_flag = first_value(
        payload, ("emergency", "estop", "eStop", "emergencyStop", "emergency_stop")
    )

    task_id = str(
        first_value(payload, ("taskId", "taskID", "task_id", "taskid"), "")
        or _find_nested_value(payload, ("taskId", "taskID", "task_id"))
        or ""
    )
    case_id = str(
        first_value(payload, ("caseId", "caseID", "case_id", "caseid"), "")
        or _find_nested_value(payload, ("caseId", "caseID", "case_id"))
        or ""
    )
    scene_type = str(
        first_value(payload, ("sceneType", "scene_type", "scenarioType", "scenario_type"), "")
        or _find_nested_value(payload, ("sceneType", "scene_type", "scenarioType"))
        or ""
    )

    reason = (
        f"source={source} type={raw_type} taskType={raw_task_type} "
        f"taskStatus={raw_task_status} emergency={emergency_flag}"
    )

    if as_bool(emergency_flag, False):
        return TaskDecision(STATE_EMERGENCY, reason + " emergency_flag", task_id, case_id, scene_type,
                            raw_type, raw_task_type, raw_task_status)
    if raw_type in mapping.emergency or raw_task_type in mapping.emergency or raw_task_status in mapping.emergency:
        return TaskDecision(STATE_EMERGENCY, reason + " emergency_value", task_id, case_id, scene_type,
                            raw_type, raw_task_type, raw_task_status)

    # /taskstatus 是周期消息，默认不参与状态切换，避免旧值把 ACTIVE 反复打断。
    use_status = mapping.use_taskstatus_transitions or source == "start_task"

    if raw_type == 1:
        return TaskDecision(STATE_PREPARE, reason + " type=1", task_id, case_id, scene_type,
                            raw_type, raw_task_type, raw_task_status)

    if raw_type == 2:
        if raw_task_type in mapping.start_cmd:
            return TaskDecision(STATE_ACTIVE, reason + " type=2 start", task_id, case_id, scene_type,
                                raw_type, raw_task_type, raw_task_status)
        if raw_task_type in mapping.stop_cmd:
            return TaskDecision(STATE_STANDBY, reason + " type=2 stop", task_id, case_id, scene_type,
                                raw_type, raw_task_type, raw_task_status)
        if use_status and raw_task_status in mapping.running_status:
            return TaskDecision(STATE_ACTIVE, reason + " type=2 status=running", task_id, case_id, scene_type,
                                raw_type, raw_task_type, raw_task_status)
        # 停止状态永远优先：/taskstatus 报停时必须回到 STANDBY，不能继续控车。
        if (use_status or source == "taskstatus") and raw_task_status in mapping.stop_status:
            return TaskDecision(STATE_STANDBY, reason + " type=2 status=stop", task_id, case_id, scene_type,
                                raw_type, raw_task_type, raw_task_status)

    if raw_type is None:
        # 有些车端 /start_task 只有 taskStatus。若现场确认“非 0 即开始”，
        # 用 --start-task-status-active 打开这个快捷映射。
        if (
            mapping.start_task_status_active
            and source == "start_task"
            and raw_task_type is None
            and raw_task_status is not None
        ):
            if raw_task_status == 0:
                return TaskDecision(STATE_STANDBY, reason + " start_task status=0", task_id, case_id, scene_type,
                                    raw_type, raw_task_type, raw_task_status)
            return TaskDecision(STATE_ACTIVE, reason + " start_task status!=0", task_id, case_id, scene_type,
                                raw_type, raw_task_type, raw_task_status)
        if raw_task_type in mapping.start_cmd:
            return TaskDecision(STATE_ACTIVE, reason + " taskType=start", task_id, case_id, scene_type,
                                raw_type, raw_task_type, raw_task_status)
        if raw_task_type in mapping.stop_cmd:
            return TaskDecision(STATE_STANDBY, reason + " taskType=stop", task_id, case_id, scene_type,
                                raw_type, raw_task_type, raw_task_status)
        if use_status and raw_task_status in mapping.prepare_status:
            return TaskDecision(STATE_PREPARE, reason + " status=prepare", task_id, case_id, scene_type,
                                raw_type, raw_task_type, raw_task_status)
        if use_status and raw_task_status in mapping.running_status:
            return TaskDecision(STATE_ACTIVE, reason + " status=running", task_id, case_id, scene_type,
                                raw_type, raw_task_type, raw_task_status)
        if (use_status or source == "taskstatus") and raw_task_status in mapping.stop_status:
            return TaskDecision(STATE_STANDBY, reason + " status=stop", task_id, case_id, scene_type,
                                raw_type, raw_task_type, raw_task_status)

    return TaskDecision(STATE_NOOP, reason, task_id, case_id, scene_type,
                        raw_type, raw_task_type, raw_task_status)


class TaskStateMachine:
    """线程安全的任务状态容器。"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.state = STATE_STANDBY
        self.task_id = ""
        self.case_id = ""
        self.scene_type = ""
        self.reason = "init"
        self.updated_ms = now_ms()

    def apply(self, decision: TaskDecision) -> bool:
        if decision.state == STATE_NOOP:
            return False
        with self._lock:
            incoming_task = decision.task_id or self.task_id
            if (
                decision.state == STATE_PREPARE
                and self.state == STATE_ACTIVE
                and incoming_task == self.task_id
            ):
                # /start_task 可能是周期消息；同一任务重复的“准备”不能把 ACTIVE 打回 PREPARE。
                self.updated_ms = now_ms()
                return False
            if decision.task_id:
                self.task_id = decision.task_id
            if decision.case_id:
                self.case_id = decision.case_id
            if decision.scene_type:
                self.scene_type = decision.scene_type
            changed = (
                self.state != decision.state
                or self.reason != decision.reason
                or self.updated_ms <= 0
            )
            self.state = decision.state
            self.reason = decision.reason
            self.updated_ms = now_ms()
            return changed

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "state": self.state,
                "task_id": self.task_id,
                "case_id": self.case_id,
                "scene_type": self.scene_type,
                "reason": self.reason,
                "stamp_ms": now_ms(),
                "updated_ms": self.updated_ms,
            }


class RouteStore:
    """线程安全地保存当前任务路线。"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._route: dict[str, Any] | None = None
        self._json = ""

    def set(self, route: dict[str, Any] | None) -> bool:
        if not route:
            return False
        route = dict(route)
        route.setdefault("stamp_ms", now_ms())
        payload = json.dumps(route, ensure_ascii=False, sort_keys=True)
        with self._lock:
            if payload == self._json:
                return False
            self._route = route
            self._json = payload
            return True

    def clear(self) -> None:
        with self._lock:
            self._route = None
            self._json = ""

    def has(self) -> bool:
        with self._lock:
            return self._route is not None

    def get(self) -> dict[str, Any] | None:
        with self._lock:
            return dict(self._route) if self._route is not None else None

    def to_json(self) -> str:
        with self._lock:
            return self._json


# ---------------------------------------------------------------------------
# task-manager
# ---------------------------------------------------------------------------

class TaskManagerRole:
    def __init__(self, args: argparse.Namespace) -> None:
        import rospy
        from std_msgs.msg import String

        self.rospy = rospy
        self.String = String
        self.args = args
        self.mapping = TaskMapping.from_args(args)
        self.state_machine = TaskStateMachine()
        self.route_store = RouteStore()
        self._raw_counts: dict[str, int] = {}
        self._latest_test_trajectory: Any = None
        self._latest_test_trajectory_time = 0.0
        self._latest_hd_route: Any = None
        self._latest_hd_route_time = 0.0
        # /test_trajectory_req 可能把起点、终点拆成两条消息发；按 siteType 缓存。
        self._site_points: dict[int, dict[str, float]] = {}
        self._site_points_time = 0.0
        self._manual_route = manual_route_from_args(args)

        if self._manual_route and args.prefer_manual_route:
            self.route_store.set(self._manual_route)
            log(f"[task-manager] 使用手动路线：{json.dumps(self._manual_route, ensure_ascii=False)}")

        rospy.init_node(args.node_name or "student_task_manager")
        self.state_pub = rospy.Publisher(
            args.state_topic, String, queue_size=10, latch=True
        )
        self.route_pub = rospy.Publisher(
            args.route_topic, String, queue_size=10, latch=True
        )

        start_cls = resolve_msg_class(
            "ROS_START_TASK_MSG",
            ("device_msgs.msg.taskpublish", "taskpublish.msg.taskpublish"),
            "device_msgs/taskpublish",
        )
        status_cls = resolve_msg_class(
            "ROS_TASK_STATUS_MSG",
            ("device_msgs.msg.taskstatus", "taskstatus.msg.taskstatus"),
            "device_msgs/taskstatus",
        )
        test_traj_cls = resolve_msg_class(
            "ROS_TEST_TRAJECTORY_MSG",
            ("device_msgs.msg.participantTrajectories", "participantTrajectories.msg.participantTrajectories"),
            "device_msgs/participantTrajectories",
        )
        hd_route_cls = resolve_msg_class(
            "ROS_HD_ROUTE_MSG",
            ("common_msgs.msg.hdroute", "hdroute.msg.hdroute"),
            "common_msgs/hdroute",
        )

        self._subscribe(start_cls, args.start_task_topic, self._on_start_task)
        self._subscribe(status_cls, args.task_status_topic, self._on_task_status)
        self._subscribe(test_traj_cls, args.test_trajectory_topic, self._on_test_trajectory)
        self._subscribe(hd_route_cls, args.hd_route_topic, self._on_hd_route)

        log(
            "[task-manager] started "
            f"start_task={args.start_task_topic} taskstatus={args.task_status_topic} "
            f"test_trajectory={args.test_trajectory_topic} hd_route={args.hd_route_topic}"
        )

    def _subscribe(self, message_class, topic: str, callback) -> None:
        if message_class is None:
            log(f"[task-manager] 跳过订阅 {topic}：消息类未解析")
            return
        self.rospy.Subscriber(topic, message_class, callback, queue_size=10)
        log(f"[task-manager] subscribed {topic}")

    def _on_start_task(self, message) -> None:
        self._handle_task_message(message, "start_task")

    def _on_task_status(self, message) -> None:
        self._handle_task_message(message, "taskstatus")

    def _handle_task_message(self, message, source: str) -> None:
        payload = parse_json_strings(ros_msg_to_dict(message))
        self._log_raw(source, payload)
        previous = self.state_machine.snapshot()
        decision = interpret_task(payload, self.mapping, source=source)
        changed = self.state_machine.apply(decision)

        manual_route_active = bool(self._manual_route and self.args.prefer_manual_route)
        new_task = bool(decision.task_id) and decision.task_id != str(previous.get("task_id", ""))
        if (new_task or decision.state == STATE_STANDBY) and not manual_route_active:
            self._site_points.clear()
            if self.route_store.has():
                self.route_store.clear()
                self._publish_route_cleared(decision.task_id)
                log(
                    f"[task-manager] 清除旧路线 "
                    f"(new_task={new_task} state={decision.state})"
                )

        if decision.state in (STATE_PREPARE, STATE_ACTIVE) and not self.route_store.has():
            route = self._resolve_route_from_sources(payload)
            if route:
                self._set_route(route, f"{source} payload")

        if changed:
            log(f"[task-manager] state -> {decision.state} ({decision.reason})")
            self._publish_state()
            self._publish_route()

    def _resolve_route_from_sources(self, payload: Any) -> dict[str, Any] | None:
        route = extract_route(
            payload,
            ego_id=self.args.ego_ptc_id,
            allow_flat=True,
            require_site_type=False,
        )
        if route:
            return route

        max_age = float(self.args.route_source_max_age_s)
        if self._latest_test_trajectory is not None:
            age = time.monotonic() - self._latest_test_trajectory_time
            if age <= max_age:
                route = extract_route(
                    self._latest_test_trajectory,
                    ego_id=self.args.ego_ptc_id,
                    allow_flat=self.args.allow_flat_route_from_test_trajectory,
                    require_site_type=not self.args.allow_flat_route_from_test_trajectory,
                )
                if route:
                    route["source"] = f"test_trajectory_req(age={age:.1f}s)"
                    return route
            else:
                log(f"[task-manager] 忽略过期 test_trajectory_req（age={age:.1f}s）")

        cached_route = self._route_from_site_cache()
        if cached_route:
            return cached_route

        if self._latest_hd_route is not None:
            age = time.monotonic() - self._latest_hd_route_time
            if age <= max_age:
                route = extract_route(
                    self._latest_hd_route,
                    ego_id=self.args.ego_ptc_id,
                    allow_flat=True,
                    require_site_type=False,
                )
                if route:
                    route["source"] = f"hd_route_points(age={age:.1f}s)"
                    return route
            else:
                log(f"[task-manager] 忽略过期 hd_route_points（age={age:.1f}s）")

        return None

    def _on_test_trajectory(self, message) -> None:
        payload = parse_json_strings(ros_msg_to_dict(message))
        self._latest_test_trajectory = payload
        self._latest_test_trajectory_time = time.monotonic()
        self._remember_site_points(payload)
        self._log_raw("test_trajectory_req", payload)
        if self._route_source_allowed() and not self.route_store.has():
            route = extract_route(
                payload,
                ego_id=self.args.ego_ptc_id,
                allow_flat=self.args.allow_flat_route_from_test_trajectory,
                require_site_type=not self.args.allow_flat_route_from_test_trajectory,
            ) or self._route_from_site_cache()
            if route:
                self._set_route(route, "test_trajectory_req")

    def _on_hd_route(self, message) -> None:
        payload = parse_json_strings(ros_msg_to_dict(message))
        self._latest_hd_route = payload
        self._latest_hd_route_time = time.monotonic()
        self._remember_site_points(payload)
        self._log_raw("hd_route_points", payload)
        if self._route_source_allowed() and not self.route_store.has():
            route = extract_route(
                payload,
                ego_id=self.args.ego_ptc_id,
                allow_flat=True,
                require_site_type=False,
            ) or self._route_from_site_cache()
            if route:
                self._set_route(route, "hd_route_points")

    def _remember_site_points(self, payload: Any) -> None:
        """缓存 siteType=0/1 的起终点，兼容起点、终点分两条消息下发的情况。"""
        updated = False
        for _, mapping in walk_mappings(payload):
            point = point_from_mapping(mapping)
            if point and "site_type" in point and int(point["site_type"]) in (0, 1):
                self._site_points[int(point["site_type"])] = point
                updated = True
        if updated:
            self._site_points_time = time.monotonic()

    def _route_from_site_cache(self) -> dict[str, Any] | None:
        if 0 not in self._site_points or 1 not in self._site_points:
            return None
        age = time.monotonic() - self._site_points_time
        if age > float(self.args.route_source_max_age_s):
            return None
        return _make_route(
            self._site_points[0], self._site_points[1], f"site_points_cache(age={age:.1f}s)"
        )

    def _route_source_allowed(self) -> bool:
        # 只在准备/执行阶段接受路线，避免任务停止后又被缓存的轨迹“重新上膛”。
        state = self.state_machine.snapshot().get("state", STATE_STANDBY)
        return state in (STATE_PREPARE, STATE_ACTIVE)

    def _set_route(self, route: dict[str, Any], source: str) -> None:
        snapshot = self.state_machine.snapshot()
        route = dict(route)
        route["task_id"] = snapshot.get("task_id", "")
        route["case_id"] = snapshot.get("case_id", "")
        if self.route_store.set(route):
            log(
                f"[task-manager] route ready source={route.get('source')} via={source} "
                f"start={route.get('start')} goal={route.get('goal')}"
            )
            self._publish_route()

    def _log_raw(self, source: str, payload: Any) -> None:
        count = self._raw_counts.get(source, 0)
        if count >= int(self.args.log_raw_messages):
            return
        self._raw_counts[source] = count + 1
        text = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), default=str)
        if len(text) > 4000:
            text = text[:4000] + "...(truncated)"
        log(f"[task-manager] raw {source} #{count + 1}: {text}")

    def _publish_state(self) -> None:
        payload = self.state_machine.snapshot()
        payload["route_ready"] = self.route_store.has()
        message = self.String()
        message.data = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        self.state_pub.publish(message)

    def _publish_route(self) -> None:
        route_json = self.route_store.to_json()
        if not route_json:
            return
        message = self.String()
        message.data = route_json
        self.route_pub.publish(message)

    def _publish_route_cleared(self, task_id: str) -> None:
        message = self.String()
        message.data = json.dumps(
            {"cleared": True, "task_id": str(task_id or ""), "stamp_ms": now_ms()},
            ensure_ascii=False,
            separators=(",", ":"),
        )
        self.route_pub.publish(message)

    def run(self) -> None:
        rate = self.rospy.Rate(max(float(self.args.bridge_hz), 1.0))
        last_route_json = ""
        while not self.rospy.is_shutdown():
            self._publish_state()
            route_json = self.route_store.to_json()
            if route_json and route_json != last_route_json:
                self._publish_route()
                last_route_json = route_json
            rate.sleep()


# ---------------------------------------------------------------------------
# adapter: sensorobjects -> /student/obstacles JSON
# ---------------------------------------------------------------------------

class SceneAdapterRole:
    def __init__(self, args: argparse.Namespace) -> None:
        import rospy
        from std_msgs.msg import String

        self.rospy = rospy
        self.String = String
        self.args = args
        self._lock = threading.Lock()
        self._latest_payload: dict[str, Any] | None = None
        self._ego_speed_mps = 0.0
        self._last_positions: dict[str, tuple[float, float, float]] = {}

        rospy.init_node(args.node_name or "student_scene_adapter")
        self.pub = rospy.Publisher(args.student_obstacles_topic, String, queue_size=10)

        object_cls = resolve_msg_class(
            "ROS_SENSOROBJECTS_MSG",
            ("common_msgs.msg.sensorobjects", "sensorobjects.msg.sensorobjects"),
            "common_msgs/sensorobjects",
        )
        gps_cls = resolve_msg_class(
            "ROS_SENSORGPS_MSG",
            ("common_msgs.msg.sensorgps", "sensorgps.msg.sensorgps"),
            "common_msgs/sensorgps",
        )
        if object_cls is not None:
            rospy.Subscriber(
                args.object_track_topic, object_cls, self._on_objects, queue_size=10
            )
            log(f"[adapter] subscribed {args.object_track_topic}")
        if gps_cls is not None:
            rospy.Subscriber(args.sensorgps_topic, gps_cls, self._on_gps, queue_size=20)

        log(
            f"[adapter] started object_track={args.object_track_topic} "
            f"frame={args.object_track_frame} heading={args.object_heading_mode} "
            f"speed={args.object_speed_mode}"
        )

    def _on_gps(self, message) -> None:
        self._ego_speed_mps = as_float(first_value(message, ("velocity", "speed")), 0.0)

    def _on_objects(self, message) -> None:
        payload = parse_json_strings(ros_msg_to_dict(message))
        raw_objects = first_value(payload, ("obs", "objects", "tracks", "pants"), [])
        if not isinstance(raw_objects, list):
            return

        stamp_ms = now_ms()
        converted = []
        for raw in raw_objects:
            if not isinstance(raw, dict):
                continue
            obstacle = self._convert_object(raw, stamp_ms)
            if obstacle is not None:
                converted.append(obstacle)

        with self._lock:
            self._latest_payload = {
                "timestamp": stamp_ms,
                "frame": self.args.object_track_frame,
                "obstacles": converted,
            }

    def _convert_object(self, raw: dict[str, Any], stamp_ms: int) -> dict[str, Any] | None:
        object_id = first_value(raw, ("id", "objectId", "objId", "trackId", "radarObjectID"))
        if object_id is None or as_int(object_id, 0) == 0:
            return None
        track_id = str(as_int(object_id, 0))

        lat = first_value(raw, ("latitude", "lat", "pos_lat"))
        lon = first_value(raw, ("longtitude", "longitude", "lon", "lng", "pos_lon"))
        has_latlon = (
            lat is not None
            and lon is not None
            and abs(as_float(lat)) > 1e-6
            and abs(as_float(lon)) > 1e-6
        )
        x = first_value(raw, ("x", "pos_x"))
        y = first_value(raw, ("y", "pos_y"))
        if not has_latlon and (x is None or y is None):
            return None

        frame = "latlon" if has_latlon else self.args.object_track_frame
        width = max(0.0, as_float(first_value(raw, ("width",)), 0.0))
        length = max(0.0, as_float(first_value(raw, ("length",)), 0.0))
        height = max(0.0, as_float(first_value(raw, ("height",)), 0.0))
        ptc_type = as_int(first_value(raw, ("classification", "ptc_type", "type"), 1), 1)

        position_key = track_id
        previous = self._last_positions.get(position_key)
        previous_xy = None
        if previous is not None:
            previous_xy = (previous[1], previous[2])

        if has_latlon:
            current_xy = (as_float(lon), as_float(lat))
        else:
            current_xy = (as_float(x), as_float(y))

        speed = self._estimate_speed(raw, previous, current_xy, has_latlon, stamp_ms)
        heading = self._estimate_heading(raw, previous_xy, current_xy, has_latlon)

        self._last_positions[position_key] = (stamp_ms, current_xy[0], current_xy[1])

        obstacle: dict[str, Any] = {
            "id": track_id,
            "frame": frame,
            "speed": float(max(0.0, speed)),
            "heading": float(heading),
            "accel": as_float(first_value(raw, ("accel", "acceleration", "acc")), 0.0),
            "length": length,
            "width": width,
            "height": height,
            "ptc_type": ptc_type,
            "source": 2,
            "isvalid": 1,
        }
        if has_latlon:
            obstacle["lat"] = as_float(lat)
            obstacle["lon"] = as_float(lon)
        else:
            obstacle["x"] = as_float(x)
            obstacle["y"] = as_float(y)
        return obstacle

    def _estimate_speed(
        self,
        raw: dict[str, Any],
        previous: tuple[float, float, float] | None,
        current_xy: tuple[float, float],
        has_latlon: bool,
        stamp_ms: int,
    ) -> float:
        mode = self.args.object_speed_mode
        if mode in ("auto", "speeds"):
            speed = first_value(raw, ("speeds", "speed", "velocity", "v"))
            if speed is not None:
                return as_float(speed, 0.0)
            # 没有绝对速度时，用相对纵向速度 + 自车速度做一个保守回退。
            if mode == "auto":
                relative = first_value(raw, ("relspeedy",))
                if relative is not None:
                    return max(0.0, self._ego_speed_mps + as_float(relative, 0.0))

        if previous is not None and mode in ("auto", "diff"):
            previous_ms, previous_x, previous_y = previous
            dt = max((stamp_ms - previous_ms) / 1000.0, 1e-3)
            if has_latlon:
                distance = self._approx_distance_m(previous_y, previous_x, current_xy[1], current_xy[0])
            else:
                distance = math.hypot(current_xy[0] - previous_x, current_xy[1] - previous_y)
            return distance / dt

        return 0.0

    def _estimate_heading(
        self,
        raw: dict[str, Any],
        previous_xy: tuple[float, float] | None,
        current_xy: tuple[float, float],
        has_latlon: bool,
    ) -> float:
        if previous_xy is not None:
            if has_latlon:
                dx = (current_xy[0] - previous_xy[0]) * 111320.0 * math.cos(
                    math.radians(current_xy[1])
                )
                dy = (current_xy[1] - previous_xy[1]) * 110540.0
            else:
                dx = current_xy[0] - previous_xy[0]
                dy = current_xy[1] - previous_xy[1]
            if math.hypot(dx, dy) > 0.05:
                return math.atan2(dy, dx)

        azimuth = first_value(raw, ("azimuth", "heading", "courseAngle", "theta"))
        if azimuth is None:
            return 0.0

        value = as_float(azimuth, 0.0)
        mode = self.args.object_heading_mode
        if mode == "math":
            return value if abs(value) <= 2.0 * math.pi else math.radians(value)
        if mode == "compass":
            degrees = value if abs(value) > 2.0 * math.pi else math.degrees(value)
            return math.radians(90.0 - degrees)
        # auto: sensorobject.azimuth 约定是 rad；超过 2π 才按度处理。
        if abs(value) <= 2.0 * math.pi:
            return value
        return math.radians(90.0 - value)

    @staticmethod
    def _approx_distance_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        dx = (lon2 - lon1) * 111320.0 * math.cos(math.radians((lat1 + lat2) / 2.0))
        dy = (lat2 - lat1) * 110540.0
        return math.hypot(dx, dy)

    def run(self) -> None:
        rate = self.rospy.Rate(max(float(self.args.bridge_hz), 1.0))
        while not self.rospy.is_shutdown():
            with self._lock:
                payload = dict(self._latest_payload) if self._latest_payload else None
            if payload is not None:
                payload["stamp_ms"] = now_ms()
                message = self.String()
                message.data = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
                self.pub.publish(message)
            rate.sleep()


# ---------------------------------------------------------------------------
# mux: /student/controlcmd -> /controlcmd
# ---------------------------------------------------------------------------

class ControlMuxRole:
    def __init__(self, args: argparse.Namespace) -> None:
        import rospy
        from std_msgs.msg import String

        self.rospy = rospy
        self.String = String
        self.args = args
        self.control_cls = resolve_msg_class(
            "ROS_CONTROLCMD_MSG",
            ("common_msgs.msg.controlcmd", "controlcmd.msg.controlcmd"),
            "common_msgs/controlcmd",
        )
        if self.control_cls is None:
            raise SystemExit("无法解析 controlcmd 消息类，请先 source 车端 ~/redis_stu/devel/setup.bash")

        self._lock = threading.Lock()
        self._state = STATE_STANDBY
        self._state_time = 0.0
        self._student_command = None
        self._student_time = 0.0
        self._last_log_time = 0.0
        self._last_mode = ""
        self._missing_fields: set[str] = set()

        rospy.init_node(args.node_name or "student_control_mux")
        self.pub = rospy.Publisher(args.controlcmd_topic, self.control_cls, queue_size=10)
        rospy.Subscriber(
            args.student_cmd_topic, self.control_cls, self._on_student_command, queue_size=10
        )
        rospy.Subscriber(args.state_topic, String, self._on_state, queue_size=10)
        self.status_pub = rospy.Publisher("/student/mux_status", String, queue_size=10)

        log(
            f"[mux] started in={args.student_cmd_topic} out={args.controlcmd_topic} "
            f"hz={args.bridge_hz} student_timeout={args.student_timeout_s}s "
            f"state_timeout={args.state_timeout_s}s"
        )

    def _on_student_command(self, message) -> None:
        with self._lock:
            self._student_command = message
            self._student_time = time.monotonic()

    def _on_state(self, message) -> None:
        try:
            payload = json.loads(message.data)
        except Exception:
            payload = {}
        state = str(payload.get("state", STATE_STANDBY)).upper()
        if state not in _VALID_STATES:
            state = STATE_STANDBY
        with self._lock:
            previous_state = self._state
            self._state = state
            self._state_time = time.monotonic()
        # 只在状态变化时立刻补一帧 HOLD，避免任务停止后最多 50 ms 的空窗；
        # 周期 HOLD 由 run() 的 20 Hz 循环负责。
        if state != STATE_ACTIVE and state != previous_state:
            self._publish(self._hold_command(), f"state={state}")

    def _build_message(self, fields: dict[str, Any]):
        message = self.control_cls()
        for name, value in fields.items():
            if hasattr(message, name):
                setattr(message, name, value)
            elif name not in self._missing_fields:
                self._missing_fields.add(name)
                log(f"[mux] ⚠ controlcmd 没有字段 {name}，该字段不会下发，请核对 ~/redis_stu 的消息定义")
        return message

    def _publish(self, fields: dict[str, Any], reason: str = "") -> None:
        try:
            self.pub.publish(self._build_message(fields))
        except Exception as exc:
            if not self.rospy.is_shutdown():
                log(f"[mux] publish failed: {exc}")
        if reason:
            self._log_mode(reason)

    def _log_mode(self, reason: str) -> None:
        current = time.monotonic()
        if reason != self._last_mode or current - self._last_log_time >= 1.0:
            log(f"[mux] output {reason}")
            self._last_mode = reason
            self._last_log_time = current

    def _hold_command(self) -> dict[str, Any]:
        return {
            "timestamp": now_ms(),
            "acceleration": 0.0,
            "wheel_angle": 0.0,
            "gear": HOLD_GEAR,
            "turnlight": 0,
            "brkreq": int(self.args.hold_brkreq),
        }

    def _emergency_command(self) -> dict[str, Any]:
        return {
            "timestamp": now_ms(),
            "acceleration": 0.0,
            "wheel_angle": 0.0,
            "gear": HOLD_GEAR,
            "turnlight": 0,
            "brkreq": 1,
        }

    def _sanitize_student_command(self, message) -> dict[str, Any] | None:
        acceleration = as_float(first_value(message, ("acceleration", "speed_kph")), float("nan"))
        wheel_angle = as_float(first_value(message, ("wheel_angle",)), float("nan"))
        if not math.isfinite(acceleration) or not math.isfinite(wheel_angle):
            return None

        turnlight = as_int(first_value(message, ("turnlight", "turnLight"), 0), 0)
        if turnlight not in (0, 1, 2, 4):
            turnlight = 0
        return {
            "timestamp": now_ms(),
            "acceleration": clamp(acceleration, 0.0, float(self.args.max_speed_kph)),
            "wheel_angle": clamp(wheel_angle, -float(self.args.max_wheel_deg), float(self.args.max_wheel_deg)),
            "gear": HOLD_GEAR,
            "turnlight": turnlight,
            "brkreq": 0,
        }

    def run(self) -> None:
        rate = self.rospy.Rate(max(float(self.args.bridge_hz), 1.0))
        status_rate_counter = 0
        while not self.rospy.is_shutdown():
            current = time.monotonic()
            with self._lock:
                state = self._state
                state_age = current - self._state_time if self._state_time > 0 else float("inf")
                student_command = self._student_command
                student_age = current - self._student_time if self._student_time > 0 else float("inf")

            state_fresh = state_age <= float(self.args.state_timeout_s)
            student_fresh = student_age <= float(self.args.student_timeout_s)

            if state == STATE_EMERGENCY and state_fresh:
                self._publish(self._emergency_command(), "EMERGENCY")
            elif state == STATE_ACTIVE and state_fresh and student_fresh and student_command is not None:
                sanitized = self._sanitize_student_command(student_command)
                if sanitized is not None:
                    self._publish(sanitized, "ACTIVE")
                else:
                    self._publish(self._hold_command(), "ACTIVE but invalid student command")
            else:
                reasons = []
                if state != STATE_ACTIVE:
                    reasons.append(f"state={state}")
                if not state_fresh:
                    reasons.append("state_timeout")
                if not student_fresh:
                    reasons.append("student_timeout")
                self._publish(self._hold_command(), "HOLD " + ",".join(reasons))

            status_rate_counter += 1
            if status_rate_counter >= max(int(float(self.args.bridge_hz) / 2.0), 1):
                status_rate_counter = 0
                status = {
                    "state": state,
                    "state_age_ms": None if not math.isfinite(state_age) else int(state_age * 1000),
                    "student_age_ms": None if not math.isfinite(student_age) else int(student_age * 1000),
                    "forwarding": state == STATE_ACTIVE and state_fresh and student_fresh,
                    "stamp_ms": now_ms(),
                }
                message = self.String()
                message.data = json.dumps(status, ensure_ascii=False, separators=(",", ":"))
                self.status_pub.publish(message)

            rate.sleep()


# ---------------------------------------------------------------------------
# algorithm: DirectControlNode 子类
# ---------------------------------------------------------------------------

def _route_point_to_xy(node, point: dict[str, Any] | None):
    if not isinstance(point, dict):
        return None
    if "x" in point and "y" in point:
        return np.array([float(point["x"]), float(point["y"])], dtype=float)
    if "lat" in point and "lon" in point:
        x, y = node.transformer_to_proj.transform(float(point["lon"]), float(point["lat"]))
        return np.array([x, y], dtype=float)
    return None


class TaskDrivenDirectControlNode:
    """用工厂函数构造，避免在非 algorithm 角色导入 direct_control_node。"""

    @staticmethod
    def build(args: argparse.Namespace):
        from direct_control_node import DirectControlNode, ObstacleState

        class _Node(DirectControlNode):
            def __init__(self, node_args: argparse.Namespace) -> None:
                self._bridge_lock = threading.Lock()
                self._task_state = STATE_STANDBY
                self._task_id = ""
                self._task_route: dict[str, Any] | None = None
                self._route_token = ""
                self._initialized_route_token = ""
                self._emergency = False
                self._last_hold_log = 0.0
                self._last_route_log = 0.0
                self._last_scene_state_log = 0.0
                super().__init__(node_args)

                from std_msgs.msg import String

                self.rospy.Subscriber(
                    self.args.state_topic, String, self._bridge_state_callback, queue_size=10
                )
                self.rospy.Subscriber(
                    self.args.route_topic, String, self._bridge_route_callback, queue_size=10
                )
                self.rospy.Subscriber(
                    self.args.student_obstacles_topic,
                    String,
                    self._bridge_obstacles_callback,
                    queue_size=10,
                )
                log(
                    f"[algorithm] STANDBY, state_topic={self.args.state_topic} "
                    f"route_topic={self.args.route_topic} "
                    f"student_cmd={self.args.controlcmd_topic}"
                )

            # ---- 任务/路线回调 ------------------------------------------------

            def _bridge_state_callback(self, message) -> None:
                try:
                    payload = json.loads(message.data)
                except Exception:
                    payload = {}
                state = str(payload.get("state", STATE_STANDBY)).upper()
                if state not in _VALID_STATES:
                    state = STATE_STANDBY
                task_id = str(payload.get("task_id", "") or "")
                with self._bridge_lock:
                    previous = self._task_state
                    self._task_state = state
                    self._emergency = state == STATE_EMERGENCY or as_bool(
                        payload.get("emergency"), False
                    )
                    if task_id and task_id != self._task_id:
                        self._task_id = task_id
                        self._initialized_route_token = ""
                        self.route_initialized = False
                        route_task = (
                            str(self._task_route.get("task_id", ""))
                            if self._task_route is not None
                            else ""
                        )
                        # 旧任务的路线不能带到新任务；手动路线（task_id 为空）保留。
                        if route_task and route_task != task_id:
                            self._task_route = None
                            self._route_token = ""
                            log(
                                f"[algorithm] 丢弃旧任务路线 task_id={route_task} "
                                f"(new={task_id})"
                            )
                if state != previous:
                    log(f"[algorithm] task state {previous} -> {state} task_id={task_id}")

            def _bridge_route_callback(self, message) -> None:
                try:
                    route = json.loads(message.data)
                except Exception as exc:
                    log(f"[algorithm] 路线 JSON 解析失败：{exc}")
                    return
                if not isinstance(route, dict):
                    return
                if as_bool(route.get("cleared"), False):
                    with self._bridge_lock:
                        self._task_route = None
                        self._route_token = ""
                        self._initialized_route_token = ""
                        self.route_initialized = False
                    log("[algorithm] route cleared by task-manager")
                    return
                token = json.dumps(route, ensure_ascii=False, sort_keys=True)
                with self._bridge_lock:
                    self._task_route = route
                    self._route_token = token
                    if str(route.get("task_id", "")) and str(route.get("task_id", "")) != self._task_id:
                        self._task_id = str(route.get("task_id", ""))
                        self._initialized_route_token = ""
                        self.route_initialized = False
                current = time.time()
                if current - self._last_route_log >= 1.0:
                    log(
                        f"[algorithm] route received source={route.get('source')} "
                        f"start={route.get('start')} goal={route.get('goal')}"
                    )
                    self._last_route_log = current

            def _bridge_obstacles_callback(self, message) -> None:
                try:
                    payload = json.loads(message.data)
                except Exception:
                    return
                if not isinstance(payload, dict):
                    return
                obstacles = payload.get("obstacles", [])
                if not isinstance(obstacles, list):
                    return
                stamp_ms = as_int(payload.get("timestamp"), now_ms())
                default_frame = str(payload.get("frame", "map"))

                updated: dict[str, ObstacleState] = {}
                for item in obstacles:
                    if not isinstance(item, dict):
                        continue
                    track_id = str(item.get("id", "") or "")
                    if not track_id:
                        continue
                    if as_int(item.get("isvalid", 1), 1) == 0:
                        continue

                    frame = str(item.get("frame", default_frame))
                    speed = max(0.0, as_float(item.get("speed"), 0.0))
                    heading = as_float(item.get("heading"), 0.0)
                    accel = as_float(item.get("accel"), 0.0)
                    width = max(0.0, as_float(item.get("width"), 0.0))
                    length = max(0.0, as_float(item.get("length"), 0.0))
                    height = max(0.0, as_float(item.get("height"), 0.0))
                    ptc_type = as_int(item.get("ptc_type"), 1)

                    lat = item.get("lat")
                    lon = item.get("lon")
                    if lat is not None and lon is not None and abs(as_float(lat)) > 1e-6 and abs(as_float(lon)) > 1e-6:
                        x, y = self.transformer_to_proj.transform(
                            as_float(lon), as_float(lat)
                        )
                    elif frame == "ego":
                        pose = self.current_pose_to_xy_heading()
                        if pose is None:
                            continue
                        local_x = as_float(item.get("x"), 0.0)
                        local_y = as_float(item.get("y"), 0.0)
                        cos_h = math.cos(pose["heading"])
                        sin_h = math.sin(pose["heading"])
                        x = pose["x"] + local_x * cos_h - local_y * sin_h
                        y = pose["y"] + local_x * sin_h + local_y * cos_h
                        heading = pose["heading"] + heading
                    else:
                        x = as_float(item.get("x"), 0.0)
                        y = as_float(item.get("y"), 0.0)

                    updated[track_id] = ObstacleState(
                        ptc_id=track_id,
                        x=float(x),
                        y=float(y),
                        v=float(speed),
                        heading=float(heading),
                        a=float(accel),
                        length=float(length),
                        width=float(width),
                        height=float(height),
                        timestamp_ms=int(stamp_ms),
                        source=as_int(item.get("source"), 2),
                        ptc_type=int(ptc_type),
                        source_id=str(item.get("source_id", "")),
                    )

                if updated:
                    with self.obu_lock:
                        for track_id, obstacle in updated.items():
                            previous = self.latest_obstacles.get(track_id)
                            if previous is None or obstacle.timestamp_ms >= previous.timestamp_ms:
                                self.latest_obstacles[track_id] = obstacle

            # ---- 输出适配 ----------------------------------------------------

            @staticmethod
            def build_hold_command():
                return {
                    "timestamp": now_ms(),
                    "speed_kph": 0.0,
                    "front_wheel_angle_deg": 0.0,
                    "gear": HOLD_GEAR,
                    "turnlight": 0,
                    "brkreq": 0,
                }

            def trajectory_to_control_cmd(self, trajectory):
                command = super().trajectory_to_control_cmd(trajectory)
                # 协议里 brkreq=1 是急刹；普通停车只用 acceleration=0。
                command["brkreq"] = 1 if self._emergency else 0
                return command

            def evaluate_actuator_ready(self):
                if not bool(self.args.require_actuator_feedback):
                    return super().evaluate_actuator_ready()
                with self.actuator_lock:
                    actuator = self.latest_actuator
                if actuator is None:
                    return False, "no_feedback"
                return super().evaluate_actuator_ready()

            # ---- 路线初始化 --------------------------------------------------

            def _bridge_snapshot(self) -> tuple[str, bool, dict[str, Any] | None, str]:
                with self._bridge_lock:
                    return (
                        self._task_state,
                        self._emergency,
                        dict(self._task_route) if self._task_route is not None else None,
                        self._route_token,
                    )

            def _set_hold(self, reason: str) -> None:
                with self.command_lock:
                    self.latest_command = self.build_hold_command()
                current = time.time()
                if current - self._last_hold_log >= 1.0:
                    log(f"[algorithm] HOLD ({reason})")
                    self._last_hold_log = current

            def _log_scene_state(self, obstacles) -> None:
                """场景要求停车时打印原因和当前障碍物 id，便于现场定位。"""
                history = getattr(self.gvt.scene_target_provider, "history", None)
                if not history:
                    return
                last = history[-1]
                mode = str(last.get("control_mode", ""))
                status = str(last.get("scene_status", ""))
                if mode != "emergency_stop" and "invalid" not in status:
                    return
                current = time.time()
                if current - self._last_scene_state_log < 1.0:
                    return
                self._last_scene_state_log = current
                obstacle_ids = sorted(
                    str(getattr(obstacle, "ptc_id", "")) for obstacle in obstacles
                )
                log(
                    f"[algorithm] scene stop mode={mode} status={status} "
                    f"detail={last.get('scene_detail', '')} "
                    f"obstacle_ids=[{','.join(obstacle_ids)}]"
                )

            def _ensure_route_initialized(self) -> bool:
                state, _, route, token = self._bridge_snapshot()
                if state != STATE_ACTIVE or route is None:
                    return False
                if self.route_initialized and self._initialized_route_token == token:
                    return True

                goal_xy = _route_point_to_xy(self, route.get("goal"))
                if goal_xy is None:
                    log("[algorithm] 路线缺少有效终点")
                    return False

                start_point = route.get("start")
                # 平台下发的起点仍然保留：它代表任务意图，用于场景路线意图输入。
                route_start_xy = _route_point_to_xy(self, start_point) if start_point else None
                if self.args.use_current_start or not start_point:
                    if not self.pose_ready_event.wait(timeout=2.0):
                        return False
                    pose = self.current_pose_to_xy_heading()
                    if pose is None:
                        return False
                    start_xy = np.array([pose["x"], pose["y"]], dtype=float)
                else:
                    start_xy = route_start_xy
                    if start_xy is None:
                        log("[algorithm] 路线缺少有效起点")
                        return False

                # 平台任务路线是自车的路线，不能拿来当主车路线意图。
                # 主车路线只认 --scene-main-start-xy/--scene-main-goal-xy；
                # 未配置时保持为空，场景会明确报 route_intent_missing。
                main_start_xy = self.gvt.scene_target_provider.main_route_start_xy
                main_goal_xy = self.gvt.scene_target_provider.main_route_goal_xy
                log(
                    f"[algorithm] main route intent start={main_start_xy.tolist() if main_start_xy is not None else None} "
                    f"goal={main_goal_xy.tolist() if main_goal_xy is not None else None}"
                )

                try:
                    self.gvt.global_planning(start_xy, goal_xy)
                except Exception as exc:
                    log(f"[algorithm] 全局路径初始化失败：{exc}")
                    self.route_initialized = False
                    return False

                self.start_xy = start_xy
                self.goal_xy = goal_xy
                self.route_initialized = True
                self._initialized_route_token = token
                self.goal_reached = False
                self._last_valid_plan_ts = 0.0
                log(
                    f"[algorithm] route initialized start={start_xy.tolist()} "
                    f"route_start={route_start_xy.tolist() if route_start_xy is not None else None} "
                    f"goal={goal_xy.tolist()} source={route.get('source')}"
                )
                return True

            # ---- 规划线程（复制父类逻辑，增加 STANDBY/ACTIVE 门控） ------------

            def run_planning_loop(self):
                interval = 1.0 / max(float(self.args.plan_hz), 1e-3)
                log(f"▶️  任务驱动规划线程启动 (STANDBY, Hz={self.args.plan_hz})")

                while self.running and not self.rospy.is_shutdown():
                    start_time = time.time()
                    state, emergency, _, _ = self._bridge_snapshot()

                    if state != STATE_ACTIVE or emergency:
                        self._set_hold(f"state={state} emergency={emergency}")
                        time.sleep(interval)
                        continue

                    if not self._ensure_route_initialized():
                        self._set_hold("route_not_ready")
                        time.sleep(interval)
                        continue

                    if not self.pose_ready_event.wait(timeout=0.2):
                        time.sleep(interval)
                        continue

                    if not self.update_vehicle_state():
                        self._set_hold("pose_invalid")
                        time.sleep(interval)
                        continue

                    if self.is_arrived():
                        if not self.goal_reached:
                            log("✅ 已到达终点，切换为安全停车命令。")
                        self.goal_reached = True
                        self._set_hold("arrived")
                        time.sleep(interval)
                        continue

                    ready, reason = self.evaluate_actuator_ready()
                    if not ready:
                        self._set_hold(f"actuator_not_ready:{reason}")
                        time.sleep(interval)
                        continue

                    obstacles = self.build_obstacle_list()
                    try:
                        trajectory = self.gvt.traj_planning(obstacles)
                    except Exception as exc:
                        log(f"⚠ 规划异常：{exc}")
                        self._set_hold("planner_exception")
                        time.sleep(interval)
                        continue

                    self._log_scene_state(obstacles)

                    if trajectory is None:
                        if self.gvt.traj_planner.attribute.get("plan_mode") == "arrived":
                            self.goal_reached = True
                            self._set_hold("planner_arrived")
                            time.sleep(interval)
                            continue
                        if start_time - self._last_plan_failure_log_ts >= 1.0:
                            attribute = self.gvt.traj_planner.attribute
                            log(
                                "⚠ 规划未输出轨迹："
                                f"obstacle_count={len(obstacles)} "
                                f"plan_mode={attribute.get('plan_mode', '')} "
                                f"candidate_count={attribute.get('candidate_count', '')} "
                                f"feasible_candidate_count={attribute.get('feasible_candidate_count', '')} "
                                f"failure_reason={attribute.get('failure_reason', '')} "
                                f"decision_reason={attribute.get('reason', '')}"
                            )
                            self._last_plan_failure_log_ts = start_time
                        if self._last_valid_plan_ts > 0.0:
                            if time.time() - self._last_valid_plan_ts < 0.5:
                                time.sleep(interval)
                                continue
                        self._set_hold("plan_failed")
                        time.sleep(interval)
                        continue

                    command = self.trajectory_to_control_cmd(trajectory)
                    self.publish_planned_trajectory(trajectory, command, obstacles)
                    with self.command_lock:
                        self.latest_command = command
                    self._last_valid_plan_ts = time.time()

                    elapsed = time.time() - start_time
                    time.sleep(max(0.0, interval - elapsed))

            def run(self):
                log(f"🚀 任务驱动直控节点启动，地图缓存：{self.cache_path}")
                log(f"🚀 等待 /student/task_state 的 ACTIVE 与 /student/route")
                self.plan_thread.start()
                self.control_thread.start()
                try:
                    while not self.rospy.is_shutdown():
                        time.sleep(0.2)
                except KeyboardInterrupt:
                    pass
                finally:
                    self.stop()

        return _Node(args)


def _cache_loadable(cache_path) -> bool:
    """地图缓存 pkl 可能由另一台机器的 numpy 版本生成，加载失败时返回 False。"""
    try:
        import pickle

        with open(cache_path, "rb") as handle:
            pickle.load(handle)
        return True
    except Exception as exc:
        log(f"[algorithm] 地图缓存不可用：{exc}")
        return False


def run_algorithm_role(args: argparse.Namespace) -> None:
    if args.planner_mode == "scene_target" and not str(args.scene_algorithm_symbol or "").strip():
        raise SystemExit("scene_target 模式必须提供 --scene-algorithm-symbol")

    if args.goal_xy is None and args.goal_latlon is None:
        args.goal_xy = [0.0, 0.0]
    if args.start_xy is None and args.start_latlon is None:
        args.start_xy = [0.0, 0.0]
    args.controlcmd_topic = args.student_cmd_topic

    # 缓存由开发机生成时可能带 numpy 2.x 的模块路径（numpy._core），
    # 车端 numpy 1.x 无法加载；这里提前探测并强制重建，避免节点启动即崩溃。
    try:
        from direct_control_node import resolve_cache_path

        cache_path = resolve_cache_path(args.map_path, args.cache_path)
        if cache_path.exists() and not args.rebuild_cache and not _cache_loadable(cache_path):
            log(f"[algorithm] 将重建地图缓存：{cache_path}")
            args.rebuild_cache = True
    except Exception as exc:
        log(f"[algorithm] 地图缓存预检查跳过：{exc}")

    node = TaskDrivenDirectControlNode.build(args)
    node.run()


# ---------------------------------------------------------------------------
# probe
# ---------------------------------------------------------------------------

def run_probe_role(args: argparse.Namespace) -> None:
    import roslib.message
    import rospy

    rospy.init_node(args.node_name or "student_probe")
    topics = [item.strip() for item in str(args.probe_topics).split(",") if item.strip()]
    topic_types = {name: msg_type for name, msg_type in rospy.get_master().getTopicTypes()[1]}

    log("[probe] topic types:")
    for topic in topics:
        log(f"[probe]   {topic} -> {topic_types.get(topic, '<unknown>')}")

    counts: dict[str, int] = {topic: 0 for topic in topics}
    subscribers: dict[str, Any] = {}
    finished: set[str] = set()

    def make_callback(topic: str):
        def callback(message) -> None:
            counts[topic] += 1
            payload = parse_json_strings(ros_msg_to_dict(message))
            text = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), default=str)
            if len(text) > 8000:
                text = text[:8000] + "...(truncated)"
            log(f"[probe] {topic} #{counts[topic]}: {text}")
            if counts[topic] >= max(int(args.probe_count), 1) and topic not in finished:
                finished.add(topic)
                subscriber = subscribers.get(topic)
                if subscriber is not None:
                    subscriber.unregister()
                if len(finished) >= len(subscribers):
                    log("[probe] all topics received, exiting")
                    rospy.signal_shutdown("probe done")
        return callback

    for topic in topics:
        msg_type = topic_types.get(topic)
        if not msg_type:
            log(f"[probe] 跳过 {topic}：master 上没有该 topic")
            continue
        message_class = roslib.message.get_message_class(msg_type)
        if message_class is None:
            log(f"[probe] 跳过 {topic}：无法导入 {msg_type}")
            continue
        subscribers[topic] = rospy.Subscriber(
            topic, message_class, make_callback(topic), queue_size=10
        )

    log(f"[probe] listening, probe_count={args.probe_count}")
    rospy.spin()


# ---------------------------------------------------------------------------
# 参数
# ---------------------------------------------------------------------------

def build_bridge_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="实车平台任务接入桥（mux / task-manager / adapter / algorithm / probe）"
    )
    parser.add_argument(
        "--role",
        choices=["mux", "task-manager", "adapter", "algorithm", "probe"],
        default="mux",
    )
    parser.add_argument("--node-name", default="")

    parser.add_argument("--state-topic", default=DEFAULT_STATE_TOPIC)
    parser.add_argument("--route-topic", default=DEFAULT_ROUTE_TOPIC)
    parser.add_argument("--student-cmd-topic", default=DEFAULT_STUDENT_CMD_TOPIC)
    parser.add_argument("--student-obstacles-topic", default=DEFAULT_STUDENT_OBSTACLES_TOPIC)
    parser.add_argument("--controlcmd-topic", default=DEFAULT_CONTROLCMD_TOPIC)
    parser.add_argument("--sensorgps-topic", default=DEFAULT_SENSORGPS_TOPIC)
    parser.add_argument("--actuator-topic", default=DEFAULT_ACTUATOR_TOPIC)
    parser.add_argument("--obu-topic", default=DEFAULT_OBU_TOPIC)

    parser.add_argument("--bridge-hz", type=float, default=20.0)
    parser.add_argument("--student-timeout-s", type=float, default=0.5)
    parser.add_argument("--state-timeout-s", type=float, default=2.0)
    parser.add_argument("--max-speed-kph", type=float, default=DEFAULT_MAX_SPEED_KPH)
    parser.add_argument("--max-wheel-deg", type=float, default=DEFAULT_MAX_WHEEL_DEG)
    parser.add_argument("--hold-brkreq", type=int, choices=[0, 1], default=0)

    parser.add_argument("--start-task-topic", default=DEFAULT_START_TASK_TOPIC)
    parser.add_argument("--task-status-topic", default=DEFAULT_TASK_STATUS_TOPIC)
    parser.add_argument("--test-trajectory-topic", default=DEFAULT_TEST_TRAJECTORY_TOPIC)
    parser.add_argument("--hd-route-topic", default=DEFAULT_HD_ROUTE_TOPIC)
    parser.add_argument("--ego-ptc-id", default="")
    parser.add_argument("--manual-start-latlon", nargs=2, type=float)
    parser.add_argument("--manual-goal-latlon", nargs=2, type=float)
    parser.add_argument("--manual-start-xy", nargs=2, type=float)
    parser.add_argument("--manual-goal-xy", nargs=2, type=float)
    parser.add_argument("--prefer-manual-route", dest="prefer_manual_route", action="store_true", default=True)
    parser.add_argument("--no-prefer-manual-route", dest="prefer_manual_route", action="store_false")
    parser.add_argument(
        "--allow-flat-route-from-test-trajectory", action="store_true", default=False
    )
    parser.add_argument("--route-source-max-age-s", type=float, default=30.0)
    parser.add_argument("--use-taskstatus-transitions", action="store_true", default=False)
    parser.add_argument("--start-task-status-active", action="store_true", default=False)
    parser.add_argument("--prepare-task-status-values", default="1")
    parser.add_argument("--running-task-status-values", default="2")
    parser.add_argument("--stop-task-status-values", default="0,3")
    parser.add_argument("--start-task-cmd-values", default="1")
    parser.add_argument("--stop-task-cmd-values", default="0")
    parser.add_argument("--emergency-values", default="")
    parser.add_argument("--log-raw-messages", type=int, default=10)

    parser.add_argument("--object-track-topic", default=DEFAULT_OBJECT_TRACK_TOPIC)
    parser.add_argument("--object-track-frame", choices=["map", "ego"], default="map")
    parser.add_argument("--object-heading-mode", choices=["auto", "compass", "math"], default="auto")
    parser.add_argument("--object-speed-mode", choices=["auto", "speeds", "diff"], default="auto")
    parser.add_argument("--object-track-timeout-ms", type=int, default=1000)

    parser.add_argument("--probe-topics", default=",".join((
        DEFAULT_START_TASK_TOPIC,
        DEFAULT_TASK_STATUS_TOPIC,
        DEFAULT_TEST_TRAJECTORY_TOPIC,
        DEFAULT_HD_ROUTE_TOPIC,
        DEFAULT_OBJECT_TRACK_TOPIC,
        DEFAULT_CONTROLCMD_TOPIC,
    )))
    parser.add_argument("--probe-count", type=int, default=1)

    parser.add_argument(
        "--require-actuator-feedback", dest="require_actuator_feedback", action="store_true", default=True
    )
    parser.add_argument(
        "--no-require-actuator-feedback", dest="require_actuator_feedback", action="store_false"
    )
    return parser


def _extract_role(argv: list[str]) -> str:
    for index, item in enumerate(argv):
        if item == "--role" and index + 1 < len(argv):
            return argv[index + 1]
        if item.startswith("--role="):
            return item.split("=", 1)[1]
    return "mux"


def parse_args(argv: list[str] | None = None) -> tuple[argparse.Namespace, argparse.Namespace | None]:
    argv = list(sys.argv[1:] if argv is None else argv)
    bridge_parser = build_bridge_parser()
    bridge_args, remaining = bridge_parser.parse_known_args(argv)

    if not bridge_args.node_name:
        bridge_args.node_name = {
            "mux": "student_control_mux",
            "task-manager": "student_task_manager",
            "adapter": "student_scene_adapter",
            "algorithm": "direct_control_node",
            "probe": "student_probe",
        }.get(bridge_args.role, "student_bridge")

    if bridge_args.role != "algorithm":
        return bridge_args, None

    from direct_control_node import build_arg_parser

    algorithm_args = build_arg_parser().parse_args(remaining)
    for name in (
        "node_name",
        "sensorgps_topic",
        "actuator_topic",
        "obu_topic",
        "controlcmd_topic",
        "max_speed_kph",
    ):
        setattr(algorithm_args, name, getattr(bridge_args, name))
    algorithm_args.controlcmd_topic = bridge_args.student_cmd_topic

    # 桥接参数也挂到算法参数上，供 TaskDrivenDirectControlNode 使用。
    for name in (
        "state_topic",
        "route_topic",
        "student_cmd_topic",
        "student_obstacles_topic",
        "require_actuator_feedback",
        "bridge_hz",
    ):
        setattr(algorithm_args, name, getattr(bridge_args, name))
    return bridge_args, algorithm_args


def main(argv: list[str] | None = None) -> None:
    bridge_args, algorithm_args = parse_args(argv)
    role = bridge_args.role

    if role == "mux":
        ControlMuxRole(bridge_args).run()
    elif role == "task-manager":
        TaskManagerRole(bridge_args).run()
    elif role == "adapter":
        SceneAdapterRole(bridge_args).run()
    elif role == "algorithm":
        assert algorithm_args is not None
        run_algorithm_role(algorithm_args)
    elif role == "probe":
        run_probe_role(bridge_args)
    else:  # pragma: no cover
        raise SystemExit(f"unknown role: {role}")


if __name__ == "__main__":
    main()
