"""平台 Redis JSON 协议解析与回传封装。

平台和本地域控 Adapter 之间通过 Redis 传 JSON 字符串。本文件集中处理
协议字段，包括 type=1 准备任务、type=2 开始/停止任务、背景车轨迹提取、
设备状态心跳、本车轨迹回传和 mergeDataChannel 回传。其它模块不要直接
拼平台 JSON，统一走这里可以降低字段不一致带来的问题。
"""

from __future__ import annotations

import json
import time
from typing import Any

from redis_ros_adapter.core.types import (
    PreparedTask,
    ProtocolChannels,
    TaskControl,
    TaskPoint,
    VehicleReportState,
)


def now_ms() -> int:
    """返回当前 Unix 时间戳，单位毫秒。"""

    return int(time.time() * 1000)


def loads_json_message(data: Any) -> dict[str, Any]:
    """把 Redis 收到的消息载荷解析成字典。

    Redis 客户端可能返回 `bytes`、`str`，测试代码也可能直接传入 `dict`；
    本函数把这些输入统一成平台协议字典。
    """

    if isinstance(data, bytes):
        data = data.decode("utf-8")
    if isinstance(data, str):
        return json.loads(data)
    if isinstance(data, dict):
        return data
    raise TypeError(f"unsupported redis message payload type: {type(data)!r}")


def dumps_json_message(payload: dict[str, Any]) -> str:
    """把平台回传字典压缩成 JSON 字符串。

    `ensure_ascii=False` 保留中文，`separators` 去掉多余空格，便于 Redis
    通道中传输和日志查看。
    """

    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def _as_float(value: Any, default: float = 0.0) -> float:
    """把协议字段安全转换为浮点数，失败时返回默认值。"""

    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_int(value: Any, default: int = 0) -> int:
    """把协议字段安全转换为整数，失败时返回默认值。"""

    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _point_from_payload(payload: dict[str, Any]) -> TaskPoint | None:
    """从平台轨迹点字段中提取经纬度、航向角、速度和点类型。

    平台不同版本可能使用 `longitude/lng/lon` 或 `latitude/lat`，这里做兼容；
    如果缺少经纬度，说明该点不能作为算法输入，直接返回 `None`。
    """

    lat = payload.get("latitude", payload.get("lat"))
    lon = payload.get("longitude", payload.get("lng", payload.get("lon")))
    if lat is None or lon is None:
        return None
    return TaskPoint(
        latitude=_as_float(lat),
        longitude=_as_float(lon),
        course_angle=_as_float(payload.get("courseAngle", payload.get("heading", 0.0))),
        speed_kph=_as_float(payload.get("speed", 0.0)),
        site_type=_as_int(payload.get("siteType", -1), -1),
        raw=dict(payload),
    )


def parse_prepare_message(payload: dict[str, Any]) -> PreparedTask | None:
    """解析平台 type=1 准备任务消息。

    当前 A-B 自由行驶只使用 `siteType=0` 的起点和 `siteType=1` 的终点；
    `siteType=2` 暂存为途经点，后续如果算法支持多途经点可以继续复用。
    非 type=1 消息返回 `None`，由上层继续判断是否是其它协议消息。
    """

    if _as_int(payload.get("type"), -1) != 1:
        return None

    params = payload.get("params") or {}
    trajectories = params.get("participantTrajectories") or payload.get("participantTrajectories") or []
    points: list[TaskPoint] = []
    for trajectory in trajectories:
        for value in trajectory.get("value", []) or []:
            point = _point_from_payload(value)
            if point is not None:
                points.append(point)

    start = next((point for point in points if int(point.site_type) == 0), None)
    goal = next((point for point in points if int(point.site_type) == 1), None)
    waypoints = [point for point in points if int(point.site_type) == 2]

    return PreparedTask(
        task_id=str(params.get("taskId", payload.get("taskId", "")) or ""),
        case_id=str(params.get("caseId", payload.get("caseId", params.get("id", ""))) or ""),
        record_id=str(params.get("recordId", payload.get("recordId", "")) or ""),
        start=start,
        goal=goal,
        waypoints=waypoints,
        raw=dict(payload),
    )


def parse_task_control_message(payload: dict[str, Any]) -> TaskControl | None:
    """解析平台 type=2 任务开始/停止控制消息。

    该消息不仅告诉 Adapter 是否开始任务，还会给出背景车通道、上传通道、
    监控通道和 mergeDataChannel。没有这些通道，算法即使跑起来也不知道
    应该往哪里回传平台可视化数据。
    """

    if _as_int(payload.get("type"), -1) != 2:
        return None

    params = payload.get("params") or {}
    channels = ProtocolChannels()
    protocols = params.get("protocols") or []
    for protocol in protocols:
        if not isinstance(protocol, dict):
            continue
        channels.raw_protocols.append(dict(protocol))
        protocol_type = _as_int(protocol.get("type"), -1)
        channel = str(protocol.get("channel", "") or "")
        protocol_params = protocol.get("params") or {}
        if protocol_type == 0 and channel:
            channels.background_channels.append(channel)
        elif protocol_type == 1:
            channels.upload_channel = channel
            channels.monitor_channel = str(protocol_params.get("monitorChannel", "") or "")
            channels.merge_data_channel = str(protocol_params.get("mergeDataChannel", "") or "")
            channels.task_id = str(protocol_params.get("taskId", "") or "")
            channels.case_id = str(protocol_params.get("caseId", "") or "")
            channels.record_id = str(protocol_params.get("recordId", "") or "")

    return TaskControl(
        task_type=_as_int(params.get("taskType"), -1),
        channels=channels,
        raw=dict(payload),
    )


def _looks_like_vehicle_value(value: Any) -> bool:
    """粗略判断一个字典是否像平台车辆状态记录。"""

    if not isinstance(value, dict):
        return False
    has_lon = any(key in value for key in ("longitude", "lon", "lng", "pos_lon"))
    has_lat = any(key in value for key in ("latitude", "lat", "pos_lat"))
    return bool(has_lon and has_lat)


def _append_vehicle_values(values: list[dict[str, Any]], candidate: Any) -> None:
    """把候选字段里的车辆记录追加到列表中。

    平台背景车消息可能直接是单车字典，也可能是车辆数组；该函数负责把两种
    形式统一展开。
    """

    if isinstance(candidate, list):
        for item in candidate:
            if _looks_like_vehicle_value(item):
                values.append(dict(item))
    elif _looks_like_vehicle_value(candidate):
        values.append(dict(candidate))


def extract_vehicle_values(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """从平台背景车或融合数据消息中提取车辆状态列表。

    当前场景算法需要把平台背景车和静态障碍物转成 ROS OBU 规划输入。平台消息
    可能嵌套在 `value.value`、`value.obstacles`、顶层 `value`、顶层
    `obstacles` 或 `participantTrajectories[].value` 中，因此这里做多种
    结构的兼容提取。
    """

    values: list[dict[str, Any]] = []

    body = payload.get("value")
    if isinstance(body, dict):
        _append_vehicle_values(values, body.get("value"))
        _append_vehicle_values(values, body.get("obstacles"))
    else:
        _append_vehicle_values(values, body)

    _append_vehicle_values(values, payload.get("value"))
    _append_vehicle_values(values, payload.get("obstacles"))

    for trajectory in payload.get("participantTrajectories", []) or []:
        if not isinstance(trajectory, dict):
            continue
        _append_vehicle_values(values, trajectory.get("value"))

    return values


def build_status_message(device_id: int, status_type: int, state: int, timestamp_ms: int | None = None) -> dict[str, Any]:
    """构造设备状态/心跳回传消息。

    `status_type=0` 通常表示在线心跳，`status_type=1` 表示任务准备状态；
    `state=1` 表示可用或在线，`state=2` 表示异常或未准备好。
    """

    return {
        "timestamp": now_ms() if timestamp_ms is None else int(timestamp_ms),
        "deviceId": int(device_id),
        "type": int(status_type),
        "state": int(state),
    }


def build_vehicle_value(state: VehicleReportState, frame_id: int) -> dict[str, Any]:
    """把内部车辆状态封装成平台 trajectory 里的单车 value 字段。"""

    timestamp_ms = int(state.timestamp_ms)
    return {
        "timestamp": str(timestamp_ms),
        "globalTimeStamp": str(timestamp_ms),
        "frameId": int(frame_id),
        "id": str(state.vehicle_id),
        "name": str(state.name),
        "longitude": float(state.longitude),
        "latitude": float(state.latitude),
        "courseAngle": float(state.course_angle),
        "speed": float(state.speed_kph),
        "lonAcc": float(state.lon_acc),
        "latAcc": float(state.lat_acc),
        "vehicleType": 1,
        "length": int(state.length_cm),
        "width": int(state.width_cm),
        "height": int(state.height_cm),
        "steeringWheelAngle": float(state.wheel_angle_deg),
        "driveType": 1,
        "acceleratorPedal": 0.0,
        "braking": 100.0 if int(state.command.brkreq) else 0.0,
        "autoStatus": 1,
        "indicatorStatus": int(state.command.turnlight),
        "blinkerStatus": 0,
        "isPerception": 0,
        "gear": int(state.command.gear),
        "wheelAngle": float(state.wheel_angle_deg),
        "acceleration": float(state.acceleration_mps2),
        "obuStatus": 1,
        "locationStatus": 1,
        "chassisStatus": 1,
    }


def _read_vehicle_id(payload: dict[str, Any]) -> str:
    """从平台车辆字典中兼容读取车辆 ID。"""

    for key in ("id", "ptc_id", "ptcid", "trackId", "track_id", "sourceId", "source_id"):
        value = payload.get(key)
        if value not in (None, ""):
            return str(value)
    return ""


def _read_first_float(payload: dict[str, Any], keys: tuple[str, ...], default: float = 0.0) -> float:
    """按多个候选字段名读取浮点值。"""

    for key in keys:
        value = payload.get(key)
        if value not in (None, ""):
            return _as_float(value, default)
    return float(default)


def _read_size_cm(payload: dict[str, Any], key: str, default_cm: int) -> int:
    """读取平台车辆尺寸并归一成厘米。

    协议说明车辆长宽高单位为 cm；平台历史样例中也出现过米制小数，因此这里
    用 `<=20` 的数值按米处理，其余按厘米原样处理。
    """

    value = payload.get(key)
    if value in (None, ""):
        return int(default_cm)
    numeric = max(0.0, _as_float(value, float(default_cm)))
    if numeric <= 20.0:
        numeric *= 100.0
    return int(round(numeric))


def build_background_vehicle_value(vehicle: dict[str, Any], frame_id: int, timestamp_ms: int) -> dict[str, Any] | None:
    """把平台下发的背景车状态封装成 mergeDataChannel 的单车 value。

    背景车是平台 type=0 通道实时下发的车辆状态。Adapter 转发给算法时会把它
    转成 `/obu/source`；回传平台融合显示时需要恢复成协议中的车辆 value。
    """

    vehicle_id = _read_vehicle_id(vehicle)
    lat = vehicle.get("latitude", vehicle.get("lat"))
    lon = vehicle.get("longitude", vehicle.get("lng", vehicle.get("lon", vehicle.get("pos_lon"))))
    if lat is None:
        lat = vehicle.get("pos_lat")
    if not vehicle_id or lat is None or lon is None:
        return None

    value = {
        "timestamp": str(timestamp_ms),
        "globalTimeStamp": str(timestamp_ms),
        "frameId": int(frame_id),
        "id": str(vehicle_id),
        "name": str(vehicle.get("name", "")),
        "longitude": _as_float(lon),
        "latitude": _as_float(lat),
        "courseAngle": _read_first_float(vehicle, ("courseAngle", "heading"), 0.0),
        "speed": _read_first_float(vehicle, ("speed",), 0.0),
        "lonAcc": _read_first_float(vehicle, ("lonAcc", "acc", "accel", "acceleration"), 0.0),
        "latAcc": _read_first_float(vehicle, ("latAcc",), 0.0),
        "vehicleType": _as_int(vehicle.get("vehicleType", vehicle.get("ptc_type", 1)), 1),
        "length": _read_size_cm(vehicle, "length", 450),
        "width": _read_size_cm(vehicle, "width", 180),
        "height": _read_size_cm(vehicle, "height", 150),
        "steeringWheelAngle": _read_first_float(vehicle, ("steeringWheelAngle",), 0.0),
        "driveType": _as_int(vehicle.get("driveType", 2), 2),
        "acceleratorPedal": _read_first_float(vehicle, ("acceleratorPedal",), 0.0),
        "braking": _read_first_float(vehicle, ("braking",), 0.0),
        "autoStatus": _as_int(vehicle.get("autoStatus", 1), 1),
        "indicatorStatus": _as_int(vehicle.get("indicatorStatus", vehicle.get("blinkerStatus", 0)), 0),
        "blinkerStatus": _as_int(vehicle.get("blinkerStatus", 0), 0),
        "isPerception": _as_int(vehicle.get("isPerception", 1), 1),
        "gear": _as_int(vehicle.get("gear", 0), 0),
        "wheelAngle": _read_first_float(vehicle, ("wheelAngle", "steeringWheelAngle"), 0.0),
        "acceleration": _read_first_float(vehicle, ("acceleration", "acc", "accel", "lonAcc"), 0.0),
        "obuStatus": _as_int(vehicle.get("obuStatus", 1), 1),
        "locationStatus": _as_int(vehicle.get("locationStatus", 1), 1),
        "chassisStatus": _as_int(vehicle.get("chassisStatus", 1), 1),
    }

    for key in ("originalColor", "vehicleColor", "picLicense"):
        if key in vehicle and vehicle[key] not in (None, ""):
            value[key] = vehicle[key]

    return value


def build_background_trajectory_entries(
    background_payloads: dict[str, dict[str, Any]] | None,
    frame_id: int,
    timestamp_ms: int,
    exclude_vehicle_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    """把最新背景车消息转换成 mergeDataChannel 的 `mvSimulation` 分组。"""

    entries: list[dict[str, Any]] = []
    exclude_ids = {str(item) for item in (exclude_vehicle_ids or set()) if str(item)}

    for source_channel, payload in (background_payloads or {}).items():
        values = []
        seen_ids = set()
        for vehicle in extract_vehicle_values(payload):
            vehicle_id = _read_vehicle_id(vehicle)
            if not vehicle_id or vehicle_id in exclude_ids or vehicle_id in seen_ids:
                continue
            value = build_background_vehicle_value(vehicle, frame_id, timestamp_ms)
            if value is None:
                continue
            values.append(value)
            seen_ids.add(vehicle_id)
        if values:
            entries.append(
                {
                    "role": "mvSimulation",
                    "source": str(source_channel or ""),
                    "timestamp": str(timestamp_ms),
                    "timestampType": "",
                    "value": values,
                }
            )

    return entries


def build_trajectory_report(state: VehicleReportState, frame_id: int, timestamp_ms: int | None = None) -> dict[str, Any]:
    """构造发往平台上传通道的本车轨迹消息。"""

    ts = now_ms() if timestamp_ms is None else int(timestamp_ms)
    return {
        "type": "trajectory",
        "value": {
            "timestampType": "CREATE_TIME",
            "timestamp": str(ts),
            "value": [build_vehicle_value(state, frame_id)],
        },
    }


def build_merge_report(
    state: VehicleReportState,
    frame_id: int,
    case_id: str,
    task_id: str,
    source_channel: str,
    timestamp_ms: int | None = None,
    background_payloads: dict[str, dict[str, Any]] | None = None,
    exclude_vehicle_ids: set[str] | None = None,
) -> dict[str, Any]:
    """构造发往 mergeDataChannel 的融合显示消息。

    平台有些场景会要求域控把本车状态写入 mergeDataChannel，便于和背景车、
    其它参与者统一显示。该格式保留 `caseId/taskId/source`，用于平台侧归档
    和通道溯源。
    """

    ts = now_ms() if timestamp_ms is None else int(timestamp_ms)
    participant_trajectories = [
        {
            "role": "av",
            "source": str(source_channel or ""),
            "timestamp": str(ts),
            "timestampType": "",
            "value": [build_vehicle_value(state, frame_id)],
        }
    ]
    participant_trajectories.extend(
        build_background_trajectory_entries(
            background_payloads=background_payloads,
            frame_id=frame_id,
            timestamp_ms=ts,
            exclude_vehicle_ids=exclude_vehicle_ids,
        )
    )
    return {
        "caseId": str(case_id or "0"),
        "taskId": str(task_id or "0"),
        "participantTrajectories": participant_trajectories,
    }
