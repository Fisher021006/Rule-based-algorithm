#!/usr/bin/env python3
"""本地 smoke 测试用的 mock 平台 Redis 发布器。

这个脚本不连接真实平台，只向本地 Redis 发布一组模拟的平台 A-B 任务：
先发 type=1 准备任务，再发 type=2 开始任务，可选发布背景车状态，
最后等待 Adapter 在上传通道回传本车轨迹。它用于验证 Adapter、ROS 算法、
虚拟车辆和 Redis 回传链路是否能闭环。
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np


PLATFORM_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PLATFORM_ROOT.parent
for path in (PROJECT_ROOT, PLATFORM_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from redis_ros_adapter.core.coordinates import (
    DEFAULT_MAP_PATH,
    MapProjector,
    internal_rad_to_platform_heading_deg,
    load_or_build_map,
)


def now_ms() -> int:
    """返回当前 Unix 时间戳，单位毫秒。"""

    return int(time.time() * 1000)


def dumps(payload: dict[str, Any]) -> str:
    """把 mock 平台消息序列化成 Redis 发送用 JSON 字符串。"""

    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def heading_from_nearest_lane(xodr_parser: Any, xy: tuple[float, float]) -> float:
    """根据地图最近车道中心线计算平台任务起点航向角。"""

    best_line = None
    best_dist = float("inf")
    point = np.asarray(xy, dtype=float)
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


def build_prepare_message(args: argparse.Namespace) -> dict[str, Any]:
    """构造平台 type=1 准备任务消息。

    输入的起终点默认是 XODR 平面坐标，函数内部会转换成平台协议使用的经纬度。
    """

    xodr_parser = load_or_build_map(args.map_path, args.cache_path, args.rebuild_cache)
    projector = MapProjector(xodr_parser)
    start_x, start_y = (float(args.start_xy[0]), float(args.start_xy[1]))
    goal_x, goal_y = (float(args.goal_xy[0]), float(args.goal_xy[1]))
    start_lat, start_lon = projector.xy_to_latlon(start_x, start_y)
    goal_lat, goal_lon = projector.xy_to_latlon(goal_x, goal_y)
    if args.course_angle_deg is None:
        course_angle = internal_rad_to_platform_heading_deg(heading_from_nearest_lane(xodr_parser, (start_x, start_y)))
    else:
        course_angle = float(args.course_angle_deg)

    return {
        "params": {
            "id": args.case_id,
            "type": 1,
            "participantTrajectories": [
                {
                    "value": [
                        {
                            "courseAngle": course_angle,
                            "latitude": start_lat,
                            "longitude": start_lon,
                            "siteType": 0,
                            "speed": 0,
                        }
                    ]
                },
                {
                    "value": [
                        {
                            "courseAngle": course_angle,
                            "latitude": goal_lat,
                            "longitude": goal_lon,
                            "siteType": 1,
                            "speed": 0,
                        }
                    ]
                },
            ],
        },
        "timestamp": now_ms(),
        "type": 1,
    }


def build_task_control_message(args: argparse.Namespace, task_type: int) -> dict[str, Any]:
    """构造平台 type=2 开始或停止任务消息。

    `task_type=1` 表示开始，`task_type=0` 表示结束；消息里同时携带背景车
    通道、上传通道和 mergeDataChannel，模拟真实平台对域控的通道分配。
    """

    return {
        "type": 2,
        "params": {
            "taskType": int(task_type),
            "protocols": [
                {
                    "type": 0,
                    "channel": args.background_channel,
                    "params": {
                        "monitorChannel": f"{args.background_channel}_monitor",
                    },
                },
                {
                    "type": 1,
                    "channel": args.upload_channel,
                    "params": {
                        "recordId": args.record_id,
                        "monitorChannel": f"{args.upload_channel}_monitor",
                        "caseId": args.case_id,
                        "taskId": args.task_id,
                        "mergeDataChannel": args.merge_channel,
                    },
                },
            ],
        },
        "timestamp": now_ms(),
        "controlChannel": args.control_channel,
    }


def build_background_message(args: argparse.Namespace) -> dict[str, Any]:
    """构造一帧 mock 背景车 trajectory 消息。"""

    xodr_parser = load_or_build_map(args.map_path, args.cache_path, False)
    projector = MapProjector(xodr_parser)
    x, y = (float(args.background_xy[0]), float(args.background_xy[1]))
    lat, lon = projector.xy_to_latlon(x, y)
    course_angle = internal_rad_to_platform_heading_deg(heading_from_nearest_lane(xodr_parser, (x, y)))
    return {
        "type": "trajectory",
        "value": {
            "timestampType": "CREATE_TIME",
            "timestamp": str(now_ms()),
            "value": [
                {
                    "timestamp": str(now_ms()),
                    "globalTimeStamp": str(now_ms()),
                    "frameId": 1,
                    "id": str(args.background_vehicle_id),
                    "name": "mock_background",
                    "longitude": lon,
                    "latitude": lat,
                    "courseAngle": course_angle,
                    "speed": float(args.background_speed_kph),
                    "lonAcc": 0.0,
                    "latAcc": 0.0,
                    "vehicleType": 1,
                    "length": 464,
                    "width": 178,
                    "height": 150,
                    "driveType": 0,
                    "isPerception": 1,
                }
            ],
        },
    }


def build_arg_parser() -> argparse.ArgumentParser:
    """创建 mock 平台脚本的命令行参数解析器。"""

    parser = argparse.ArgumentParser(description="通过 Redis 模拟平台 A-B 任务下发")
    parser.add_argument("--redis-host", default="host.docker.internal")
    parser.add_argument("--redis-port", type=int, default=6379)
    parser.add_argument("--redis-db", type=int, default=0)
    parser.add_argument("--redis-password", default="")
    parser.add_argument("--control-channel", default="control_mock_channel")
    parser.add_argument("--status-channel", default="STATUSResult")
    parser.add_argument("--background-channel", default="mock_background_channel")
    parser.add_argument("--upload-channel", default="mock_upload_channel")
    parser.add_argument("--merge-channel", default="mock_merge_channel")
    parser.add_argument("--task-id", default="mock-task-001")
    parser.add_argument("--case-id", default="mock-case-001")
    parser.add_argument("--record-id", default="mock-record-001")
    parser.add_argument("--map-path", default=str(DEFAULT_MAP_PATH))
    parser.add_argument("--cache-path", default="")
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--start-xy", nargs=2, type=float, default=(-418.0, 59.0), metavar=("X", "Y"))
    parser.add_argument("--goal-xy", nargs=2, type=float, default=(-429.0, 129.0), metavar=("X", "Y"))
    parser.add_argument("--course-angle-deg", type=float, default=None)
    parser.add_argument("--publish-background", action="store_true")
    parser.add_argument("--background-vehicle-id", default="101")
    parser.add_argument("--background-xy", nargs=2, type=float, default=(-423.0, 90.0), metavar=("X", "Y"))
    parser.add_argument("--background-speed-kph", type=float, default=10.0)
    parser.add_argument("--timeout-s", type=float, default=45.0)
    parser.add_argument("--min-upload-speed-kph", type=float, default=0.1)
    parser.add_argument("--keep-running", action="store_true", help="收到第一帧回传后不自动发送任务结束")
    return parser


def main() -> None:
    """脚本入口：发布 mock 平台任务，并等待 Adapter 回传轨迹。"""

    args = build_arg_parser().parse_args()
    import redis

    client = redis.Redis(
        host=args.redis_host,
        port=args.redis_port,
        db=args.redis_db,
        password=args.redis_password or None,
        decode_responses=True,
    )
    client.ping()
    pubsub = client.pubsub(ignore_subscribe_messages=True)
    pubsub.subscribe(args.status_channel, args.upload_channel, args.merge_channel)

    print(f"[mock] publish prepare -> {args.control_channel}", flush=True)
    client.publish(args.control_channel, dumps(build_prepare_message(args)))
    time.sleep(1.0)
    print(f"[mock] publish task start -> {args.control_channel}", flush=True)
    client.publish(args.control_channel, dumps(build_task_control_message(args, task_type=1)))
    if args.publish_background:
        time.sleep(0.5)
        background_payload = dumps(build_background_message(args))
        for _ in range(5):
            print(f"[mock] publish background -> {args.background_channel}: {background_payload[:300]}", flush=True)
            client.publish(args.background_channel, background_payload)
            time.sleep(0.1)

    deadline = time.time() + float(args.timeout_s)
    got_upload = False
    got_status = False
    try:
        while time.time() < deadline:
            message = pubsub.get_message(timeout=1.0)
            if message is None or message.get("type") != "message":
                continue
            channel = str(message.get("channel", ""))
            data = str(message.get("data", ""))
            print(f"[mock] recv {channel}: {data[:500]}", flush=True)
            if channel == args.status_channel:
                got_status = True
            if channel == args.upload_channel:
                try:
                    payload = json.loads(data)
                    values = payload.get("value", {}).get("value", [])
                    speed = float(values[0].get("speed", 0.0)) if values else 0.0
                except Exception:
                    speed = 0.0
                if speed >= float(args.min_upload_speed_kph):
                    got_upload = True
                    break
        if not got_status:
            raise SystemExit("[mock] did not receive status heartbeat")
        if not got_upload:
            raise SystemExit("[mock] did not receive upload trajectory")
    finally:
        if not args.keep_running:
            print(f"[mock] publish task end -> {args.control_channel}", flush=True)
            client.publish(args.control_channel, dumps(build_task_control_message(args, task_type=0)))
        pubsub.close()

    print("[mock] PASS", flush=True)


if __name__ == "__main__":
    main()
