#!/usr/bin/env python3
"""验证控制量驱动虚拟车辆模型。

验证目标：
1. 从离线仿真输出 JSON 读取原始 `ego_trace`。
2. 把连续轨迹窗口转换成当前项目的 `/controlcmd` 控制指令格式。
3. 用 `VirtualVehicle` 的理想自行车模型前向积分。
4. 统计控制量积分轨迹与原始 `ego_trace` 的误差。

注意：只靠 `/controlcmd` 的速度和前轮角无法唯一还原 x/y 轨迹，因此本验证
用于量化第一版控制量闭环模型相对原始规划轨迹的误差，而不是要求零误差。
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "platform_integration"))

from redis_ros_adapter.core.types import ControlCommandState  # noqa: E402
from redis_ros_adapter.core.virtual_vehicle import VirtualVehicle, VirtualVehicleConfig  # noqa: E402


class IdentityProjector:
    """验证脚本使用的最小投影器。

    本脚本只验证投影坐标下的车辆模型，不需要真实经纬度转换。
    """

    def latlon_to_xy(self, latitude: float, longitude: float) -> tuple[float, float]:
        return float(longitude), float(latitude)

    def xy_to_latlon(self, x: float, y: float) -> tuple[float, float]:
        return float(y), float(x)

    def nearest_lane_heading(self, x: float, y: float) -> tuple[None, float]:
        return None, float("inf")


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _wrap_angle_rad(value: float) -> float:
    return (value + math.pi) % (2.0 * math.pi) - math.pi


def trajectory_to_control_command(
    trajectory: Sequence[Sequence[float]],
    *,
    wheelbase_m: float,
    max_speed_kph: float,
    segment_dt_s: float,
) -> ControlCommandState:
    """复刻 `direct_control_node.py` 里的轨迹转控制量规则。"""

    if len(trajectory) < 2:
        return ControlCommandState(timestamp_ms=0, target_speed_kph=0.0, wheel_angle_deg=0.0, gear=4, turnlight=0, brkreq=1)
    current = trajectory[0]
    base_dt = max(float(segment_dt_s), 1e-3)
    max_segment_index = min(5, len(trajectory) - 1)
    target_index = max_segment_index
    min_command_speed_mps = 0.1 / 3.6
    for candidate_index in range(1, max_segment_index + 1):
        candidate = trajectory[candidate_index]
        candidate_dt = candidate_index * base_dt
        candidate_ds = math.hypot(float(candidate[0]) - float(current[0]), float(candidate[1]) - float(current[1]))
        candidate_speed = candidate_ds / max(candidate_dt, 1e-3)
        if candidate_speed >= min_command_speed_mps:
            target_index = candidate_index
            break

    next_point = trajectory[target_index]
    dt = max(float(target_index) * base_dt, 1e-3)
    dx = float(next_point[0]) - float(current[0])
    dy = float(next_point[1]) - float(current[1])
    ds = math.hypot(dx, dy)
    heading_delta = _wrap_angle_rad(float(next_point[2]) - float(current[2]))
    speed_kph = _clamp(max(ds / dt, 0.0) * 3.6, 0.0, float(max_speed_kph))
    kappa_cmd = heading_delta / max(ds, 1e-3)
    wheel_angle_rad = math.atan(float(wheelbase_m) * float(kappa_cmd))
    wheel_angle_deg = _clamp(math.degrees(wheel_angle_rad), -26.0, 26.0)
    return ControlCommandState(
        timestamp_ms=0,
        target_speed_kph=float(speed_kph),
        wheel_angle_deg=float(wheel_angle_deg),
        gear=4,
        turnlight=0,
        brkreq=1 if speed_kph < 0.1 else 0,
    )


def load_ego_trace(path: Path) -> list[list[float]]:
    """读取离线仿真 JSON 里的 `ego_trace` 并校验点格式。"""

    data = json.loads(path.read_text(encoding="utf-8"))
    trace = data.get("ego_trace")
    if not isinstance(trace, list) or len(trace) < 2:
        raise ValueError(f"missing ego_trace with at least 2 points: {path}")

    normalized: list[list[float]] = []
    for index, point in enumerate(trace):
        if not isinstance(point, list) or len(point) < 6:
            raise ValueError(f"ego_trace[{index}] must be [x,y,heading,v,a,kappa]")
        normalized.append([float(point[i]) for i in range(6)])
    return normalized


def verify_trace(args: argparse.Namespace) -> dict[str, float | int]:
    """执行一致性验证并返回误差统计。"""

    trace = load_ego_trace(Path(args.json))
    vehicle = VirtualVehicle(
        projector=IdentityProjector(),  # type: ignore[arg-type]
        config=VirtualVehicleConfig(
            wheelbase_m=float(args.wheelbase_m),
            max_speed_kph=float(args.max_speed_kph),
            trajectory_advance_index=int(args.advance_index),
        ),
    )

    first = trace[0]
    vehicle.x = first[0]
    vehicle.y = first[1]
    vehicle.heading = first[2]
    vehicle.v = first[3]
    vehicle.a = first[4]
    vehicle.kappa = first[5]

    max_pos_err = 0.0
    max_heading_err = 0.0
    max_speed_err = 0.0
    max_acc_err = 0.0
    max_kappa_err = 0.0
    sum_pos_err_sq = 0.0
    sum_heading_err_sq = 0.0
    sum_speed_err_sq = 0.0
    checked_steps = 0

    window_size = max(int(args.window_size), int(args.advance_index) + 1, 2)
    for step_index in range(len(trace) - 1):
        window = trace[step_index : min(len(trace), step_index + window_size)]
        if len(window) < 2:
            break

        command = trajectory_to_control_command(
            window,
            wheelbase_m=float(args.wheelbase_m),
            max_speed_kph=float(args.max_speed_kph),
            segment_dt_s=float(args.dt),
        )
        vehicle.apply_control_command(command)
        vehicle.step(float(args.dt))

        expected_index = min(max(0, int(args.advance_index)), len(window) - 1)
        expected = window[expected_index]
        pos_err = math.hypot(vehicle.x - expected[0], vehicle.y - expected[1])
        heading_err = abs(_wrap_angle_rad(vehicle.heading - expected[2]))
        speed_err = abs(vehicle.v - expected[3])
        acc_err = abs(vehicle.a - expected[4])
        kappa_err = abs(vehicle.kappa - expected[5])

        max_pos_err = max(max_pos_err, pos_err)
        max_heading_err = max(max_heading_err, heading_err)
        max_speed_err = max(max_speed_err, speed_err)
        max_acc_err = max(max_acc_err, acc_err)
        max_kappa_err = max(max_kappa_err, kappa_err)
        sum_pos_err_sq += pos_err * pos_err
        sum_heading_err_sq += heading_err * heading_err
        sum_speed_err_sq += speed_err * speed_err
        checked_steps += 1

    denom = max(1, checked_steps)
    return {
        "checked_steps": checked_steps,
        "max_position_error_m": max_pos_err,
        "max_heading_error_rad": max_heading_err,
        "max_speed_error_mps": max_speed_err,
        "max_acc_error_mps2": max_acc_err,
        "max_kappa_error": max_kappa_err,
        "rms_position_error_m": math.sqrt(sum_pos_err_sq / denom),
        "rms_heading_error_rad": math.sqrt(sum_heading_err_sq / denom),
        "rms_speed_error_mps": math.sqrt(sum_speed_err_sq / denom),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="验证控制量驱动虚拟车辆模型")
    parser.add_argument(
        "--json",
        default=str(PROJECT_ROOT / "outputs" / "scene_offline" / "case_local_sampling_static_-188_156.json"),
        help="离线仿真输出 JSON，需包含 ego_trace",
    )
    parser.add_argument("--advance-index", type=int, default=1, help="执行轨迹中的第几个未来点")
    parser.add_argument("--window-size", type=int, default=8, help="每次用于转控制量和模型执行的轨迹窗口长度")
    parser.add_argument("--dt", type=float, default=0.1, help="验证步长，仅用于兼容 step(dt) 接口")
    parser.add_argument("--wheelbase-m", type=float, default=2.7)
    parser.add_argument("--max-speed-kph", type=float, default=25.0)
    parser.add_argument("--tolerance", type=float, default=0.0, help="大于 0 时按最大位置误差做阈值检查")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    summary = verify_trace(args)
    print(f"[verify] input_json={args.json}")
    print(
        "[verify] checked_steps={checked_steps} advance_index={advance} window_size={window}".format(
            checked_steps=summary["checked_steps"],
            advance=int(args.advance_index),
            window=int(args.window_size),
        )
    )
    for key in (
        "max_position_error_m",
        "max_heading_error_rad",
        "max_speed_error_mps",
        "max_acc_error_mps2",
        "max_kappa_error",
        "rms_position_error_m",
        "rms_heading_error_rad",
        "rms_speed_error_mps",
    ):
        print(f"[verify] {key}={float(summary[key]):.12g}")

    if float(args.tolerance) > 0.0 and float(summary["max_position_error_m"]) > float(args.tolerance):
        print(
            "[verify] FAIL max_position_error_m={error:.12g} tolerance={tolerance:.12g}".format(
                error=float(summary["max_position_error_m"]),
                tolerance=float(args.tolerance),
            )
        )
        return 1
    print("[verify] DONE controlcmd-driven model error computed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
