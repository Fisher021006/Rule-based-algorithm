"""Cut-out 场景三车间距反解器。

用途：
- 输入主车速度、暴露车速度和目标 THW/TTC；
- 按 cut-out 场景的半程评估时刻、源车道释放安全、固定基础 IDM 跟驰约束反解合理间距范围；
- 可选把推荐间距映射到当前同济测试场默认路段坐标，便于直接仿真。

该脚本不运行场景算法，也不修改 cut_out_algorithm.py。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from global_planner.curve_planner.planner import CurvePlanner, TrajectoryGenerator
from global_planner.curve_planner.tools import project_and_cut_path
from traj_planner.tools import calculate_path_cumulative_distance, get_path_point_at_s, project_point_to_path
from xodr_parser.xodr_parser import XODRParser


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_MAP_CACHE = PROJECT_ROOT / "xodr_parser" / "cache" / "同济测试场.pkl"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "切出参数反解"

DEFAULT_ROUTE_START = (305.985199, 241.739304)
DEFAULT_ROUTE_GOAL = (158.323736, 251.743110)
DEFAULT_GVT_ANCHOR = (266.075005, 244.418185)
DEFAULT_GVT_GOAL = (211.846765, 244.355472)
IDM_MIN_GAP_M = 1.0
IDM_TIME_HEADWAY_S = 0.6
IDM_MAX_ACC_MPS2 = 2.5
IDM_COMFORT_DECEL_MPS2 = 3.5
IDM_DELTA = 4.0


def _parse_float_list(raw: str) -> list[float]:
    """解析逗号分隔浮点列表。"""

    return [float(item.strip()) for item in str(raw).split(",") if item.strip()]


def _fmt(value: float) -> str:
    """短格式化浮点数，便于打印和文件名使用。"""

    return f"{float(value):.3f}".rstrip("0").rstrip(".")


def _build_source_route(
    xodr_parser: XODRParser,
    route_start_xy: np.ndarray,
    route_goal_xy: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """构建源车道参考路线，用于把反解间距转换为地图坐标。"""

    planner = CurvePlanner(xodr_parser)
    lane_ids, actions = planner.plan_route(
        float(route_start_xy[0]),
        float(route_start_xy[1]),
        float(route_goal_xy[0]),
        float(route_goal_xy[1]),
    )
    if not lane_ids:
        raise RuntimeError("源车道参考路线规划失败")
    trajectory_generator = TrajectoryGenerator(planner)
    path_xy, _ = trajectory_generator.generate_full_trajectory(lane_ids, actions)
    path = project_and_cut_path(path_xy, route_start_xy, route_goal_xy)
    if path.ndim != 2 or len(path) < 2:
        raise RuntimeError("源车道参考路线点数不足")
    cumulative_s = calculate_path_cumulative_distance(path)
    return np.asarray(path, dtype=float), np.asarray(cumulative_s, dtype=float)


def _point_at_s(path: np.ndarray, cumulative_s: np.ndarray, s_value: float) -> np.ndarray:
    """按参考线 s 获取 XY 坐标。"""

    point, _ = get_path_point_at_s(path, float(s_value), cumulative_s)
    return np.asarray(point, dtype=float)


def _metric_type(exposed_speed: float, static_eps: float) -> str:
    """静止暴露车用 THW，低速暴露车用 TTC。"""

    return "THW" if float(exposed_speed) <= float(static_eps) else "TTC"


def _required_main_exposed_gap(
    *,
    main_speed: float,
    exposed_speed: float,
    hold_t: float,
    metric_eval_time_s: float,
    metric_name: str,
    target_thw_s: float,
    target_ttc_s: float,
    exposed_length_m: float,
) -> float:
    """反解初始 VUT->暴露车净距。

    这里的净距定义与仿真 debug 一致：两车中心线投影距离减去两车半长。
    静止目标的 THW 使用代码里的 effective_gap = net_gap + leader_length。
    """

    total_t = float(hold_t) + float(metric_eval_time_s)
    relative_speed = float(main_speed) - float(exposed_speed)
    if metric_name == "THW":
        return float(target_thw_s) * float(main_speed) - float(exposed_length_m) + relative_speed * total_t
    if relative_speed <= 1e-6:
        return float("nan")
    return relative_speed * (float(target_ttc_s) + total_t)


def _source_gap_min(
    *,
    gvt_speed: float,
    exposed_speed: float,
    hold_t: float,
    source_release_time_s: float,
    source_abort_gap_m: float,
    min_gvt_to_exposed_gap_m: float,
) -> float:
    """反解 GVT->暴露车初始净距下限。"""

    closing_speed = max(0.0, float(gvt_speed) - float(exposed_speed))
    entry_gap_min = float(min_gvt_to_exposed_gap_m) + closing_speed * float(hold_t)
    release_gap_min = float(source_abort_gap_m) + closing_speed * (float(hold_t) + float(source_release_time_s))
    return max(entry_gap_min, release_gap_min)


def _idm_required_gap(
    *,
    main_speed: float,
    gvt_speed: float,
    desired_speed: float,
    min_gap: float,
    time_headway: float,
    max_acc: float,
    comfort_decel: float,
    delta: float,
    allowed_decel: float,
) -> float:
    """反解 VUT-GVT 间距下限，使 IDM 初始减速度不超过 allowed_decel。"""

    desired_speed = max(float(desired_speed), 1e-3)
    max_acc = max(float(max_acc), 1e-3)
    comfort_decel = max(float(comfort_decel), 1e-3)
    allowed_decel = max(float(allowed_decel), 1e-3)
    main_speed = max(float(main_speed), 0.0)
    gvt_speed = max(float(gvt_speed), 0.0)

    free_term = 1.0 - (main_speed / desired_speed) ** max(float(delta), 1.0)
    delta_v = main_speed - gvt_speed
    braking_term = main_speed * delta_v / (2.0 * math.sqrt(max_acc * comfort_decel))
    desired_gap = float(min_gap) + max(0.0, main_speed * float(time_headway) + braking_term)

    # IDM: acc = a * (free_term - (s_star / gap)^2)
    # 要求 acc >= -allowed_decel，得到 gap >= s_star / sqrt(free_term + allowed/a)。
    denominator = free_term + allowed_decel / max_acc
    if denominator <= 1e-6:
        return float("inf")
    return desired_gap / math.sqrt(denominator)


def _main_gvt_gap_min(
    *,
    main_speed: float,
    gvt_speed: float,
    hold_t: float,
    min_vut_to_gvt_gap_m: float,
    idm_gap_min_m: float,
) -> float:
    """反解 VUT->GVT 初始净距下限。"""

    closing_speed = max(0.0, float(main_speed) - float(gvt_speed))
    order_gap_min = float(min_vut_to_gvt_gap_m) + closing_speed * float(hold_t)
    return max(order_gap_min, float(idm_gap_min_m))


def _build_sim_command(row: dict[str, Any], args: argparse.Namespace) -> str:
    """为推荐样本生成可直接复制运行的仿真命令。"""

    exposed_motion = "static" if float(row["exposed_speed_mps"]) <= float(args.static_speed_eps_mps) else "constant"
    cmd = [
        "python3",
        "simulate_scene_offline.py",
        "--map-cache",
        str(args.map_cache),
        "--planner-mode",
        "scene_target",
        "--scene-algorithm-symbol",
        "scenarios.cut_out_algorithm.cut_out_algorithm",
        "--ego-start-xy",
        _fmt(row["gvt_xy"][0]),
        _fmt(row["gvt_xy"][1]),
        "--ego-goal-xy",
        _fmt(args.gvt_goal_xy[0]),
        _fmt(args.gvt_goal_xy[1]),
        "--ego-start-speed",
        _fmt(row["gvt_speed_mps"]),
        "--ego-max-speed-kph",
        _fmt(args.ego_max_speed_kph),
        "--main-motion",
        "idm",
        "--main-start-xy",
        _fmt(row["main_xy"][0]),
        _fmt(row["main_xy"][1]),
        "--main-goal-xy",
        _fmt(args.route_goal_xy[0]),
        _fmt(args.route_goal_xy[1]),
        "--main-start-speed",
        _fmt(row["main_speed_mps"]),
        "--main-desired-speed",
        _fmt(row["main_speed_mps"]),
        "--enable-exposed-obstacle",
        "--exposed-motion",
        exposed_motion,
    ]
    if exposed_motion == "static":
        cmd.extend(["--exposed-static-xy", _fmt(row["exposed_xy"][0]), _fmt(row["exposed_xy"][1])])
    else:
        cmd.extend(
            [
                "--exposed-start-xy",
                _fmt(row["exposed_xy"][0]),
                _fmt(row["exposed_xy"][1]),
                "--exposed-goal-xy",
                _fmt(args.route_goal_xy[0]),
                _fmt(args.route_goal_xy[1]),
                "--exposed-start-speed",
                _fmt(row["exposed_speed_mps"]),
                "--exposed-desired-speed",
                _fmt(row["exposed_speed_mps"]),
            ]
        )
    cmd.extend(
        [
            "--max-steps",
            str(int(args.max_steps)),
            "--log-every",
            str(int(args.log_every)),
            "--output-dir",
            str(args.verify_output_dir),
            "--output-prefix",
            str(row["case_id"]),
        ]
    )
    return " ".join(cmd)


def _write_outputs(output_dir: Path, prefix: str, rows: list[dict[str, Any]]) -> None:
    """保存反解结果。"""

    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{prefix}_design.json"
    csv_path = output_dir / f"{prefix}_design.csv"
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    if rows:
        with csv_path.open("w", encoding="utf-8", newline="") as fp:
            writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    else:
        csv_path.write_text("", encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    """构造命令行参数。"""

    parser = argparse.ArgumentParser(description="按速度和指标反解 cut-out 三车间距")
    parser.add_argument("--main-speeds", default="15.0", help="主车速度列表 m/s")
    parser.add_argument("--exposed-speeds", default="0.0", help="暴露车速度列表 m/s；0 为静止车")
    parser.add_argument("--gvt-speeds", default="", help="GVT 切出前速度列表；空值表示跟随主车速度")
    parser.add_argument("--ego-max-speed-kph", type=float, default=60.0, help="生成仿真命令时使用的 GVT 最大速度 km/h")
    parser.add_argument("--hold-min-s", type=float, default=0.0, help="候选触发等待时间下限")
    parser.add_argument("--hold-max-s", type=float, default=8.0, help="候选触发等待时间上限")
    parser.add_argument("--hold-step-s", type=float, default=0.5, help="候选触发等待时间步长")
    parser.add_argument("--target-thw-s", type=float, default=2.0)
    parser.add_argument("--target-ttc-s", type=float, default=2.0)
    parser.add_argument("--metric-eval-time-s", type=float, default=2.0, help="ACTIVE 后多久评估指标，半换道默认 2s")
    parser.add_argument("--source-release-time-s", type=float, default=2.0, help="GVT 基本脱离源车道的时间估计")
    parser.add_argument("--source-abort-gap-m", type=float, default=2.0)
    parser.add_argument("--min-gvt-to-exposed-gap-m", type=float, default=5.0)
    parser.add_argument("--min-vut-to-gvt-gap-m", type=float, default=4.0)
    parser.add_argument("--vehicle-length-m", type=float, default=4.6)
    parser.add_argument("--static-speed-eps-mps", type=float, default=0.05)
    parser.add_argument("--allowed-initial-idm-decel-mps2", type=float, default=1.0)
    parser.add_argument("--gap-margin-m", type=float, default=1.0, help="推荐值相对下限增加的安全余量")
    parser.add_argument("--max-total-main-exposed-gap-m", type=float, default=160.0, help="过滤过长场景")
    parser.add_argument("--top", type=int, default=20, help="终端打印前 N 个推荐样本")
    parser.add_argument("--map-cache", type=Path, default=DEFAULT_MAP_CACHE)
    parser.add_argument("--route-start-xy", nargs=2, type=float, default=DEFAULT_ROUTE_START)
    parser.add_argument("--route-goal-xy", nargs=2, type=float, default=DEFAULT_ROUTE_GOAL)
    parser.add_argument("--gvt-anchor-xy", nargs=2, type=float, default=DEFAULT_GVT_ANCHOR)
    parser.add_argument("--gvt-goal-xy", nargs=2, type=float, default=DEFAULT_GVT_GOAL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-prefix", default="cutout_design")
    parser.add_argument("--verify-output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "切出参数反解验证")
    parser.add_argument("--max-steps", type=int, default=180)
    parser.add_argument("--log-every", type=int, default=20)
    return parser


def main() -> None:
    """脚本入口。"""

    parser = build_arg_parser()
    args = parser.parse_args()
    args.map_cache = Path(args.map_cache)
    args.output_dir = Path(args.output_dir)
    args.verify_output_dir = Path(args.verify_output_dir)
    args.route_start_xy = np.asarray(args.route_start_xy, dtype=float)
    args.route_goal_xy = np.asarray(args.route_goal_xy, dtype=float)
    args.gvt_anchor_xy = np.asarray(args.gvt_anchor_xy, dtype=float)
    args.gvt_goal_xy = np.asarray(args.gvt_goal_xy, dtype=float)

    main_speeds = _parse_float_list(args.main_speeds)
    exposed_speeds = _parse_float_list(args.exposed_speeds)
    configured_gvt_speeds = _parse_float_list(args.gvt_speeds) if str(args.gvt_speeds).strip() else []
    hold_times = np.arange(float(args.hold_min_s), float(args.hold_max_s) + 1e-9, float(args.hold_step_s))

    xodr_parser = XODRParser.load_cache(str(args.map_cache))
    source_path, source_cumulative_s = _build_source_route(xodr_parser, args.route_start_xy, args.route_goal_xy)
    _, gvt_s, _ = project_point_to_path(args.gvt_anchor_xy, source_path)
    gvt_xy = _point_at_s(source_path, source_cumulative_s, gvt_s)
    route_end_s = float(source_cumulative_s[-1])
    half_pair_length = float(args.vehicle_length_m)

    rows: list[dict[str, Any]] = []
    case_index = 0
    for main_speed in main_speeds:
        gvt_speeds = configured_gvt_speeds or [main_speed]
        for exposed_speed in exposed_speeds:
            metric_name = _metric_type(exposed_speed, args.static_speed_eps_mps)
            for gvt_speed in gvt_speeds:
                idm_gap_min = _idm_required_gap(
                    main_speed=main_speed,
                    gvt_speed=gvt_speed,
                    desired_speed=main_speed,
                    min_gap=IDM_MIN_GAP_M,
                    time_headway=IDM_TIME_HEADWAY_S,
                    max_acc=IDM_MAX_ACC_MPS2,
                    comfort_decel=IDM_COMFORT_DECEL_MPS2,
                    delta=IDM_DELTA,
                    allowed_decel=args.allowed_initial_idm_decel_mps2,
                )
                for hold_t in hold_times:
                    main_exposed_required = _required_main_exposed_gap(
                        main_speed=main_speed,
                        exposed_speed=exposed_speed,
                        hold_t=float(hold_t),
                        metric_eval_time_s=args.metric_eval_time_s,
                        metric_name=metric_name,
                        target_thw_s=args.target_thw_s,
                        target_ttc_s=args.target_ttc_s,
                        exposed_length_m=args.vehicle_length_m,
                    )
                    if not np.isfinite(main_exposed_required):
                        continue
                    if main_exposed_required > float(args.max_total_main_exposed_gap_m):
                        continue
                    gvt_exposed_min = _source_gap_min(
                        gvt_speed=gvt_speed,
                        exposed_speed=exposed_speed,
                        hold_t=float(hold_t),
                        source_release_time_s=args.source_release_time_s,
                        source_abort_gap_m=args.source_abort_gap_m,
                        min_gvt_to_exposed_gap_m=args.min_gvt_to_exposed_gap_m,
                    )
                    main_gvt_min = _main_gvt_gap_min(
                        main_speed=main_speed,
                        gvt_speed=gvt_speed,
                        hold_t=float(hold_t),
                        min_vut_to_gvt_gap_m=args.min_vut_to_gvt_gap_m,
                        idm_gap_min_m=idm_gap_min,
                    )

                    main_gvt_max = main_exposed_required - gvt_exposed_min - float(args.vehicle_length_m)
                    gvt_exposed_max = main_exposed_required - main_gvt_min - float(args.vehicle_length_m)
                    feasible = bool(main_gvt_max >= main_gvt_min and gvt_exposed_max >= gvt_exposed_min)
                    if not feasible:
                        continue

                    recommended_main_gvt = min(
                        main_gvt_max,
                        main_gvt_min + float(args.gap_margin_m),
                    )
                    recommended_gvt_exposed = main_exposed_required - recommended_main_gvt - float(args.vehicle_length_m)
                    if recommended_gvt_exposed < gvt_exposed_min:
                        recommended_gvt_exposed = gvt_exposed_min
                        recommended_main_gvt = main_exposed_required - recommended_gvt_exposed - float(args.vehicle_length_m)

                    main_s = float(gvt_s) - recommended_main_gvt - half_pair_length
                    exposed_s = float(gvt_s) + recommended_gvt_exposed + half_pair_length
                    map_fit = bool(main_s >= 0.0 and exposed_s <= route_end_s)
                    main_xy = _point_at_s(source_path, source_cumulative_s, main_s) if map_fit else np.array([float("nan"), float("nan")])
                    exposed_xy = _point_at_s(source_path, source_cumulative_s, exposed_s) if map_fit else np.array([float("nan"), float("nan")])

                    row: dict[str, Any] = {
                        "case_id": f"{args.output_prefix}_{case_index:03d}",
                        "metric_name": metric_name,
                        "main_speed_mps": round(float(main_speed), 3),
                        "gvt_speed_mps": round(float(gvt_speed), 3),
                        "exposed_speed_mps": round(float(exposed_speed), 3),
                        "hold_t_s": round(float(hold_t), 3),
                        "metric_eval_time_s": round(float(args.metric_eval_time_s), 3),
                        "target_metric_s": round(float(args.target_thw_s if metric_name == "THW" else args.target_ttc_s), 3),
                        "main_exposed_required_gap_m": round(float(main_exposed_required), 3),
                        "main_gvt_gap_min_m": round(float(main_gvt_min), 3),
                        "main_gvt_gap_max_m": round(float(main_gvt_max), 3),
                        "gvt_exposed_gap_min_m": round(float(gvt_exposed_min), 3),
                        "gvt_exposed_gap_max_m": round(float(gvt_exposed_max), 3),
                        "recommended_main_gvt_gap_m": round(float(recommended_main_gvt), 3),
                        "recommended_gvt_exposed_gap_m": round(float(recommended_gvt_exposed), 3),
                        "idm_initial_gap_min_m": round(float(idm_gap_min), 3),
                        "idm_time_headway_s": round(float(IDM_TIME_HEADWAY_S), 3),
                        "idm_min_gap_m": round(float(IDM_MIN_GAP_M), 3),
                        "allowed_initial_idm_decel_mps2": round(float(args.allowed_initial_idm_decel_mps2), 3),
                        "map_fit": map_fit,
                        "main_s": round(float(main_s), 3),
                        "gvt_s": round(float(gvt_s), 3),
                        "exposed_s": round(float(exposed_s), 3),
                        "main_xy": [round(float(main_xy[0]), 6), round(float(main_xy[1]), 6)],
                        "gvt_xy": [round(float(gvt_xy[0]), 6), round(float(gvt_xy[1]), 6)],
                        "exposed_xy": [round(float(exposed_xy[0]), 6), round(float(exposed_xy[1]), 6)],
                    }
                    row["simulate_command"] = _build_sim_command(row, args) if map_fit else ""
                    rows.append(row)
                    case_index += 1

    rows.sort(
        key=lambda item: (
            not bool(item["map_fit"]),
            abs(float(item["hold_t_s"]) - 2.0),
            float(item["main_exposed_required_gap_m"]),
        )
    )
    _write_outputs(args.output_dir, str(args.output_prefix), rows)

    print(f"[design] feasible_rows={len(rows)} output_dir={args.output_dir}", flush=True)
    for row in rows[: max(0, int(args.top))]:
        print(
            "[case] "
            f"{row['case_id']} metric={row['metric_name']} "
            f"vm={row['main_speed_mps']} vg={row['gvt_speed_mps']} ve={row['exposed_speed_mps']} "
            f"hold={row['hold_t_s']} "
            f"VUT-GVT={row['recommended_main_gvt_gap_m']}m "
            f"GVT-exposed={row['recommended_gvt_exposed_gap_m']}m "
            f"range_mg=[{row['main_gvt_gap_min_m']},{row['main_gvt_gap_max_m']}] "
            f"range_ge=[{row['gvt_exposed_gap_min_m']},{row['gvt_exposed_gap_max_m']}] "
            f"map_fit={row['map_fit']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
