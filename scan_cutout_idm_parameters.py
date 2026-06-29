"""Cut-out 场景 IDM 主车参数扫描器。

该脚本只负责离线测试参数组合，不修改 cut-out 场景算法本身：
- 按源车道参考线把 VUT/GVT/暴露车摆到指定纵向净距；
- 批量调用 simulate_scene_offline.py，主车使用 IDM；
- 汇总每组是否触发 ACTIVE、是否 cutout_success、最终指标和速度变化。
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

from global_planner.curve_planner.planner import CurvePlanner, TrajectoryGenerator
from global_planner.curve_planner.tools import project_and_cut_path
from traj_planner.tools import calculate_path_cumulative_distance, get_path_point_at_s, project_point_to_path
from xodr_parser.xodr_parser import XODRParser


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_MAP_CACHE = PROJECT_ROOT / "xodr_parser" / "cache" / "同济测试场.pkl"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "切出IDM参数扫描"

# 这组默认坐标来自当前已经验证过的同济测试场直线路段。
DEFAULT_ROUTE_START = (305.985199, 241.739304)
DEFAULT_ROUTE_GOAL = (158.323736, 251.743110)
DEFAULT_GVT_ANCHOR = (266.075005, 244.418185)
DEFAULT_GVT_GOAL = (211.846765, 244.355472)


def _parse_float_list(raw: str) -> list[float]:
    """解析逗号分隔浮点列表。"""

    return [float(item.strip()) for item in str(raw).split(",") if item.strip()]


def _normalise_subprocess_timeout(raw_value: float) -> float | None:
    """subprocess timeout 语义；None 表示不按墙钟时间截断。"""

    value = float(raw_value)
    return value if value > 0.0 else None


def _fmt_token(value: float) -> str:
    """把浮点数转成适合文件名的短 token。"""

    text = f"{float(value):.2f}".rstrip("0").rstrip(".")
    return text.replace("-", "n").replace(".", "p")


def _build_source_route(
    xodr_parser: XODRParser,
    route_start_xy: np.ndarray,
    route_goal_xy: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """构建三车同向源车道参考线，用于由纵向间距反算坐标。"""

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
    """按参考线 s 获取 XY 点。"""

    point, _ = get_path_point_at_s(path, float(s_value), cumulative_s)
    return np.asarray(point, dtype=float)


def _build_case_command(
    *,
    args: argparse.Namespace,
    prefix: str,
    main_xy: np.ndarray,
    gvt_xy: np.ndarray,
    exposed_xy: np.ndarray,
    main_speed: float,
    gvt_speed: float,
    exposed_speed: float,
    skip_gif: bool,
) -> list[str]:
    """把一个参数组合封装为 simulate_scene_offline.py 命令。"""

    exposed_motion = "static" if float(exposed_speed) <= 0.05 else "constant"
    command = [
        sys.executable,
        str(PROJECT_ROOT / "simulate_scene_offline.py"),
        "--map-cache",
        str(args.map_cache),
        "--planner-mode",
        "scene_target",
        "--scene-algorithm-symbol",
        "scenarios.cut_out_algorithm.cut_out_algorithm",
        "--ego-start-xy",
        f"{gvt_xy[0]:.6f}",
        f"{gvt_xy[1]:.6f}",
        "--ego-goal-xy",
        f"{args.gvt_goal_xy[0]:.6f}",
        f"{args.gvt_goal_xy[1]:.6f}",
        "--ego-start-speed",
        str(float(gvt_speed)),
        "--ego-max-speed-kph",
        str(float(args.ego_max_speed_kph)),
        "--main-motion",
        "idm",
        "--main-start-xy",
        f"{main_xy[0]:.6f}",
        f"{main_xy[1]:.6f}",
        "--main-goal-xy",
        f"{args.route_goal_xy[0]:.6f}",
        f"{args.route_goal_xy[1]:.6f}",
        "--main-start-speed",
        str(float(main_speed)),
        "--main-desired-speed",
        str(float(main_speed)),
        "--enable-exposed-obstacle",
        "--exposed-motion",
        exposed_motion,
        "--max-steps",
        str(int(args.max_steps)),
        "--log-every",
        str(int(args.log_every)),
        "--output-dir",
        str(args.output_dir),
        "--output-prefix",
        prefix,
    ]
    if exposed_motion == "static":
        command.extend(["--exposed-static-xy", f"{exposed_xy[0]:.6f}", f"{exposed_xy[1]:.6f}"])
    else:
        command.extend(
            [
                "--exposed-start-xy",
                f"{exposed_xy[0]:.6f}",
                f"{exposed_xy[1]:.6f}",
                "--exposed-goal-xy",
                f"{args.route_goal_xy[0]:.6f}",
                f"{args.route_goal_xy[1]:.6f}",
                "--exposed-start-speed",
                str(float(exposed_speed)),
                "--exposed-desired-speed",
                str(float(exposed_speed)),
            ]
        )
    if skip_gif:
        command.append("--skip-gif")
    return command


def _extract_result(json_path: Path) -> dict[str, Any]:
    """从单次仿真 JSON 提取扫描关注的核心指标。"""

    data = json.loads(json_path.read_text(encoding="utf-8"))
    scene_targets = data.get("scene_targets") or []
    main_trace = data.get("main_trace") or []
    ego_trace = data.get("ego_trace") or []

    final_target = scene_targets[-1] if scene_targets else {}
    active_index = next(
        (idx for idx, target in enumerate(scene_targets) if target.get("scene_status") == "active_cutout"),
        -1,
    )
    success_index = next(
        (idx for idx, target in enumerate(scene_targets) if target.get("scene_status") == "cutout_success"),
        -1,
    )
    finish_target = next(
        (
            target
            for target in reversed(scene_targets)
            if target.get("scene_status") in {"cutout_success", "cutout_metric_miss", "cutout_hard_metric_fail"}
        ),
        final_target,
    )
    finish_debug = finish_target.get("scene_debug") or {}
    prepare_targets = [
        target for target in scene_targets if str(target.get("scene_status", "")).startswith("prepare")
    ]
    last_prepare_debug = (prepare_targets[-1].get("scene_debug") or {}) if prepare_targets else {}
    min_gvt_exposed_gap = min(
        [
            float((target.get("scene_debug") or {}).get("gvt_to_exposed_gap_m"))
            for target in scene_targets
            if isinstance((target.get("scene_debug") or {}).get("gvt_to_exposed_gap_m"), (int, float))
        ]
        or [float("inf")]
    )
    main_speeds = [float(item[3]) for item in main_trace if len(item) > 3]
    ego_speeds = [float(item[3]) for item in ego_trace if len(item) > 3]

    return {
        "status": str(data.get("status", "")),
        "completed_steps": int(data.get("completed_steps", 0) or 0),
        "final_scene_status": str(final_target.get("scene_status", "")),
        "final_scene_detail": str(final_target.get("scene_detail", final_target.get("reason", ""))),
        "active_index": int(active_index),
        "success_index": int(success_index),
        "cutout_success": bool(success_index >= 0),
        "finish_metric_name": str(finish_debug.get("finish_metric_name", "")),
        "finish_metric_value_s": finish_debug.get("finish_metric_value_s", ""),
        "finish_metric_error_s": finish_debug.get("finish_metric_error_s", ""),
        "finish_metric_phase": str(finish_debug.get("finish_metric_eval_phase", "")),
        "last_prepare_immediate_metric_s": last_prepare_debug.get("trigger_immediate_metric_s", ""),
        "last_prepare_best_hold_t_s": last_prepare_debug.get("trigger_best_hold_t_s", ""),
        "last_prepare_best_metric_s": last_prepare_debug.get("trigger_best_metric_s", ""),
        "last_prepare_safety_ok": last_prepare_debug.get("safety_decision_ok", ""),
        "last_prepare_release_reason": str(last_prepare_debug.get("safety_source_release_reason", "")),
        "min_gvt_exposed_gap_m": min_gvt_exposed_gap,
        "min_main_speed_mps": min(main_speeds) if main_speeds else "",
        "min_ego_speed_mps": min(ego_speeds) if ego_speeds else "",
    }


def _write_summary(output_dir: Path, rows: list[dict[str, Any]], summary_prefix: str) -> None:
    """把扫描结果同时保存为 CSV 和 JSON。"""

    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{summary_prefix}_summary.json"
    csv_path = output_dir / f"{summary_prefix}_summary.csv"
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    if not rows:
        csv_path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with csv_path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_arg_parser() -> argparse.ArgumentParser:
    """构造扫描器命令行参数。"""

    parser = argparse.ArgumentParser(description="批量扫描 cut-out + 固定基础 IDM 主车测试参数")
    parser.add_argument("--map-cache", type=Path, default=DEFAULT_MAP_CACHE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-prefix", default="cutout_idm_scan")
    parser.add_argument("--route-start-xy", nargs=2, type=float, default=DEFAULT_ROUTE_START)
    parser.add_argument("--route-goal-xy", nargs=2, type=float, default=DEFAULT_ROUTE_GOAL)
    parser.add_argument("--gvt-anchor-xy", nargs=2, type=float, default=DEFAULT_GVT_ANCHOR)
    parser.add_argument("--gvt-goal-xy", nargs=2, type=float, default=DEFAULT_GVT_GOAL)
    parser.add_argument("--vehicle-length-m", type=float, default=4.6, help="三车纵向净距换算使用的车长")
    parser.add_argument("--main-speeds", default="15.0", help="主车速度列表 m/s，例如 10,12,15")
    parser.add_argument("--gvt-speeds", default="", help="GVT 初始速度列表 m/s；空值表示跟随 main-speeds")
    parser.add_argument("--ego-max-speed-kph", type=float, default=60.0, help="生成仿真命令时使用的 GVT 最大速度 km/h")
    parser.add_argument("--exposed-speeds", default="0.0", help="暴露车速度列表 m/s；0 表示静止车")
    parser.add_argument("--main-gvt-gaps", default="30,35,40,45", help="VUT 到 GVT 的初始纵向净距列表 m")
    parser.add_argument("--gvt-exposed-gaps", default="25,30,35,40", help="GVT 到暴露车的初始纵向净距列表 m")
    parser.add_argument("--max-steps", type=int, default=0, help="传给离线仿真的最大步数；<=0 表示不限步数")
    parser.add_argument("--log-every", type=int, default=9999)
    parser.add_argument("--limit", type=int, default=0, help="最多运行多少个组合；0 表示不限制")
    parser.add_argument("--timeout-s", type=float, default=0.0, help="单次仿真墙钟超时时间；<=0 表示不限时间")
    parser.add_argument("--dry-run", action="store_true", help="只打印待运行组合，不执行仿真")
    parser.add_argument("--verbose", action="store_true", help="显示每次仿真的完整 stdout/stderr")
    parser.add_argument("--generate-gif-on-success", type=int, default=0, help="对前 N 个成功样本补生成 GIF")
    return parser


def main() -> None:
    """扫描器入口。"""

    parser = build_arg_parser()
    args = parser.parse_args()
    args.map_cache = Path(args.map_cache)
    args.output_dir = Path(args.output_dir)
    args.route_start_xy = np.asarray(args.route_start_xy, dtype=float)
    args.route_goal_xy = np.asarray(args.route_goal_xy, dtype=float)
    args.gvt_anchor_xy = np.asarray(args.gvt_anchor_xy, dtype=float)
    args.gvt_goal_xy = np.asarray(args.gvt_goal_xy, dtype=float)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    xodr_parser = XODRParser.load_cache(str(args.map_cache))
    source_path, source_cumulative_s = _build_source_route(xodr_parser, args.route_start_xy, args.route_goal_xy)
    _, gvt_s, _ = project_point_to_path(args.gvt_anchor_xy, source_path)
    gvt_xy = _point_at_s(source_path, source_cumulative_s, gvt_s)
    half_pair_length = float(args.vehicle_length_m)

    main_speeds = _parse_float_list(args.main_speeds)
    gvt_speeds = _parse_float_list(args.gvt_speeds) if str(args.gvt_speeds).strip() else []
    exposed_speeds = _parse_float_list(args.exposed_speeds)
    main_gvt_gaps = _parse_float_list(args.main_gvt_gaps)
    gvt_exposed_gaps = _parse_float_list(args.gvt_exposed_gaps)
    rows: list[dict[str, Any]] = []
    success_commands: list[list[str]] = []
    run_index = 0
    combinations = itertools.product(
        main_speeds,
        exposed_speeds,
        main_gvt_gaps,
        gvt_exposed_gaps,
    )
    for (
        main_speed,
        exposed_speed,
        main_gvt_gap,
        gvt_exposed_gap,
    ) in combinations:
        active_gvt_speeds = gvt_speeds or [main_speed]
        for gvt_speed in active_gvt_speeds:
            if args.limit and run_index >= int(args.limit):
                break
            main_s = float(gvt_s) - float(main_gvt_gap) - half_pair_length
            exposed_s = float(gvt_s) + float(gvt_exposed_gap) + half_pair_length
            if main_s < 0.0 or exposed_s > float(source_cumulative_s[-1]):
                print(
                    f"[skip] invalid s main_s={main_s:.2f} exposed_s={exposed_s:.2f} "
                    f"path_end={float(source_cumulative_s[-1]):.2f}",
                    flush=True,
                )
                continue

            main_xy = _point_at_s(source_path, source_cumulative_s, main_s)
            exposed_xy = _point_at_s(source_path, source_cumulative_s, exposed_s)
            prefix = (
                f"{args.output_prefix}_{run_index:03d}"
                f"_vm{_fmt_token(main_speed)}_vg{_fmt_token(gvt_speed)}_ve{_fmt_token(exposed_speed)}"
                f"_mg{_fmt_token(main_gvt_gap)}_ge{_fmt_token(gvt_exposed_gap)}"
            )
            command = _build_case_command(
                args=args,
                prefix=prefix,
                main_xy=main_xy,
                gvt_xy=gvt_xy,
                exposed_xy=exposed_xy,
                main_speed=main_speed,
                gvt_speed=gvt_speed,
                exposed_speed=exposed_speed,
                skip_gif=True,
            )
            base_row: dict[str, Any] = {
                "prefix": prefix,
                "main_speed_mps": float(main_speed),
                "gvt_speed_mps": float(gvt_speed),
                "exposed_speed_mps": float(exposed_speed),
                "main_gvt_gap_m": float(main_gvt_gap),
                "gvt_exposed_gap_m": float(gvt_exposed_gap),
                "main_xy": [round(float(main_xy[0]), 6), round(float(main_xy[1]), 6)],
                "gvt_xy": [round(float(gvt_xy[0]), 6), round(float(gvt_xy[1]), 6)],
                "exposed_xy": [round(float(exposed_xy[0]), 6), round(float(exposed_xy[1]), 6)],
                "json_path": str(args.output_dir / f"{prefix}.json"),
                "command": " ".join(command),
            }
            if args.dry_run:
                print("[dry-run]", " ".join(command), flush=True)
                rows.append({**base_row, "status": "dry_run"})
                run_index += 1
                continue

            print(
                f"[scan] {run_index:03d} vm={main_speed:.1f} vg={gvt_speed:.1f} ve={exposed_speed:.1f} "
                f"main_gvt={main_gvt_gap:.1f} gvt_exp={gvt_exposed_gap:.1f}",
                flush=True,
            )
            try:
                completed = subprocess.run(
                    command,
                    cwd=str(PROJECT_ROOT),
                    env=os.environ.copy(),
                    text=True,
                    capture_output=not bool(args.verbose),
                    check=False,
                    timeout=_normalise_subprocess_timeout(float(args.timeout_s)),
                )
                json_path = args.output_dir / f"{prefix}.json"
                if completed.returncode != 0 or not json_path.exists():
                    row = {
                        **base_row,
                        "status": "subprocess_failed",
                        "returncode": completed.returncode,
                        "stderr_tail": (completed.stderr or "")[-500:] if not args.verbose else "",
                    }
                else:
                    result = _extract_result(json_path)
                    row = {**base_row, **result, "returncode": completed.returncode}
                    print(
                        f"[result] {prefix} status={row.get('status')} final={row.get('final_scene_status')}/"
                        f"{row.get('final_scene_detail')} success={row.get('cutout_success')} "
                        f"metric={row.get('finish_metric_value_s')}",
                        flush=True,
                    )
                    if bool(row.get("cutout_success")):
                        gif_prefix = f"{prefix}_gif"
                        gif_command = _build_case_command(
                            args=args,
                            prefix=gif_prefix,
                            main_xy=main_xy,
                            gvt_xy=gvt_xy,
                            exposed_xy=exposed_xy,
                            main_speed=main_speed,
                            gvt_speed=gvt_speed,
                            exposed_speed=exposed_speed,
                            skip_gif=False,
                        )
                        success_commands.append(gif_command)
                rows.append(row)
            except subprocess.TimeoutExpired:
                rows.append({**base_row, "status": "timeout"})
                print(f"[timeout] {prefix}", flush=True)
            run_index += 1
        if args.limit and run_index >= int(args.limit):
            break

    for gif_index, gif_command in enumerate(success_commands[: max(0, int(args.generate_gif_on_success))]):
        print(f"[gif] generating success GIF {gif_index + 1}/{int(args.generate_gif_on_success)}", flush=True)
        subprocess.run(
            gif_command,
            cwd=str(PROJECT_ROOT),
            env=os.environ.copy(),
            text=True,
            capture_output=not bool(args.verbose),
            check=False,
            timeout=_normalise_subprocess_timeout(float(args.timeout_s)),
        )

    _write_summary(args.output_dir, rows, str(args.output_prefix))
    print(f"[summary] rows={len(rows)} dir={args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
