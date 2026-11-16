#!/usr/bin/env python3
"""切出（cut-out）泛化 case 批量运行器。

按 configs/cut_out_generalized_cases.json 定义的位置 / 切出方向 / 主车速度 /
暴露车速度 / 目标指标，逐个 case 起独立子进程运行 simulate_scene_offline.py，
抽取初始间距、触发时刻、换道过半指标与安全门结果，写 summary.json / summary.md。

用法：
    python3 batch_sim/run_cut_out_matrix.py --dry-run
    python3 batch_sim/run_cut_out_matrix.py --pilot
    python3 batch_sim/run_cut_out_matrix.py --case P1_R_static_vm15_ve0
    python3 batch_sim/run_cut_out_matrix.py --gif          # 默认不生成 GIF
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "cut_out_generalized_cases.json"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "cut_out_generalized"

PILOT_CASE_IDS = [
    "P1_R_static_vm15_ve0",
    "P2_R_slow_vm10_ve3",
    "P3_L_static_vm10_fix",
    "P1_L_static_vm20",
]

# 场景构建阶段直接拒绝的失败原因，归为放置/几何问题而不是不可行。
PLACEMENT_FAILURES = {
    "main_heading_mismatch",
    "exposed_heading_mismatch",
    "not_same_source_lane",
    "gvt_too_close_to_exposed",
    "target_lane_missing",
    "source_lane_missing",
}


def load_config(path: Path) -> dict[str, Any]:
    """读取 case 配置。"""

    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def select_cases(
    config: dict[str, Any],
    case_filters: list[str],
    location_filters: list[str],
    side_filters: list[str],
    pilot: bool,
) -> list[dict[str, Any]]:
    """按 case id / 位置 / 切出方向过滤；--pilot 只跑代表点。"""

    cases = list(config["cases"])
    if pilot and not (case_filters or location_filters or side_filters):
        wanted_pilot = set(PILOT_CASE_IDS)
        return [case for case in cases if str(case["id"]) in wanted_pilot]

    wanted_ids = {str(item).strip() for item in case_filters if str(item).strip()}
    wanted_locations = {str(item).strip() for item in location_filters if str(item).strip()}
    wanted_sides = {str(item).strip() for item in side_filters if str(item).strip()}

    selected = [
        case
        for case in cases
        if (not wanted_ids or str(case["id"]) in wanted_ids)
        and (not wanted_locations or str(case.get("location", "")) in wanted_locations)
        and (not wanted_sides or str(case.get("side", "")) in wanted_sides)
    ]
    if wanted_ids:
        missing = wanted_ids - {str(case["id"]) for case in selected}
        if missing:
            raise SystemExit(f"未知 case: {', '.join(sorted(missing))}")
    return selected


def build_command(
    config: dict[str, Any],
    case: dict[str, Any],
    output_dir: Path,
    output_prefix: str,
    *,
    max_steps: int | None,
    skip_gif: bool,
) -> list[str]:
    """构造 simulate_scene_offline.py 命令。"""

    simulation = config["simulation"]
    vut_motion = config["vut_motion"]
    steps = int(max_steps if max_steps is not None else simulation["max_steps"])
    exposed_speed = float(case["exposed_speed_mps"])

    command = [
        sys.executable,
        str(PROJECT_ROOT / "simulate_scene_offline.py"),
        "--planner-mode",
        str(config["planner_mode"]),
        "--scene-algorithm-symbol",
        str(config["scenario_algorithm"]),
        "--map-cache",
        str(PROJECT_ROOT / str(config["map_cache"])),
        "--ego-start-xy",
        f"{float(case['ego_start_xy'][0]):.12f}",
        f"{float(case['ego_start_xy'][1]):.12f}",
        "--ego-goal-xy",
        f"{float(case['ego_goal_xy'][0]):.12f}",
        f"{float(case['ego_goal_xy'][1]):.12f}",
        "--ego-start-speed",
        str(float(case["vut_speed_mps"])),
        "--ego-max-speed-kph",
        str(float(case.get("ego_max_speed_kph", 60.0))),
        "--main-start-xy",
        f"{float(case['main_start_xy'][0]):.12f}",
        f"{float(case['main_start_xy'][1]):.12f}",
        "--main-goal-xy",
        f"{float(case['main_goal_xy'][0]):.12f}",
        f"{float(case['main_goal_xy'][1]):.12f}",
        "--main-motion",
        str(vut_motion["motion"]),
        "--main-start-speed",
        str(float(case["vut_speed_mps"])),
        "--main-desired-speed",
        str(float(case["vut_desired_speed_mps"])),
        "--main-max-acc",
        str(float(vut_motion["max_acc"])),
        "--enable-exposed-obstacle",
        "--exposed-ptc-id",
        "exposed",
        "--exposed-start-xy",
        f"{float(case['exposed_start_xy'][0]):.12f}",
        f"{float(case['exposed_start_xy'][1]):.12f}",
        "--exposed-goal-xy",
        f"{float(case['exposed_goal_xy'][0]):.12f}",
        f"{float(case['exposed_goal_xy'][1]):.12f}",
        "--exposed-motion",
        "constant",
        "--exposed-start-speed",
        str(exposed_speed),
        "--exposed-desired-speed",
        str(exposed_speed),
        "--dt",
        str(float(simulation["dt"])),
        "--max-steps",
        str(steps),
        "--output-dir",
        str(output_dir),
        "--output-prefix",
        output_prefix,
    ]
    if skip_gif:
        command.append("--skip-gif")
    return command


def build_environment(config: dict[str, Any], case: dict[str, Any]) -> dict[str, str]:
    """构造子进程环境变量：切出方向、目标指标、容差与换道时长。"""

    environment = os.environ.copy()
    environment["ROS_SCENE_CUTOUT_SIDE"] = str(case.get("side", "left"))
    environment["ROS_SCENE_CUTOUT_TARGET_THW_S"] = str(float(case["target_thw_s"]))
    environment["ROS_SCENE_CUTOUT_TARGET_TTC_S"] = str(float(case["target_ttc_s"]))
    environment["ROS_SCENE_CUTOUT_THW_TOL_S"] = str(float(config["thw_tol_s"]))
    environment["ROS_SCENE_CUTOUT_TTC_TOL_S"] = str(float(config["ttc_tol_s"]))
    environment["ROS_SCENE_CUTOUT_HARD_MIN_THW_S"] = str(float(config["hard_min_thw_s"]))
    environment["ROS_SCENE_CUTOUT_HARD_MIN_TTC_S"] = str(float(config["hard_min_ttc_s"]))
    environment["ROS_SCENE_CUTOUT_LANE_CHANGE_TIME_S"] = str(float(case["lane_change_time_s"]))
    environment["ROS_SCENE_CUTOUT_LANE_CHANGE_TIME_MIN_S"] = str(
        float(case.get("lane_change_time_min_s", case["lane_change_time_s"]))
    )
    return environment


def _finite(value: Any) -> float | None:
    """把 JSON 里的数值转成有限浮点数；NaN/Infinity/None 返回 None。"""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _debug(frame: dict[str, Any]) -> dict[str, Any]:
    """取一帧的 scene_debug。"""

    debug = frame.get("scene_debug")
    return debug if isinstance(debug, dict) else {}


def _first_active_frame(targets: list[dict[str, Any]]) -> dict[str, Any] | None:
    """返回首个进入 ACTIVE_CUTOUT 的帧。"""

    for frame in targets:
        if str(frame.get("scene_status", "")).startswith("active_cutout"):
            return frame
    return None


def _finish_frame(targets: list[dict[str, Any]]) -> dict[str, Any] | None:
    """返回首个 cutout_success 帧。"""

    for frame in targets:
        if str(frame.get("scene_status", "")) == "cutout_success":
            return frame
    return None


def _scene_failure_reasons(targets: list[dict[str, Any]]) -> list[str]:
    """收集场景构建/等待阶段的失败原因。"""

    reasons: list[str] = []
    for frame in targets:
        detail = str(frame.get("scene_detail", "") or "")
        status = str(frame.get("scene_status", "") or "")
        for reason in PLACEMENT_FAILURES:
            if detail == reason or status == reason:
                if reason not in reasons:
                    reasons.append(reason)
    return reasons


def extract_result(
    output_json: Path,
    returncode: int,
    elapsed_s: float,
    case: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    """从输出 JSON 抽取初始几何、触发指标、终态指标与判定。"""

    result: dict[str, Any] = {
        "output_json": str(output_json),
        "returncode": int(returncode),
        "elapsed_s": round(float(elapsed_s), 2),
    }
    if not output_json.exists():
        result["error"] = "output_json_missing"
        result["verdict"] = "ERROR"
        return result

    try:
        with output_json.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except Exception as exc:  # noqa: BLE001 - 单 case 失败不应中断整个矩阵
        result["error"] = f"output_json_unreadable: {exc}"
        result["verdict"] = "ERROR"
        return result

    result["status"] = data.get("status")
    result["failure_reason"] = data.get("failure_reason")
    result["completed_steps"] = data.get("completed_steps")
    result["elapsed_wall_time_s"] = data.get("elapsed_wall_time_s")

    targets = data.get("scene_targets") or []
    if not targets:
        result["error"] = "scene_targets_missing"
        result["verdict"] = "ERROR"
        return result

    # ---------- 初始几何 ----------
    initial = _debug(targets[0])
    result["initial_vut_gvt_gap_m"] = _finite(initial.get("vut_to_gvt_gap_m"))
    result["initial_gvt_exposed_gap_m"] = _finite(initial.get("gvt_to_exposed_gap_m"))
    result["initial_vut_d_on_source_m"] = _finite(initial.get("vut_d_on_source"))
    result["initial_same_source_lane"] = initial.get("same_source_lane")
    result["source_lane_id"] = initial.get("source_lane_id")
    result["target_lane_id"] = initial.get("target_lane_id")

    # ---------- 触发帧 ----------
    trigger = _first_active_frame(targets)
    result["triggered"] = trigger is not None
    if trigger is not None:
        trigger_debug = _debug(trigger)
        result["trigger_t_s"] = _finite(trigger.get("time_s"))
        result["trigger_step"] = trigger.get("step_index")
        result["trigger_status"] = trigger.get("scene_status")
        result["trigger_detail"] = trigger.get("scene_detail")
        result["trigger_metric_name"] = trigger_debug.get("trigger_metric_name")
        result["trigger_target_metric_s"] = _finite(trigger_debug.get("trigger_target_metric_s"))
        result["trigger_metric_s"] = _finite(trigger_debug.get("trigger_best_metric_s"))
        result["trigger_metric_error_s"] = _finite(trigger_debug.get("trigger_best_metric_error_s"))
        result["trigger_hold_t_s"] = _finite(trigger_debug.get("trigger_best_hold_t_s"))
        result["trigger_late_abort"] = trigger_debug.get("trigger_late_abort")
        result["trigger_immediate_hit"] = trigger_debug.get("trigger_immediate_metric_hit")
        result["trigger_immediate_hard_safe"] = trigger_debug.get("trigger_immediate_hard_safe")

    # ---------- 终态指标 ----------
    finish = _finish_frame(targets)
    result["finished"] = finish is not None
    if finish is not None:
        finish_debug = _debug(finish)
        result["finish_t_s"] = _finite(finish.get("time_s"))
        result["finish_metric_name"] = finish_debug.get("finish_metric_name")
        result["finish_metric_s"] = _finite(finish_debug.get("finish_metric_value_s"))
        result["finish_target_metric_s"] = _finite(finish_debug.get("finish_target_metric_s"))
        result["finish_metric_error_s"] = _finite(finish_debug.get("finish_metric_error_s"))
        result["finish_metric_tolerance_s"] = _finite(finish_debug.get("finish_metric_tolerance_s"))
        result["finish_metric_hit"] = finish_debug.get("finish_metric_hit")
        result["finish_hard_safe"] = finish_debug.get("finish_hard_safe")
        result["finish_metric_eval_time_s"] = _finite(finish_debug.get("finish_metric_eval_time_s"))
        result["finish_metric_lane_change_time_s"] = _finite(
            finish_debug.get("finish_metric_lane_change_time_s")
        )
        result["finish_source"] = finish_debug.get("finish_source")
        result["locked_source_lane_id"] = finish_debug.get("locked_source_lane_id")

    result["final_scene_status"] = str(targets[-1].get("scene_status", ""))
    result["final_scene_detail"] = str(targets[-1].get("scene_detail", ""))
    result["placement_failures"] = _scene_failure_reasons(targets)

    result["verdict"] = _verdict(result, case, config)
    return result


def _verdict(result: dict[str, Any], case: dict[str, Any], config: dict[str, Any]) -> str:
    """按构建失败 → 安全 → 功能指标的顺序给出判定。"""

    if result.get("error"):
        return "ERROR"

    final_status = str(result.get("final_scene_status", ""))
    placement_failures = result.get("placement_failures") or []
    if placement_failures and not result.get("finished"):
        return "INVALID_PLACEMENT"

    if result.get("finished"):
        if not bool(result.get("finish_hard_safe", True)):
            return "SAFETY_FAIL"
        if bool(result.get("finish_metric_hit", False)):
            return "PASS"
        error = result.get("finish_metric_error_s")
        tolerance = result.get("finish_metric_tolerance_s")
        if error is not None and tolerance is not None:
            if float(error) < -float(tolerance):
                return "LATE"
            if float(error) > float(tolerance):
                return "EARLY"
        return "METRIC_MISS"

    if result.get("triggered"):
        return "NO_COMPLETION"

    # 未触发且没有几何拒绝：场景在 PREPARE 阶段等不到满足硬门的触发点。
    # 包括 PREPARE 超时、触发窗口滑过后进入 cooldown、以及随后规划器失败。
    sim_status = str(result.get("status", ""))
    if final_status in {"max_steps_reached", "planner_failed", "waiting_cutout_scene"} or sim_status in {
        "max_steps_reached",
        "planner_failed",
    }:
        return "INFEASIBLE"

    return "NO_COMPLETION"


def format_value(value: Any, digits: int = 3) -> str:
    """把结果值格式化为表格文本。"""

    if value is None:
        return "-"
    if isinstance(value, bool):
        return "Y" if value else "N"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def write_summary(output_root: Path, rows: list[dict[str, Any]], config_path: Path) -> None:
    """写 summary.json 和 summary.md。"""

    output_root.mkdir(parents=True, exist_ok=True)
    verdict_counts: dict[str, int] = {}
    for row in rows:
        verdict = str(row.get("verdict", "ERROR"))
        verdict_counts[verdict] = verdict_counts.get(verdict, 0) + 1

    payload = {
        "config": str(config_path),
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total": len(rows),
        "verdict_counts": verdict_counts,
        "rows": rows,
    }
    with (output_root / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)

    lines = [
        "# 切出泛化测试汇总",
        "",
        f"- 配置：`{config_path}`",
        f"- 运行时间：{payload['generated_at']}",
        f"- 总数：{payload['total']}，"
        + "，".join(f"{key} {value}" for key, value in sorted(verdict_counts.items())),
        "",
        "## 主表",
        "",
        "| case | 位置 | 切出 | 主车 | 暴露车 | 指标 | 目标 | 实测 | 误差 | 触发 t | 换道/评估 | 初距 VUT-GVT | 初距 GVT-暴露 | 硬安全 | 终态 | 预期 | 符合 | 判定 | 步数 |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        expected = str(row.get("expected") or "")
        matched = "Y" if expected and expected == str(row.get("verdict")) else "N"
        lines.append(
            "| {case_id} | {location} | {side} | {vut} | {exposed} | {metric} | {target} | {value} | {error} | {trigger_t} | {eval_time} | {gap_vg} | {gap_ge} | {hard_safe} | {final} | {expected} | {matched} | {verdict} | {steps} |".format(
                case_id=row["case_id"],
                location=format_value(row.get("location")),
                side=format_value(row.get("side")),
                vut=format_value(row.get("vut_speed_mps"), 1),
                exposed=format_value(row.get("exposed_speed_mps"), 1),
                metric=format_value(row.get("finish_metric_name") or row.get("metric_name")),
                target=format_value(row.get("finish_target_metric_s") or row.get("target_metric_s")),
                value=format_value(row.get("finish_metric_s")),
                error=format_value(row.get("finish_metric_error_s")),
                trigger_t=format_value(row.get("trigger_t_s"), 1),
                eval_time=format_value(row.get("finish_metric_eval_time_s"), 2),
                gap_vg=format_value(row.get("initial_vut_gvt_gap_m"), 1),
                gap_ge=format_value(row.get("initial_gvt_exposed_gap_m"), 1),
                hard_safe=format_value(row.get("finish_hard_safe")),
                final=format_value(row.get("final_scene_status")),
                expected=format_value(expected),
                matched=matched,
                verdict=format_value(row.get("verdict")),
                steps=format_value(row.get("completed_steps")),
            )
        )

    lines.extend(
        [
            "",
            "## 判定说明",
            "",
            "- PASS：`cutout_success`，换道过半指标在目标容差内，且硬安全门通过。",
            "- LATE：完成切出但指标低于目标下沿（保守方向）。",
            "- EARLY：完成切出但指标高于目标上沿。",
            "- INFEASIBLE：PREPARE 超时未触发（初距/目标指标组合在硬门约束下不可达）。",
            "- INVALID_PLACEMENT：场景构建阶段被几何/同车道检查拒绝。",
            "- SAFETY_FAIL：硬安全门失败。",
        ]
    )

    (output_root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    """构造命令行参数解析器。"""

    parser = argparse.ArgumentParser(description="切出泛化 case 批量运行器")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="case 配置 JSON")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT), help="输出根目录")
    parser.add_argument("--case", action="append", default=[], help="只跑指定 case id，可重复")
    parser.add_argument("--location", action="append", default=[], help="只跑指定位置，可重复")
    parser.add_argument("--side", action="append", default=[], help="只跑指定切出方向 left/right，可重复")
    parser.add_argument("--pilot", action="store_true", help="只跑 4 个代表点")
    parser.add_argument("--max-steps", type=int, default=None, help="覆盖所有 case 的 max_steps")
    parser.add_argument("--gif", action="store_true", help="生成 GIF；默认跳过以加快矩阵")
    parser.add_argument("--timeout", type=float, default=300.0, help="单 case 超时秒数")
    parser.add_argument("--dry-run", action="store_true", help="只打印命令，不执行")
    parser.add_argument("--fail-fast", action="store_true", help="遇到失败立即停止")
    return parser


def main() -> int:
    """脚本入口。"""

    args = build_arg_parser().parse_args()
    config_path = Path(args.config).resolve()
    config = load_config(config_path)
    cases = select_cases(config, args.case, args.location, args.side, args.pilot)
    if not cases:
        raise SystemExit("没有匹配的 case")
    output_root = Path(args.output_root).resolve()

    rows: list[dict[str, Any]] = []
    failures = 0

    for case in cases:
        case_dir = output_root / str(case["id"])
        output_prefix = f"cutout_{case['id']}"
        command = build_command(
            config,
            case,
            case_dir,
            output_prefix,
            max_steps=args.max_steps,
            skip_gif=not args.gif,
        )
        environment = build_environment(config, case)

        print(f"\n=== {case['id']} ===")
        print(" ".join(command))
        if args.dry_run:
            continue

        case_dir.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        try:
            completed = subprocess.run(
                command,
                cwd=str(PROJECT_ROOT),
                env=environment,
                capture_output=True,
                text=True,
                timeout=float(args.timeout),
            )
            returncode = int(completed.returncode)
            stdout_tail = (completed.stdout or "")[-3000:]
            stderr_tail = (completed.stderr or "")[-3000:]
        except subprocess.TimeoutExpired as exc:
            returncode = -1
            stdout_tail = (exc.stdout or "")[-3000:] if isinstance(exc.stdout, str) else ""
            stderr_tail = f"timeout after {args.timeout}s"
        elapsed_s = time.perf_counter() - started

        output_json = case_dir / f"{output_prefix}.json"
        result = extract_result(output_json, returncode, elapsed_s, case, config)
        result.update(
            {
                "case_id": str(case["id"]),
                "location": case.get("location"),
                "side": case.get("side"),
                "vut_speed_mps": float(case["vut_speed_mps"]),
                "vut_desired_speed_mps": float(case["vut_desired_speed_mps"]),
                "exposed_speed_mps": float(case["exposed_speed_mps"]),
                "metric_name": case.get("metric_name"),
                "target_metric_s": float(case["target_metric_s"]),
                "expected": case.get("expected"),
                "output_dir": str(case_dir),
                "stdout_tail": stdout_tail,
                "stderr_tail": stderr_tail,
            }
        )
        rows.append(result)

        print(
            "  verdict={verdict} trigger_t={t} metric={metric} target={target} err={err} "
            "gap_vg={gap_vg} gap_ge={gap_ge} hard_safe={hard} final={final} steps={steps}".format(
                verdict=format_value(result.get("verdict")),
                t=format_value(result.get("trigger_t_s"), 1),
                metric=format_value(result.get("finish_metric_s")),
                target=format_value(result.get("finish_target_metric_s")),
                err=format_value(result.get("finish_metric_error_s")),
                gap_vg=format_value(result.get("initial_vut_gvt_gap_m"), 1),
                gap_ge=format_value(result.get("initial_gvt_exposed_gap_m"), 1),
                hard=format_value(result.get("finish_hard_safe")),
                final=format_value(result.get("final_scene_status")),
                steps=format_value(result.get("completed_steps")),
            )
        )
        if returncode != 0 or result.get("verdict") in {"ERROR", "SAFETY_FAIL"}:
            failures += 1
            print(f"  returncode={returncode}")
            if stderr_tail.strip():
                print("  stderr tail:")
                print("  " + stderr_tail.strip().replace("\n", "\n  "))
            if args.fail_fast:
                write_summary(output_root, rows, config_path)
                return 1

    if args.dry_run:
        print(f"\n共 {len(cases)} 个 case（dry-run，未执行）")
        return 0

    write_summary(output_root, rows, config_path)
    print(f"\n汇总：{output_root / 'summary.md'}")
    print(f"总数 {len(rows)}，ERROR/SAFETY_FAIL {failures}")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
