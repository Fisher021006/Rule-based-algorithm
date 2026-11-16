#!/usr/bin/env python3
"""紧急制动泛化 case 批量运行器。

按 configs/emergency_braking_generalized_cases.json 定义的目标 THW × 主车速度 ×
初始净间距，逐个 case 起独立子进程运行 simulate_scene_offline.py，并抽取
初始间距 / 触发 THW / 触发主车速度三类指标写 summary.json 与 summary.md。

用法：
    python3 batch_sim/run_emergency_braking_matrix.py --dry-run
    python3 batch_sim/run_emergency_braking_matrix.py --pilot
    python3 batch_sim/run_emergency_braking_matrix.py --case T40_V50_G20
    python3 batch_sim/run_emergency_braking_matrix.py --thw 4.0 --speed 5.0
    python3 batch_sim/run_emergency_braking_matrix.py --gif          # 默认不生成 GIF
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
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "emergency_braking_generalized_cases.json"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "emergency_braking_generalized"

PILOT_CASE_IDS = [
    "B0_BASELINE",
    "T25_V50_G20",
    "T40_V50_G20",
    "T50_V50_G20",
]


def load_config(path: Path) -> dict[str, Any]:
    """读取 case 配置。"""

    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def select_cases(
    config: dict[str, Any],
    case_filters: list[str],
    thw_filters: list[float],
    speed_filters: list[float],
    gap_filters: list[float],
    pilot: bool,
) -> list[dict[str, Any]]:
    """按 case id / 目标 THW / 主车速度 / 初始间距过滤；--pilot 只跑代表点。"""

    cases = list(config["cases"])
    if pilot and not (case_filters or thw_filters or speed_filters or gap_filters):
        wanted_pilot = set(PILOT_CASE_IDS)
        return [case for case in cases if str(case["id"]) in wanted_pilot]

    wanted_ids = {str(item).strip() for item in case_filters if str(item).strip()}
    wanted_thw = {round(float(item), 3) for item in thw_filters}
    wanted_speed = {round(float(item), 3) for item in speed_filters}
    wanted_gap = {round(float(item), 3) for item in gap_filters}

    selected = [
        case
        for case in cases
        if (not wanted_ids or str(case["id"]) in wanted_ids)
        and (not wanted_thw or round(float(case["target_thw_s"]), 3) in wanted_thw)
        and (not wanted_speed or round(float(case["vut_speed_mps"]), 3) in wanted_speed)
        and (not wanted_gap or round(float(case["initial_net_gap_m"]), 3) in wanted_gap)
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

    route = config["route"]
    ego = config["ego"]
    vut_motion = config["vut_motion"]
    simulation = config["simulation"]
    steps = int(
        max_steps
        if max_steps is not None
        else case.get("max_steps", simulation["max_steps"])
    )
    start_speed = float(case.get("vut_start_speed_mps", case["vut_speed_mps"]))
    desired_speed = float(case.get("vut_desired_speed_mps", case["vut_speed_mps"]))

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
        f"{float(case['gvt_start_xy'][0]):.12f}",
        f"{float(case['gvt_start_xy'][1]):.12f}",
        "--ego-goal-xy",
        f"{float(route['gvt_goal_xy'][0]):.12f}",
        f"{float(route['gvt_goal_xy'][1]):.12f}",
        "--ego-start-speed",
        str(float(ego["start_speed"])),
        "--ego-max-speed-kph",
        str(float(ego["max_speed_kph"])),
        "--main-start-xy",
        f"{float(route['vut_start_xy'][0]):.12f}",
        f"{float(route['vut_start_xy'][1]):.12f}",
        "--main-goal-xy",
        f"{float(route['vut_goal_xy'][0]):.12f}",
        f"{float(route['vut_goal_xy'][1]):.12f}",
        "--main-motion",
        str(vut_motion["motion"]),
        "--main-start-speed",
        str(start_speed),
        "--main-desired-speed",
        str(desired_speed),
        "--main-max-acc",
        str(float(vut_motion["max_acc"])),
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
    """构造子进程环境变量：目标 THW 与容差。"""

    environment = os.environ.copy()
    environment["ROS_SCENE_BRAKING_TARGET_THW_S"] = str(float(case["target_thw_s"]))
    environment["ROS_SCENE_BRAKING_THW_TOL_S"] = str(float(config["thw_tol_s"]))
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


def _first_brake_frame(targets: list[dict[str, Any]]) -> dict[str, Any] | None:
    """返回首个进入 ACTIVE_BRAKE 的帧。"""

    for frame in targets:
        if str(frame.get("scene_status", "")).startswith("active_brake"):
            return frame
    return None


def extract_result(
    output_json: Path,
    returncode: int,
    elapsed_s: float,
    case: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    """从输出 JSON 抽取初始条件、触发 THW、触发主车速度与安全守卫指标。"""

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

    # ---------- 初始条件 ----------
    initial_debug = _debug(targets[0])
    result["initial_net_gap_m"] = _finite(initial_debug.get("net_gap_m"))
    result["initial_main_v_mps"] = _finite(initial_debug.get("main_v"))
    result["initial_thw_s"] = _finite(initial_debug.get("thw_s"))

    nominal_gap = float(case["initial_net_gap_m"])
    nominal_start_speed = float(case.get("vut_start_speed_mps", case["vut_speed_mps"]))
    result["nominal_net_gap_m"] = nominal_gap
    result["nominal_start_speed_mps"] = nominal_start_speed
    if result["initial_net_gap_m"] is not None:
        result["initial_gap_error_m"] = result["initial_net_gap_m"] - nominal_gap
    if result["initial_main_v_mps"] is not None:
        result["initial_speed_error_mps"] = result["initial_main_v_mps"] - nominal_start_speed

    # ---------- 触发帧 ----------
    trigger = _first_brake_frame(targets)
    result["triggered"] = trigger is not None
    if trigger is not None:
        trigger_debug = _debug(trigger)
        result["trigger_t_s"] = _finite(trigger.get("time_s"))
        result["trigger_step"] = trigger.get("step_index")
        result["trigger_status"] = trigger.get("scene_status")
        result["trigger_reason"] = trigger.get("reason")
        result["trigger_thw_s"] = _finite(trigger_debug.get("thw_s"))
        result["trigger_main_v_mps"] = _finite(trigger_debug.get("main_v"))
        result["trigger_ego_v_mps"] = _finite(trigger_debug.get("ego_v"))
        result["trigger_net_gap_m"] = _finite(trigger_debug.get("net_gap_m"))
        result["trigger_activation_reason"] = trigger_debug.get("thw_activation_reason")
        if result["trigger_thw_s"] is not None:
            result["trigger_thw_error_s"] = result["trigger_thw_s"] - float(case["target_thw_s"])
        nominal_trigger_speed = float(case.get("vut_desired_speed_mps", case["vut_speed_mps"]))
        result["nominal_trigger_speed_mps"] = nominal_trigger_speed
        if result["trigger_main_v_mps"] is not None:
            result["trigger_speed_error_mps"] = result["trigger_main_v_mps"] - nominal_trigger_speed

    # ---------- 安全守卫与终态 ----------
    net_gaps = [
        gap
        for gap in (_finite(_debug(frame).get("net_gap_m")) for frame in targets)
        if gap is not None
    ]
    result["min_net_gap_m"] = min(net_gaps) if net_gaps else None
    result["collision"] = bool(result["min_net_gap_m"] is not None and result["min_net_gap_m"] <= 0.0)
    result["final_scene_status"] = str(targets[-1].get("scene_status", ""))

    main_trace = data.get("main_trace") or []
    if main_trace and len(main_trace[-1]) > 3:
        result["final_main_v_mps"] = _finite(main_trace[-1][3])
    result["main_stopped"] = bool(
        result.get("final_main_v_mps") is not None and result["final_main_v_mps"] <= 0.3
    )

    result["verdict"] = _verdict(result, case, config)
    return result


def _verdict(result: dict[str, Any], case: dict[str, Any], config: dict[str, Any]) -> str:
    """按安全守卫 → 初始条件 → 功能触发的顺序给出判定。"""

    if result.get("error"):
        return "ERROR"
    if bool(result.get("collision", False)):
        return "SAFETY_FAIL"

    gap_error = result.get("initial_gap_error_m")
    speed_error = result.get("initial_speed_error_mps")
    if gap_error is not None and abs(float(gap_error)) > float(config["net_gap_tol_m"]):
        return "PLACEMENT_FAIL"
    if speed_error is not None and abs(float(speed_error)) > float(config["speed_tol_mps"]):
        return "PLACEMENT_FAIL"

    if not bool(result.get("triggered", False)):
        if bool(case.get("expected_infeasible", False)):
            return "INFEASIBLE"
        return "NO_TRIGGER"

    if str(result.get("final_scene_status", "")) not in {"brake_hold_stop", "brake_completed_stop"}:
        return "NO_STOP"

    thw = result.get("trigger_thw_s")
    if thw is None:
        return "THW_MISS"
    target = float(case["target_thw_s"])
    tolerance = float(config["thw_tol_s"])
    if float(thw) < target - tolerance:
        return "LATE"
    if float(thw) > target + tolerance:
        return "THW_MISS"

    # 触发时的主车速度是 IDM 跟车后的实测值，只作为指标报告；
    # 主车速度作为初始条件已在 t=0 的 initial_speed_error_mps 校验。
    trigger_speed_error = result.get("trigger_speed_error_mps")
    if trigger_speed_error is not None:
        result["trigger_speed_ok"] = bool(
            abs(float(trigger_speed_error)) <= float(config["speed_tol_mps"])
        )
    return "PASS"


def format_value(value: Any, digits: int = 3) -> str:
    """把结果值格式化为表格文本。"""

    if value is None:
        return "-"
    if isinstance(value, bool):
        return "Y" if value else "N"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def expected_matches(row: dict[str, Any]) -> bool:
    """判断实测触发方式是否与配置里的预期一致。

    LATE 在配置里写成 expected_trigger=immediate_late 时属于预期行为，
    这里把它和真正的功能失败区分开，避免汇总表被误读。
    """

    if row.get("expected_infeasible"):
        return row.get("verdict") == "INFEASIBLE"
    expected = str(row.get("expected_trigger") or "")
    verdict = str(row.get("verdict") or "")
    activation = str(row.get("trigger_activation_reason") or "")
    trigger_t = row.get("trigger_t_s")
    if expected == "none":
        return trigger_t is None
    if expected == "immediate_late":
        return verdict == "LATE" and activation == "late_thw_trigger"
    if expected == "immediate_exact":
        return (
            verdict == "PASS"
            and activation == "exact_thw_ready"
            and float(trigger_t or 0.0) <= 1e-9
        )
    if expected in {"approach", "approach_from_rest"}:
        return verdict == "PASS" and trigger_t is not None and float(trigger_t) > 1e-9
    return False


def write_summary(output_root: Path, rows: list[dict[str, Any]], config_path: Path) -> None:
    """写 summary.json 和 summary.md。"""

    output_root.mkdir(parents=True, exist_ok=True)
    verdict_counts: dict[str, int] = {}
    for row in rows:
        verdict = str(row.get("verdict", "ERROR"))
        verdict_counts[verdict] = verdict_counts.get(verdict, 0) + 1
        row["expected_ok"] = expected_matches(row)

    payload = {
        "config": str(config_path),
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total": len(rows),
        "verdict_counts": verdict_counts,
        "rows": rows,
    }
    with (output_root / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)

    expected_ok_count = sum(1 for row in rows if row.get("expected_ok"))
    lines = [
        "# 紧急制动泛化测试汇总",
        "",
        f"- 配置：`{config_path}`",
        f"- 运行时间：{payload['generated_at']}",
        f"- 总数：{payload['total']}，"
        + "，".join(f"{key} {value}" for key, value in sorted(verdict_counts.items())),
        f"- 预期一致：{expected_ok_count}/{payload['total']}",
        "",
        "## 主表",
        "",
        "| case | 目标 THW | 主车速度 | 初始间距(标称) | 初始间距(实测) | 初始 THW | 触发 t | 触发 THW | THW 误差 | 触发主车速度 | 速度偏差 | 触发类型 | 预期触发 | 符合预期 | 最小净距 | 终态 | 判定 | 步数 | 耗时 |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            "| {case_id} | {target} | {speed} | {gap_nominal} | {gap_actual} | {thw0} | {t} | {thw} | {thw_err} | {v_trigger} | {v_err} | {trigger_type} | {expected} | {expected_ok} | {min_gap} | {final_status} | {verdict} | {steps} | {elapsed} |".format(
                case_id=row["case_id"],
                target=format_value(row.get("target_thw_s"), 1),
                speed=format_value(row.get("vut_speed_mps"), 1),
                gap_nominal=format_value(row.get("nominal_net_gap_m"), 1),
                gap_actual=format_value(row.get("initial_net_gap_m")),
                thw0=format_value(row.get("expected_initial_thw_s")),
                t=format_value(row.get("trigger_t_s"), 1),
                thw=format_value(row.get("trigger_thw_s")),
                thw_err=format_value(row.get("trigger_thw_error_s")),
                v_trigger=format_value(row.get("trigger_main_v_mps")),
                v_err=format_value(row.get("trigger_speed_error_mps")),
                trigger_type=format_value(row.get("trigger_activation_reason")),
                expected=format_value(row.get("expected_trigger")),
                expected_ok="Y" if row.get("expected_ok") else "N",
                min_gap=format_value(row.get("min_net_gap_m")),
                final_status=format_value(row.get("final_scene_status")),
                verdict=format_value(row.get("verdict")),
                steps=format_value(row.get("completed_steps")),
                elapsed=format_value(row.get("elapsed_s"), 1),
            )
        )

    # 按目标 THW 汇总，便于看每个 THW 档位的覆盖情况。
    lines.extend(["", "## 按目标 THW 汇总", "", "| 目标 THW | 总数 | PASS | LATE | 其他 |", "|---|---|---|---|---|"])
    targets = sorted({round(float(row["target_thw_s"]), 1) for row in rows})
    for target in targets:
        group = [row for row in rows if round(float(row["target_thw_s"]), 1) == target]
        passed = sum(1 for row in group if row.get("verdict") == "PASS")
        late = sum(1 for row in group if row.get("verdict") == "LATE")
        other = len(group) - passed - late
        lines.append(f"| {target:.1f} | {len(group)} | {passed} | {late} | {other} |")

    lines.extend(
        [
            "",
            "## 判定说明",
            "",
            "- PASS：触发 THW 落在目标 ±thw_tol_s 内。",
            "- LATE：t=0 时初始 THW 已低于目标窗口下沿，走 late_thw_trigger；"
            "配置里 expected_trigger=immediate_late 的 case 属于预期行为。",
            "- INFEASIBLE：负样本，PREPARE 超时未等到触发点，配置 expected_infeasible=true。",
            "- 触发主车速度是 IDM 跟车后的实测值，只作为指标报告；"
            "主车速度作为初始条件在 t=0 用 initial_speed_error_mps 校验。",
        ]
    )

    (output_root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    """构造命令行参数解析器。"""

    parser = argparse.ArgumentParser(description="紧急制动泛化 case 批量运行器")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="case 配置 JSON")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT), help="输出根目录")
    parser.add_argument("--case", action="append", default=[], help="只跑指定 case id，可重复")
    parser.add_argument("--thw", action="append", type=float, default=[], help="只跑指定目标 THW，可重复")
    parser.add_argument("--speed", action="append", type=float, default=[], help="只跑指定主车速度，可重复")
    parser.add_argument("--gap", action="append", type=float, default=[], help="只跑指定初始净间距，可重复")
    parser.add_argument("--pilot", action="store_true", help="只跑 4 个代表点（基线 + 三个 THW 档）")
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
    cases = select_cases(
        config,
        args.case,
        args.thw,
        args.speed,
        args.gap,
        args.pilot,
    )
    if not cases:
        raise SystemExit("没有匹配的 case")
    output_root = Path(args.output_root).resolve()

    rows: list[dict[str, Any]] = []
    failures = 0

    for case in cases:
        case_dir = output_root / str(case["id"])
        output_prefix = f"eb_{case['id']}"
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
                "target_thw_s": float(case["target_thw_s"]),
                "vut_speed_mps": float(case["vut_speed_mps"]),
                "expected_initial_thw_s": case.get("expected_initial_thw_s"),
                "expected_trigger": case.get("expected_trigger"),
                "expected_infeasible": bool(case.get("expected_infeasible", False)),
                "gvt_start_xy": case.get("gvt_start_xy"),
                "output_dir": str(case_dir),
                "stdout_tail": stdout_tail,
                "stderr_tail": stderr_tail,
            }
        )
        rows.append(result)

        print(
            "  verdict={verdict} trigger_t={t} thw={thw} v_main={v} gap0={gap} min_gap={min_gap} final={final} steps={steps}".format(
                verdict=format_value(result.get("verdict")),
                t=format_value(result.get("trigger_t_s"), 1),
                thw=format_value(result.get("trigger_thw_s")),
                v=format_value(result.get("trigger_main_v_mps")),
                gap=format_value(result.get("initial_net_gap_m")),
                min_gap=format_value(result.get("min_net_gap_m")),
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
