#!/usr/bin/env python3
"""盲区场景泛化 case 批量运行器。

按 configs/blind_spot_generalized_cases.json 定义的 case 列表，逐个 case 起独立
子进程运行 simulate_scene_offline.py，并抽取相对车道 / 换道起点 / 完成 THW 三类
指标写 summary.json 与 summary.md。

用法：
    python3 batch_sim/run_blind_spot_matrix.py --dry-run
    python3 batch_sim/run_blind_spot_matrix.py --case L_P3_T30 --case R_P3_T30
    python3 batch_sim/run_blind_spot_matrix.py --side R --thw 3.0
    python3 batch_sim/run_blind_spot_matrix.py --gif          # 默认不生成 GIF
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
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "blind_spot_generalized_cases.json"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "blind_spot_generalized"


def load_config(path: Path) -> dict[str, Any]:
    """读取 case 配置。"""

    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def select_cases(
    config: dict[str, Any],
    case_filters: list[str],
    side_filters: list[str],
    thw_filters: list[float],
) -> list[dict[str, Any]]:
    """按 case id / 侧 / 目标 THW 过滤；未给过滤时返回全部。"""

    cases = list(config["cases"])
    wanted_ids = {str(item).strip() for item in case_filters if str(item).strip()}
    wanted_sides = {str(item).strip().upper() for item in side_filters if str(item).strip()}
    wanted_thw = {round(float(item), 3) for item in thw_filters}

    selected = [
        case
        for case in cases
        if (not wanted_ids or str(case["id"]) in wanted_ids)
        and (not wanted_sides or str(case["side"]).upper() in wanted_sides)
        and (not wanted_thw or round(float(case["thw_s"]), 3) in wanted_thw)
    ]
    if wanted_ids:
        missing = wanted_ids - {str(case["id"]) for case in selected}
        if missing:
            raise SystemExit(f"未知 case: {', '.join(sorted(missing))}")
    return selected


def case_label(case: dict[str, Any]) -> str:
    """case 目录与文件名标签，例如 L_P3_T30。"""

    return str(case["id"])


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

    side = config["sides"][case["side"]]
    start_point = config["start_points"][case["start_point"]]
    vut_motion = config["vut_motion"]
    ego = config["ego"]
    simulation = config["simulation"]
    steps = int(simulation["max_steps"] if max_steps is None else max_steps)

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
        f"{float(side['ego']['start_xy'][0]):.12f}",
        f"{float(side['ego']['start_xy'][1]):.12f}",
        "--ego-goal-xy",
        f"{float(side['ego']['goal_xy'][0]):.12f}",
        f"{float(side['ego']['goal_xy'][1]):.12f}",
        "--ego-start-speed",
        str(float(ego["start_speed"])),
        "--ego-max-speed-kph",
        str(float(ego["max_speed_kph"])),
        "--main-start-xy",
        f"{float(side['vut']['start_xy'][0]):.12f}",
        f"{float(side['vut']['start_xy'][1]):.12f}",
        "--main-goal-xy",
        f"{float(side['vut']['goal_xy'][0]):.12f}",
        f"{float(side['vut']['goal_xy'][1]):.12f}",
        "--main-motion",
        str(vut_motion["motion"]),
        "--main-start-speed",
        str(float(vut_motion["start_speed"])),
        "--main-desired-speed",
        str(float(vut_motion["desired_speed"])),
        "--main-lane-change-side",
        str(side["vut_lane_change_side"]),
        "--main-lane-change-start-s",
        str(float(start_point["s_m"])),
        "--main-lane-change-duration-s",
        str(float(vut_motion["lane_change_duration_s"])),
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
    """构造子进程环境变量：场景侧 + 目标 THW + 容差。"""

    side = config["sides"][case["side"]]
    environment = os.environ.copy()
    environment["ROS_SCENE_BLIND_SCENE_SIDE"] = str(side["scene_side"])
    environment["ROS_SCENE_BLIND_TARGET_THW_S"] = str(float(case["thw_s"]))
    environment["ROS_SCENE_BLIND_THW_TOL_S"] = str(float(config["thw_tol_s"]))
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


def _first_frame(targets: list[dict[str, Any]], key: str) -> dict[str, Any] | None:
    """返回首个 scene_debug[key] 为真的帧。"""

    for frame in targets:
        if bool(_debug(frame).get(key, False)):
            return frame
    return None


def extract_result(output_json: Path, returncode: int, elapsed_s: float) -> dict[str, Any]:
    """从输出 JSON 抽取相对车道、换道起点、完成 THW 与安全守卫指标。"""

    result: dict[str, Any] = {
        "output_json": str(output_json),
        "returncode": int(returncode),
        "elapsed_s": round(float(elapsed_s), 2),
    }
    if not output_json.exists():
        result["error"] = "output_json_missing"
        return result

    try:
        with output_json.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except Exception as exc:  # noqa: BLE001 - 单 case 失败不应中断整个矩阵
        result["error"] = f"output_json_unreadable: {exc}"
        return result

    result["status"] = data.get("status")
    result["failure_reason"] = data.get("failure_reason")
    result["completed_steps"] = data.get("completed_steps")

    targets = data.get("scene_targets") or []
    if not targets:
        result["error"] = "scene_targets_missing"
        return result

    # ---------- 相对车道 ----------
    initial = targets[0]
    initial_debug = _debug(initial)
    result["initial_ego_lane"] = initial.get("ego_lane_id")
    result["initial_main_lane"] = initial.get("main_lane_id")
    result["initial_main_abs_d"] = _finite(initial_debug.get("main_abs_d_on_target_line_m"))
    result["initial_rel_s"] = _finite(initial_debug.get("rel_s_m"))
    result["initial_in_blind_zone"] = bool(initial_debug.get("in_blind_zone", False))

    # ---------- 换道起点 ----------
    start = _first_frame(targets, "lane_change_started")
    if start is None:
        result["lane_change_started"] = False
        result["hold_before_start"] = False
    else:
        start_debug = _debug(start)
        result["lane_change_started"] = True
        result["lane_change_start_t_s"] = _finite(start.get("time_s"))
        result["lane_change_start_step"] = start.get("step_index")
        result["lane_change_start_rel_s"] = _finite(start_debug.get("rel_s_m"))
        result["lane_change_start_main_abs_d"] = _finite(start_debug.get("main_abs_d_on_target_line_m"))
        result["lane_change_start_in_blind_zone"] = bool(start_debug.get("in_blind_zone", False))
        result["lane_change_start_ego_lane"] = start.get("ego_lane_id")
        result["lane_change_start_main_lane"] = start.get("main_lane_id")
        start_index = int(start.get("step_index", 0))
        result["hold_before_start"] = any(
            str(frame.get("scene_status", "")) == "blind_zone_hold_success"
            for frame in targets[: start_index + 1]
        )

    # ---------- 完成 THW ----------
    done = _first_frame(targets, "vut_lane_change_complete")
    if done is None:
        result["vut_lane_change_complete"] = False
    else:
        done_debug = _debug(done)
        result["vut_lane_change_complete"] = True
        result["completion_t_s"] = _finite(done.get("time_s"))
        result["completion_step"] = done.get("step_index")
        result["completion_thw_s"] = _finite(done_debug.get("completion_thw_s"))
        result["completion_target_thw_s"] = _finite(done_debug.get("completion_target_thw_s"))
        result["completion_thw_error_s"] = _finite(done_debug.get("completion_thw_error_s"))
        result["completion_success"] = bool(done_debug.get("completion_success", False))
        result["completion_rel_s"] = _finite(done_debug.get("rel_s_m"))
        result["completion_net_gap_m"] = _finite(done_debug.get("completion_net_gap_m"))
        result["completion_ego_v_mps"] = _finite(done_debug.get("ego_v_mps"))
        result["completion_main_v_mps"] = _finite(done_debug.get("main_v_mps"))
        result["completion_ego_lane"] = done.get("ego_lane_id")
        result["completion_main_lane"] = done.get("main_lane_id")

    # ---------- 安全守卫 ----------
    result["hard_collision_any"] = any(
        bool(_debug(frame).get("hard_collision_risk", False)) for frame in targets
    )
    net_gaps = [
        gap
        for gap in (_finite(_debug(frame).get("completion_net_gap_m")) for frame in targets)
        if gap is not None
    ]
    result["min_net_gap_m"] = min(net_gaps) if net_gaps else None
    thws = [
        thw
        for thw in (_finite(_debug(frame).get("completion_thw_s")) for frame in targets)
        if thw is not None
    ]
    result["min_thw_s"] = min(thws) if thws else None
    result["final_scene_status"] = str(targets[-1].get("scene_status", ""))

    # ---------- 判定 ----------
    result["lane_ok"] = bool(
        result.get("completion_ego_lane")
        and result.get("completion_main_lane")
        and result["completion_ego_lane"] == result["completion_main_lane"]
    )
    result["verdict"] = _verdict(result)
    return result


def _verdict(result: dict[str, Any]) -> str:
    """按安全守卫 → 场景有效性 → THW 的顺序给出判定。"""

    if result.get("error"):
        return "ERROR"
    if result.get("hard_collision_any", False):
        return "SAFETY_FAIL"
    status = str(result.get("status", ""))
    final_scene_status = str(result.get("final_scene_status", ""))
    if status == "invalid_scene_geometry" or final_scene_status == "invalid_scene_geometry":
        return "INFEASIBLE"
    if status == "scene_completed":
        if bool(result.get("completion_success", False)):
            return "PASS"
        return "THW_MISS"
    if status == "blind_spot_thw_miss":
        return "THW_MISS"
    return "ERROR"


def format_value(value: Any, digits: int = 3) -> str:
    """把结果值格式化为表格文本。"""

    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    if isinstance(value, bool):
        return "Y" if value else "N"
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
        "# 盲区场景泛化测试汇总",
        "",
        f"- 配置：`{config_path}`",
        f"- 运行时间：{payload['generated_at']}",
        f"- 总数：{payload['total']}，"
        + "，".join(f"{key} {value}" for key, value in sorted(verdict_counts.items())),
        "",
        "## 主表",
        "",
        "| case | 侧 | 起点 s | 目标 THW | status | 实测 THW | 误差 | 判定 | 换道起点 t | 起点 rel_s | 起点在盲区 | 盲区已建立 | 完成 rel_s | 完成净距 | 最小净距 | 碰撞 | 完成车道 | 步数 | 耗时 |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            "| {case} | {side} | {start_s} | {target} | {status} | {thw} | {error} | {verdict} | {start_t} | {start_rel_s} | {start_in_blind} | {hold} | {done_rel_s} | {net_gap} | {min_gap} | {collision} | {lane} | {steps} | {elapsed} |".format(
                case=row.get("case_id"),
                side=row.get("side"),
                start_s=format_value(row.get("start_s_m"), 1),
                target=format_value(row.get("target_thw_s"), 1),
                status=format_value(row.get("status")),
                thw=format_value(row.get("completion_thw_s"), 3),
                error=format_value(row.get("completion_thw_error_s"), 3),
                verdict=row.get("verdict"),
                start_t=format_value(row.get("lane_change_start_t_s"), 1),
                start_rel_s=format_value(row.get("lane_change_start_rel_s"), 3),
                start_in_blind=format_value(row.get("lane_change_start_in_blind_zone")),
                hold=format_value(row.get("hold_before_start")),
                done_rel_s=format_value(row.get("completion_rel_s"), 3),
                net_gap=format_value(row.get("completion_net_gap_m"), 3),
                min_gap=format_value(row.get("min_net_gap_m"), 3),
                collision=format_value(row.get("hard_collision_any")),
                lane="{}/{}".format(
                    format_value(row.get("completion_ego_lane")),
                    format_value(row.get("completion_main_lane")),
                ),
                steps=format_value(row.get("completed_steps")),
                elapsed=format_value(row.get("elapsed_s"), 1),
            )
        )
    (output_root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    """构造命令行参数解析器。"""

    parser = argparse.ArgumentParser(description="盲区场景泛化 case 批量运行器")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="case 配置 JSON")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT), help="输出根目录")
    parser.add_argument("--case", action="append", default=[], help="只跑指定 case id，可重复")
    parser.add_argument("--side", action="append", default=[], help="只跑指定侧 L/R，可重复")
    parser.add_argument("--thw", action="append", type=float, default=[], help="只跑指定目标 THW，可重复")
    parser.add_argument("--max-steps", type=int, default=None, help="覆盖配置里的 max_steps")
    parser.add_argument("--gif", action="store_true", help="生成 GIF；默认不生成")
    parser.add_argument("--timeout", type=float, default=300.0, help="单 case 超时秒数")
    parser.add_argument("--dry-run", action="store_true", help="只打印命令，不执行")
    parser.add_argument("--fail-fast", action="store_true", help="遇到失败立即停止")
    return parser


def main() -> int:
    """脚本入口。"""

    args = build_arg_parser().parse_args()
    config_path = Path(args.config).resolve()
    config = load_config(config_path)
    cases = select_cases(config, args.case, args.side, args.thw)
    output_root = Path(args.output_root).resolve()

    rows: list[dict[str, Any]] = []
    failures = 0

    for case in cases:
        identifier = case_label(case)
        case_dir = output_root / identifier
        output_prefix = f"blind_spot_{identifier}"
        command = build_command(
            config,
            case,
            case_dir,
            output_prefix,
            max_steps=args.max_steps,
            skip_gif=not args.gif,
        )
        environment = build_environment(config, case)

        print(f"\n=== {identifier} ===")
        if args.dry_run:
            print(" ".join(command))
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
        result = extract_result(output_json, returncode, elapsed_s)
        result.update(
            {
                "case_id": identifier,
                "side": case["side"],
                "start_point": case["start_point"],
                "start_s_m": float(config["start_points"][case["start_point"]]["s_m"]),
                "target_thw_s": float(case["thw_s"]),
                "output_dir": str(case_dir),
                "stdout_tail": stdout_tail,
                "stderr_tail": stderr_tail,
            }
        )
        rows.append(result)

        print(
            "  verdict={verdict} status={status} THW={thw} err={error} "
            "start_t={start_t} start_rel_s={start_rel_s} hold={hold} lanes={lane}/{main_lane}".format(
                verdict=result.get("verdict"),
                status=format_value(result.get("status")),
                thw=format_value(result.get("completion_thw_s"), 3),
                error=format_value(result.get("completion_thw_error_s"), 3),
                start_t=format_value(result.get("lane_change_start_t_s"), 1),
                start_rel_s=format_value(result.get("lane_change_start_rel_s"), 3),
                hold=format_value(result.get("hold_before_start")),
                lane=format_value(result.get("completion_ego_lane")),
                main_lane=format_value(result.get("completion_main_lane")),
            )
        )
        if result.get("verdict") in {"ERROR", "SAFETY_FAIL"}:
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
