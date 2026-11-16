#!/usr/bin/env python3
"""交叉口泛化 case 批量运行器。

按 configs/intersection_generalized_cases.json 定义的模式和入区时差，
逐个 case 起独立子进程运行 simulate_scene_offline.py。

用法：
    python3 batch_sim/run_intersection_matrix.py --pilot
    python3 batch_sim/run_intersection_matrix.py --delta 0 --delta 1.5
    python3 batch_sim/run_intersection_matrix.py --mode M2 --dry-run
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
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "intersection_generalized_cases.json"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "交叉口泛化测试"


def load_config(path: Path) -> dict[str, Any]:
    """读取 case 配置。"""

    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def select_modes(config: dict[str, Any], mode_filters: list[str]) -> list[dict[str, Any]]:
    """按 id 或 name 过滤模式；未给过滤时返回全部。"""

    modes = list(config["modes"])
    if not mode_filters:
        return modes
    wanted = {str(item).strip() for item in mode_filters if str(item).strip()}
    selected = [
        mode
        for mode in modes
        if str(mode["id"]) in wanted or str(mode["name"]) in wanted
    ]
    missing = wanted - {str(mode["id"]) for mode in selected} - {str(mode["name"]) for mode in selected}
    if missing:
        raise SystemExit(f"未知模式: {', '.join(sorted(missing))}")
    return selected


def select_deltas(config: dict[str, Any], delta_filters: list[float], pilot: bool) -> list[float]:
    """选择入区时差；--pilot 固定只跑 0。"""

    if pilot and not delta_filters:
        return [0.0]
    deltas = list(delta_filters) if delta_filters else list(config["deltas"])
    return [float(value) for value in deltas]


def delta_tag(delta: float) -> str:
    """文件名安全的时间差标签，例如 +0.0 -> p0p0，-1.5 -> m1p5。"""

    return f"{delta:+.1f}".replace("+", "p").replace("-", "m").replace(".", "p")


def case_id(mode: dict[str, Any], delta: float) -> str:
    """稳定且可读的 case id。"""

    return f"{mode['id']}_{mode['name']}_d{delta:+.1f}"


def build_command(
    config: dict[str, Any],
    mode: dict[str, Any],
    delta: float,
    output_dir: Path,
    output_prefix: str,
    *,
    max_steps: int | None,
    skip_gif: bool,
) -> list[str]:
    """构造 simulate_scene_offline.py 命令。"""

    ego = config["ego"]
    vut = config["vut"]
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
        f"{float(mode['gvt']['start_xy'][0]):.12f}",
        f"{float(mode['gvt']['start_xy'][1]):.12f}",
        "--ego-goal-xy",
        f"{float(mode['gvt']['goal_xy'][0]):.12f}",
        f"{float(mode['gvt']['goal_xy'][1]):.12f}",
        "--ego-start-speed",
        str(float(ego["start_speed"])),
        "--ego-max-speed-kph",
        str(float(ego["max_speed_kph"])),
        "--main-start-xy",
        f"{float(mode['vut']['start_xy'][0]):.12f}",
        f"{float(mode['vut']['start_xy'][1]):.12f}",
        "--main-goal-xy",
        f"{float(mode['vut']['goal_xy'][0]):.12f}",
        f"{float(mode['vut']['goal_xy'][1]):.12f}",
        "--main-motion",
        str(vut["motion"]),
        "--main-start-speed",
        str(float(vut["start_speed"])),
        "--main-desired-speed",
        str(float(vut["desired_speed"])),
        "--main-max-acc",
        str(float(vut["max_acc"])),
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


def build_environment(config: dict[str, Any], mode: dict[str, Any], delta: float) -> dict[str, str]:
    """构造子进程环境变量。"""

    environment = os.environ.copy()
    environment["ROS_SCENE_INTERSECTION_CONFLICT_TYPE"] = str(mode["conflict_type"])
    environment["ROS_SCENE_INTERSECTION_TARGET_DELTA_T_S"] = str(float(delta))
    environment["ROS_SCENE_INTERSECTION_SUCCESS_TOL_S"] = str(float(config["success_time_tol_s"]))
    return environment


def _finite(value: Any) -> float | None:
    """把 JSON 里的数值转成有限浮点数；NaN/Infinity/None 返回 None。"""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def extract_result(output_json: Path, returncode: int, elapsed_s: float) -> dict[str, Any]:
    """从输出 JSON 提取最终指标。"""

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
    except Exception as exc:  # noqa: BLE001 - 结果收集不应因单 case 失败中断
        result["error"] = f"output_json_unreadable: {exc}"
        return result

    result["status"] = data.get("status")
    result["failure_reason"] = data.get("failure_reason")
    result["completed_steps"] = data.get("completed_steps")
    result["elapsed_wall_time_s"] = data.get("elapsed_wall_time_s")

    targets = data.get("scene_targets") or []
    last_debug: dict[str, Any] | None = None
    last_actual_debug: dict[str, Any] | None = None
    for target in reversed(targets):
        debug = target.get("scene_debug")
        if not isinstance(debug, dict):
            continue
        if last_debug is None:
            last_debug = debug
        if last_actual_debug is None and _finite(debug.get("actual_entry_delta_t_s")) is not None:
            last_actual_debug = debug
        if last_debug is not None and last_actual_debug is not None:
            break

    debug = last_actual_debug or last_debug
    if debug is None:
        result["error"] = "scene_debug_missing"
        return result

    result["conflict_xy"] = debug.get("conflict_xy")
    result["conflict_angle_deg"] = _finite(debug.get("conflict_angle_deg"))
    result["conflict_type"] = debug.get("conflict_type")
    result["target_delta_t_s"] = _finite(debug.get("target_delta_t_s"))
    result["actual_entry_delta_t_s"] = _finite(debug.get("actual_entry_delta_t_s"))
    result["actual_entry_delta_t_error_s"] = _finite(debug.get("actual_entry_delta_t_error_s"))
    result["actual_entry_delta_t_result"] = debug.get("actual_entry_delta_t_result")
    result["sync_result"] = debug.get("sync_result")
    result["execution_phase"] = debug.get("execution_phase")
    result["execution_complete"] = debug.get("execution_complete")
    result["execution_end_reason"] = debug.get("execution_end_reason")
    result["success_time_tol_s"] = _finite(debug.get("success_time_tol_s"))
    result["conflict_zone_radius_m"] = _finite(debug.get("conflict_zone_radius_m"))
    return result


def format_value(value: Any, digits: int = 3) -> str:
    """把结果值格式化为表格文本。"""

    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def write_summary(output_root: Path, rows: list[dict[str, Any]], config_path: Path) -> None:
    """写 summary.json 和 summary.md。"""

    output_root.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": str(config_path),
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total": len(rows),
        "passed": sum(1 for row in rows if row.get("actual_entry_delta_t_result") == "within_tolerance"),
        "rows": rows,
    }
    with (output_root / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)

    lines = [
        "# 交叉口泛化测试汇总",
        "",
        f"- 配置：`{config_path}`",
        f"- 运行时间：{payload['generated_at']}",
        f"- 总数：{payload['total']}，within_tolerance：{payload['passed']}",
        "",
        "| case | 交互模式 | delta | status | 实测 delta | 误差 | 判定 | sync | 冲突点 | 夹角 | 步数 | 耗时 |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            "| {case_id} | {interaction} | {delta} | {status} | {actual} | {error} | {result} | {sync} | {conflict} | {angle} | {steps} | {elapsed} |".format(
                case_id=row["case_id"],
                interaction=row["interaction"],
                delta=format_value(row.get("delta"), 1),
                status=format_value(row.get("status")),
                actual=format_value(row.get("actual_entry_delta_t_s")),
                error=format_value(row.get("actual_entry_delta_t_error_s")),
                result=format_value(row.get("actual_entry_delta_t_result")),
                sync=format_value(row.get("sync_result")),
                conflict=format_value(row.get("conflict_xy")),
                angle=format_value(row.get("conflict_angle_deg"), 1),
                steps=format_value(row.get("completed_steps")),
                elapsed=format_value(row.get("elapsed_s"), 1),
            )
        )
    (output_root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    """构造命令行参数解析器。"""

    parser = argparse.ArgumentParser(description="交叉口泛化 case 批量运行器")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="case 配置 JSON")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT), help="输出根目录")
    parser.add_argument("--mode", action="append", default=[], help="只跑指定模式 id 或 name，可重复")
    parser.add_argument("--delta", action="append", type=float, default=[], help="只跑指定入区时差，可重复")
    parser.add_argument("--pilot", action="store_true", help="只跑 delta=0 的 pilot")
    parser.add_argument("--max-steps", type=int, default=None, help="覆盖配置里的 max_steps")
    parser.add_argument("--skip-gif", action="store_true", help="不生成 GIF，加快 pilot")
    parser.add_argument("--timeout", type=float, default=600.0, help="单 case 超时秒数")
    parser.add_argument("--dry-run", action="store_true", help="只打印命令，不执行")
    parser.add_argument("--fail-fast", action="store_true", help="遇到失败立即停止")
    return parser


def main() -> int:
    """脚本入口。"""

    args = build_arg_parser().parse_args()
    config_path = Path(args.config).resolve()
    config = load_config(config_path)
    modes = select_modes(config, args.mode)
    deltas = select_deltas(config, args.delta, args.pilot)
    output_root = Path(args.output_root).resolve()

    rows: list[dict[str, Any]] = []
    failures = 0

    for mode in modes:
        for delta in deltas:
            identifier = case_id(mode, delta)
            case_dir = output_root / f"{mode['id']}_{mode['name']}_d{delta:+.1f}"
            output_prefix = f"intersection_{mode['id']}_d{delta:+.1f}"
            command = build_command(
                config,
                mode,
                delta,
                case_dir,
                output_prefix,
                max_steps=args.max_steps,
                skip_gif=args.skip_gif,
            )
            environment = build_environment(config, mode, delta)

            print(f"\n=== {identifier} ===")
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
            result = extract_result(output_json, returncode, elapsed_s)
            result.update(
                {
                    "case_id": identifier,
                    "mode_id": mode["id"],
                    "mode_name": mode["name"],
                    "interaction": mode["interaction"],
                    "delta": float(delta),
                    "expected_conflict_xy": mode.get("expected_conflict_xy"),
                    "expected_conflict_angle_deg": mode.get("expected_conflict_angle_deg"),
                    "output_dir": str(case_dir),
                    "stdout_tail": stdout_tail,
                    "stderr_tail": stderr_tail,
                }
            )
            rows.append(result)

            print(
                "  status={status} actual_delta={actual} error={error} result={verdict} sync={sync} frames={frames}".format(
                    status=format_value(result.get("status")),
                    actual=format_value(result.get("actual_entry_delta_t_s")),
                    error=format_value(result.get("actual_entry_delta_t_error_s")),
                    verdict=format_value(result.get("actual_entry_delta_t_result")),
                    sync=format_value(result.get("sync_result")),
                    frames=format_value(result.get("completed_steps")),
                )
            )
            if returncode != 0 or result.get("status") != "scene_completed":
                failures += 1
                print(f"  returncode={returncode}")
                if stderr_tail.strip():
                    print("  stderr tail:")
                    print("  " + stderr_tail.strip().replace("\n", "\n  "))
                if args.fail_fast:
                    write_summary(output_root, rows, config_path)
                    return 1

    if args.dry_run:
        print(f"\n共 {len(modes) * len(deltas)} 个 case（dry-run，未执行）")
        return 0

    write_summary(output_root, rows, config_path)
    print(f"\n汇总：{output_root / 'summary.md'}")
    print(f"总数 {len(rows)}，失败 {failures}")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
