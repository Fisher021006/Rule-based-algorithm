"""仿真产物序列化与落盘辅助函数。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from offline_sim.render import plot_results, save_animation_gif
from xodr_parser.xodr_parser import XODRParser

if TYPE_CHECKING:
    from planning_runtime.scene.provider import SceneTargetProvider


def to_builtin(value: Any) -> Any:
    """递归把 numpy / 对象值转换成 JSON 安全的基础类型。"""

    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {str(key): to_builtin(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_builtin(item) for item in value]
    if hasattr(value, "__dict__"):
        return {key: to_builtin(val) for key, val in value.__dict__.items()}
    return value


def write_summary_json(output_json: Path, summary: dict[str, Any]) -> None:
    """把当前摘要写入 JSON。"""

    output_json.write_text(json.dumps(to_builtin(summary), ensure_ascii=False, indent=2), encoding="utf-8")


def write_timing_json(output_json: Path, summary: dict[str, Any]) -> None:
    """把模块耗时摘要单独写入 JSON，便于直接做可视化。"""

    payload = {
        "planner_mode": summary.get("planner_mode"),
        "scene_algorithm_symbol": summary.get("scene_algorithm_symbol"),
        "status": summary.get("status"),
        "failure_reason": summary.get("failure_reason"),
        "max_steps": summary.get("max_steps"),
        "completed_steps": summary.get("completed_steps"),
        "timing_frames": summary.get("timing_frames", []),
    }
    output_json.write_text(json.dumps(to_builtin(payload), ensure_ascii=False, indent=2), encoding="utf-8")


def persist_simulation_artifacts(
    args: argparse.Namespace,
    xodr_parser: XODRParser,
    ego_trace: list[list[float]],
    main_trace: list[list[float]] | None,
    provider: "SceneTargetProvider",
    ego_goal_xy: np.ndarray,
    summary: dict[str, Any],
    exposed_trace: list[list[float]] | None = None,
    show_main_vehicle: bool = True,
) -> dict[str, Any]:
    """为一次仿真运行落盘 PNG / GIF / JSON 产物。"""

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_png = output_dir / f"{args.output_prefix}.png"
    output_gif = None if bool(args.skip_gif) else output_dir / f"{args.output_prefix}.gif"
    output_json = output_dir / f"{args.output_prefix}.json"
    output_timing_json = output_dir / f"{args.output_prefix}_timing.json"

    summary["output_png"] = str(output_png)
    summary["output_gif"] = None if output_gif is None else str(output_gif)
    summary["output_json"] = str(output_json)
    summary["output_timing_json"] = str(output_timing_json)
    summary["gif_saved"] = False

    print(f"[render] writing PNG -> {output_png}", flush=True)
    plot_results(
        xodr_parser=xodr_parser,
        ego_trace=ego_trace,
        main_trace=main_trace,
        provider=provider,
        ego_goal_xy=ego_goal_xy,
        output_png=output_png,
        show=bool(args.show),
        planner_mode=str(args.planner_mode),
        exposed_trace=exposed_trace,
        show_main_vehicle=show_main_vehicle,
    )
    write_summary_json(output_json, summary)
    write_timing_json(output_timing_json, summary)

    if output_gif is None:
        summary["gif_saved"] = False
        summary["gif_skipped"] = True
        write_summary_json(output_json, summary)
        write_timing_json(output_timing_json, summary)
        return summary

    try:
        save_animation_gif(
            xodr_parser=xodr_parser,
            ego_trace=ego_trace,
            main_trace=main_trace,
            provider=provider,
            ego_goal_xy=ego_goal_xy,
            planner_mode=str(args.planner_mode),
            dt=float(args.dt),
            output_gif=output_gif,
            log_every=int(args.log_every),
            exposed_trace=exposed_trace,
            show_main_vehicle=show_main_vehicle,
        )
        summary["gif_saved"] = True
        summary["gif_skipped"] = False
        print(f"[render] GIF saved -> {output_gif}", flush=True)
    except KeyboardInterrupt:
        summary["gif_saved"] = False
        summary["gif_skipped"] = False
        summary["render_interrupted"] = True
        if summary.get("status") not in {"arrived", "max_steps_reached", "local_sampling_failed"}:
            summary["status"] = "interrupted"
            summary["failure_reason"] = str(summary.get("failure_reason") or "keyboard_interrupt")
        print("[render] GIF generation interrupted; PNG/JSON 已保存", flush=True)
    write_summary_json(output_json, summary)
    write_timing_json(output_timing_json, summary)
    return summary
