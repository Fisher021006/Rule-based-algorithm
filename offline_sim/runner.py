"""离线仿真器主编排循环。"""

from __future__ import annotations

import argparse
import os
import signal
import time
from pathlib import Path
from typing import Any

import numpy as np

from drivable_lanes_planner.planner import DrivableLanesExtender
from offline_sim.actors import IDMRouteVehicle, instantiate_ego_vehicle, serialize_obstacle_input
from offline_sim.debug_sampling import record_local_sampling_result
from planning_runtime.scene.map_views import initial_heading_from_line
from offline_sim.persistence import persist_simulation_artifacts
from offline_sim.render import LiveSimulationViewer
from planning_runtime.scene.provider import SceneTargetProvider
from xodr_parser.xodr_parser import XODRParser


def log_status(message: str) -> None:
    """打印仿真器状态日志，并立即刷新输出。"""

    print(message, flush=True)


def install_interrupt_handlers() -> None:
    """把常见终止信号统一转换成 `KeyboardInterrupt`。"""

    def _raise_keyboard_interrupt(signum: int, _frame: Any) -> None:
        signal_name = signal.Signals(signum).name
        raise KeyboardInterrupt(f"received {signal_name}")

    signal.signal(signal.SIGINT, _raise_keyboard_interrupt)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _raise_keyboard_interrupt)


def _build_route_actor(
    *,
    xodr_parser: XODRParser,
    ptc_id: str,
    motion_mode: str,
    start_xy: np.ndarray,
    goal_xy: np.ndarray | None,
    static_xy: np.ndarray,
    start_speed: float,
    desired_speed: float,
    lane_change_side: str = "left",
    lane_change_start_s: float = 10.0,
    lane_change_duration_s: float = 4.0,
) -> tuple[IDMRouteVehicle, np.ndarray, np.ndarray | None]:
    """按离线仿真参数创建一辆路线/静止障碍车。"""

    if str(motion_mode) == "static":
        actor_start_xy = np.asarray(static_xy, dtype=float)
        actor_goal_xy = None
        actor = IDMRouteVehicle(
            xodr_parser=xodr_parser,
            start_xy=actor_start_xy,
            goal_xy=None,
            start_speed=0.0,
            desired_speed=0.0,
            motion_mode="static",
            lane_change_side=lane_change_side,
            lane_change_start_s=lane_change_start_s,
            lane_change_duration_s=lane_change_duration_s,
        )
    else:
        actor_start_xy = np.asarray(start_xy, dtype=float)
        actor_goal_xy = np.asarray(goal_xy, dtype=float) if goal_xy is not None else None
        if actor_goal_xy is None:
            raise ValueError(f"{ptc_id} 的 {motion_mode} 模式需要 goal_xy")
        resolved_desired_speed = float(start_speed) if str(motion_mode) == "constant" else float(desired_speed)
        actor = IDMRouteVehicle(
            xodr_parser=xodr_parser,
            start_xy=actor_start_xy,
            goal_xy=actor_goal_xy,
            start_speed=float(start_speed),
            desired_speed=resolved_desired_speed,
            motion_mode=str(motion_mode),
            lane_change_side=lane_change_side,
            lane_change_start_s=lane_change_start_s,
            lane_change_duration_s=lane_change_duration_s,
        )
    actor.ptc_id = str(ptc_id)
    return actor, actor_start_xy, actor_goal_xy


def _build_route_extender(
    xodr_parser: XODRParser,
    start_xy: np.ndarray,
    goal_xy: np.ndarray | None,
    label: str,
) -> DrivableLanesExtender | None:
    """为动态障碍车构造参考线扩展器；静止车不需要。"""

    if goal_xy is None:
        return None
    extender = DrivableLanesExtender(xodr_parser)
    if not extender.plan_shortest_path(
        float(start_xy[0]),
        float(start_xy[1]),
        float(goal_xy[0]),
        float(goal_xy[1]),
    ):
        raise ValueError(f"{label} 可行驶参考线规划失败")
    return extender


def run_simulation(args: argparse.Namespace) -> dict[str, Any]:
    """完整执行一次离线仿真，并返回结果摘要。"""

    install_interrupt_handlers()
    wall_start = time.perf_counter()
    max_steps = int(args.max_steps)
    log_every = max(int(args.log_every), 1)
    log_status(
        f"[sim] start planner_mode={args.planner_mode} scene_algorithm={getattr(args, 'scene_algorithm_symbol', '-')} "
        f"main_motion={args.main_motion} exposed_enabled={bool(getattr(args, 'enable_exposed_obstacle', False))} "
        f"max_steps={max_steps} dt={float(args.dt):.2f}s"
    )
    xodr_parser = XODRParser.load_cache(str(args.map_cache))

    ego_vehicle = instantiate_ego_vehicle(
        xodr_parser,
        max_speed_kph=float(getattr(args, "ego_max_speed_kph", 25.0)),
    )
    ego_start_xy = np.asarray(args.ego_start_xy, dtype=float)
    ego_goal_xy = np.asarray(args.ego_goal_xy, dtype=float)
    ego_vehicle.global_planning(ego_start_xy, ego_goal_xy, s_v=float(args.ego_start_speed))
    ego_vehicle.x = float(ego_start_xy[0])
    ego_vehicle.y = float(ego_start_xy[1])
    ego_vehicle.heading = initial_heading_from_line(ego_vehicle.refer_lines[0])
    ego_vehicle.v = float(max(args.ego_start_speed, 0.0))
    ego_vehicle.a = 0.0
    ego_vehicle.kappa = 0.0

    main_vehicle, main_start_xy, main_goal_xy = _build_route_actor(
        xodr_parser=xodr_parser,
        ptc_id=str(getattr(args, "main_ptc_id", "main")),
        motion_mode=str(args.main_motion),
        start_xy=np.asarray(args.main_start_xy, dtype=float),
        goal_xy=np.asarray(args.main_goal_xy, dtype=float),
        static_xy=np.asarray(args.main_static_xy, dtype=float),
        start_speed=float(args.main_start_speed),
        desired_speed=float(args.main_desired_speed),
        lane_change_side=str(getattr(args, "main_lane_change_side", "left")),
        lane_change_start_s=float(getattr(args, "main_lane_change_start_s", 10.0)),
        lane_change_duration_s=float(getattr(args, "main_lane_change_duration_s", 4.0)),
    )
    main_lane_extender = _build_route_extender(
        xodr_parser=xodr_parser,
        start_xy=main_start_xy,
        goal_xy=main_goal_xy,
        label="主车",
    )

    exposed_vehicle = None
    exposed_start_xy = None
    exposed_goal_xy = None
    exposed_trace: list[list[float]] = []
    if bool(getattr(args, "enable_exposed_obstacle", False)):
        os.environ["ROS_SCENE_EXPOSED_PTC_ID"] = str(getattr(args, "exposed_ptc_id", "exposed"))
        exposed_vehicle, exposed_start_xy, exposed_goal_xy = _build_route_actor(
            xodr_parser=xodr_parser,
            ptc_id=str(getattr(args, "exposed_ptc_id", "exposed")),
            motion_mode=str(getattr(args, "exposed_motion", "static")),
            start_xy=np.asarray(args.exposed_start_xy, dtype=float),
            goal_xy=np.asarray(args.exposed_goal_xy, dtype=float),
            static_xy=np.asarray(args.exposed_static_xy, dtype=float),
            start_speed=float(args.exposed_start_speed),
            desired_speed=float(args.exposed_desired_speed),
        )

    provider = SceneTargetProvider(
        xodr_parser=xodr_parser,
        main_vehicle=main_vehicle,
        main_lane_extender=main_lane_extender,
        dt=float(args.dt),
        max_lane_segments=int(args.ref_lane_count),
        scene_algorithm_symbol=str(getattr(args, "scene_algorithm_symbol", "scenarios.scene_algorithm.scene_algorithm")),
    )
    provider.set_main_vehicle_ptc_id(str(getattr(args, "main_ptc_id", "main")))
    ego_vehicle.scene_target_provider = provider
    ego_vehicle.set_planner_mode(str(args.planner_mode))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    live_output_png = output_dir / f"{args.output_prefix}.live.png" if bool(args.live_show) else None
    viewer = LiveSimulationViewer(
        xodr_parser=xodr_parser,
        provider=provider,
        ego_goal_xy=ego_goal_xy,
        planner_mode=str(args.planner_mode),
        output_png=live_output_png,
        refresh_every=int(args.live_refresh_every),
    )
    if live_output_png is not None:
        log_status(f"[render] live view enabled -> {live_output_png}")

    ego_trace = [[ego_vehicle.x, ego_vehicle.y, ego_vehicle.heading, ego_vehicle.v, ego_vehicle.a, ego_vehicle.kappa]]
    main_trace = [[main_vehicle.x, main_vehicle.y, main_vehicle.heading, main_vehicle.v, main_vehicle.a]]
    if exposed_vehicle is not None:
        exposed_trace = [[exposed_vehicle.x, exposed_vehicle.y, exposed_vehicle.heading, exposed_vehicle.v, exposed_vehicle.a]]
    planner_frames: list[dict[str, Any]] = []
    timing_frames: list[dict[str, Any]] = []
    obstacle_input_history: list[dict[str, Any]] = []
    status = "running"
    failure_reason = ""
    viewer.update(
        ego_trace=ego_trace,
        main_trace=main_trace,
        exposed_trace=exposed_trace if exposed_vehicle is not None else None,
        step_index=-1,
        status_text="step 0 initial state",
        force=True,
    )
    try:
        for step_index in range(max_steps):
            time_s = step_index * float(args.dt)
            provider.set_runtime(step_index=step_index, time_s=time_s)
            log_status(
                f"[sim] step {step_index + 1}/{max_steps} planning "
                f"t={time_s:.1f}s ego_v={ego_vehicle.v:.2f} main_v={main_vehicle.v:.2f}"
            )
            step_wall_start = time.perf_counter()
            planner_obstacles = []
            obstacle_snapshots = []
            if not args.disable_main_obstacle:
                planner_obstacles.append(main_vehicle)
                obstacle_snapshots.append(serialize_obstacle_input(main_vehicle))
            if exposed_vehicle is not None:
                planner_obstacles.append(exposed_vehicle)
                obstacle_snapshots.append(serialize_obstacle_input(exposed_vehicle))
            obstacle_input_history.append(
                {
                    "step_index": step_index,
                    "time_s": time_s,
                    "obstacles": obstacle_snapshots,
                }
            )
            trajectory = ego_vehicle.traj_planning(planner_obstacles)
            if trajectory is None:
                failure_reason = str(ego_vehicle.traj_planner.attribute.get("failure_reason", ""))
                status = (
                    str(ego_vehicle.traj_planner.attribute.get("plan_mode", "planner_failed"))
                    if args.planner_mode == "scene_target"
                    else "local_sampling_failed"
                )
                log_status(
                    f"[sim] step {step_index + 1}/{max_steps} failed "
                    f"status={status} reason={failure_reason or '-'}"
                )
                local_timing = dict(getattr(ego_vehicle.traj_planner, "last_module_timing_ms", {}) or {})
                scene_time_ms = float(getattr(provider, "last_scene_algorithm_timing_ms", 0.0) or 0.0)
                module_timing_ms = dict(local_timing)
                module_timing_ms["scene_algorithm_ms"] = round(scene_time_ms, 3)
                module_timing_ms["end_to_end_total_ms"] = round(scene_time_ms + float(local_timing.get("plan_total_ms", 0.0)), 3)
                budget_ms = round(float(args.dt) * 1000.0, 3)
                timing_frames.append(
                    {
                        "step_index": step_index,
                        "time_s": time_s,
                        "module_timing_ms": module_timing_ms,
                        "planning_budget_ms": budget_ms,
                        "realtime_ok": bool(module_timing_ms["end_to_end_total_ms"] <= budget_ms),
                        "overrun_ms": round(max(module_timing_ms["end_to_end_total_ms"] - budget_ms, 0.0), 3),
                    }
                )
                break
            if args.planner_mode == "local_sampling":
                record_local_sampling_result(
                    ego_vehicle=ego_vehicle,
                    provider=provider,
                    planning_frame=ego_vehicle.last_planning_frame,
                    planning_input=ego_vehicle.last_planning_input,
                    trajectory=np.asarray(trajectory, dtype=float),
                )

            traj = np.asarray(trajectory, dtype=float)
            exec_index = min(max(1, int(args.advance_index)), len(traj) - 1)
            exec_point = traj[exec_index]
            ego_vehicle.x = float(exec_point[0])
            ego_vehicle.y = float(exec_point[1])
            ego_vehicle.heading = float(exec_point[2])
            ego_vehicle.v = float(exec_point[3])
            ego_vehicle.a = float(exec_point[4])
            ego_vehicle.kappa = float(exec_point[5])
            ego_vehicle.timestamp_ms = int(round((time_s + float(args.dt)) * 1000.0))
            ego_trace.append(exec_point.tolist())

            main_step_obstacles = [ego_vehicle]
            if exposed_vehicle is not None:
                main_step_obstacles.append(exposed_vehicle)
            main_vehicle.step(main_step_obstacles, dt=float(args.dt), time_s=time_s + float(args.dt))
            main_trace.append([main_vehicle.x, main_vehicle.y, main_vehicle.heading, main_vehicle.v, main_vehicle.a])
            if exposed_vehicle is not None:
                exposed_vehicle.step([ego_vehicle, main_vehicle], dt=float(args.dt), time_s=time_s + float(args.dt))
                exposed_trace.append([exposed_vehicle.x, exposed_vehicle.y, exposed_vehicle.heading, exposed_vehicle.v, exposed_vehicle.a])

            remain_dist = float(np.linalg.norm(np.array([ego_vehicle.x, ego_vehicle.y], dtype=float) - ego_goal_xy))
            local_timing = dict(getattr(ego_vehicle.traj_planner, "last_module_timing_ms", {}) or {})
            scene_time_ms = float(getattr(provider, "last_scene_algorithm_timing_ms", 0.0) or 0.0)
            module_timing_ms = dict(local_timing)
            module_timing_ms["scene_algorithm_ms"] = round(scene_time_ms, 3)
            module_timing_ms["end_to_end_total_ms"] = round(scene_time_ms + float(local_timing.get("plan_total_ms", 0.0)), 3)
            budget_ms = round(float(args.dt) * 1000.0, 3)
            timing_frames.append(
                {
                    "step_index": step_index,
                    "time_s": time_s,
                    "module_timing_ms": module_timing_ms,
                    "planning_budget_ms": budget_ms,
                    "realtime_ok": bool(module_timing_ms["end_to_end_total_ms"] <= budget_ms),
                    "overrun_ms": round(max(module_timing_ms["end_to_end_total_ms"] - budget_ms, 0.0), 3),
                }
            )
            planner_frames.append(
                {
                    "step_index": step_index,
                    "time_s": time_s,
                    "ego": {
                        "x": ego_vehicle.x,
                        "y": ego_vehicle.y,
                        "heading": ego_vehicle.heading,
                        "v": ego_vehicle.v,
                        "a": ego_vehicle.a,
                        "kappa": ego_vehicle.kappa,
                    },
                    "main": {
                        "x": main_vehicle.x,
                        "y": main_vehicle.y,
                        "heading": main_vehicle.heading,
                        "v": main_vehicle.v,
                        "a": main_vehicle.a,
                    },
                    "exposed": None if exposed_vehicle is None else {
                        "x": exposed_vehicle.x,
                        "y": exposed_vehicle.y,
                        "heading": exposed_vehicle.heading,
                        "v": exposed_vehicle.v,
                        "a": exposed_vehicle.a,
                    },
                    "planner_attr": dict(ego_vehicle.traj_planner.attribute),
                    "module_timing_ms": module_timing_ms,
                    "planning_budget_ms": budget_ms,
                    "realtime_ok": bool(module_timing_ms["end_to_end_total_ms"] <= budget_ms),
                    "overrun_ms": round(max(module_timing_ms["end_to_end_total_ms"] - budget_ms, 0.0), 3),
                    "obstacle_inputs": obstacle_snapshots,
                    "remain_dist": remain_dist,
                }
            )
            viewer.update(
                ego_trace=ego_trace,
                main_trace=main_trace,
                exposed_trace=exposed_trace if exposed_vehicle is not None else None,
                step_index=step_index,
                planned_trajectory=traj.tolist(),
                status_text=(
                    f"step {step_index + 1}/{max_steps}  "
                    f"remain={remain_dist:.2f}m  "
                    f"ego_v={ego_vehicle.v:.2f}m/s  "
                    f"main_v={main_vehicle.v:.2f}m/s"
                ),
            )

            if step_index == 0 or (step_index + 1) % log_every == 0:
                log_status(
                    f"[sim] step {step_index + 1}/{max_steps} done "
                    f"remain={remain_dist:.2f}m ego_v={ego_vehicle.v:.2f} "
                    f"wall={time.perf_counter() - step_wall_start:.2f}s"
                )

            latest_scene_status = (
                str(provider.history[-1].get("scene_status", ""))
                if provider.history
                else ""
            )
            if latest_scene_status == "brake_completed_stop":
                status = "scene_completed"
                log_status(
                    f"[sim] scene completed at step {step_index + 1}/{max_steps} "
                    f"scene_status={latest_scene_status} ego_v={ego_vehicle.v:.2f}"
                )
                break
            if latest_scene_status in {"cutout_success", "cutout_metric_miss", "cutout_hard_metric_fail"}:
                status = "scene_completed" if latest_scene_status == "cutout_success" else latest_scene_status
                log_status(
                    f"[sim] cut-out scene finished at step {step_index + 1}/{max_steps} "
                    f"scene_status={latest_scene_status} ego_v={ego_vehicle.v:.2f}"
                )
                break
            latest_control_mode = (
                str(provider.history[-1].get("control_mode", ""))
                if provider.history
                else ""
            )
            if (
                latest_scene_status == "complete"
                and latest_control_mode == "emergency_stop"
                and ego_vehicle.v <= float(args.arrive_speed)
            ):
                status = "scene_completed"
                log_status(
                    f"[sim] blind-spot scene completed at step {step_index + 1}/{max_steps} "
                    f"scene_status={latest_scene_status} ego_v={ego_vehicle.v:.2f}"
                )
                break

            if remain_dist <= float(args.arrive_dist) and ego_vehicle.v <= float(args.arrive_speed):
                status = "arrived"
                log_status(
                    f"[sim] arrived at step {step_index + 1}/{max_steps} "
                    f"remain={remain_dist:.2f}m ego_v={ego_vehicle.v:.2f}"
                )
                break
        else:
            status = "max_steps_reached"
            log_status(f"[sim] reached max_steps={max_steps}")
    except KeyboardInterrupt as exc:
        status = "interrupted"
        failure_reason = str(exc) or "keyboard_interrupt"
        log_status(
            f"[sim] interrupted after completed_steps={max(len(ego_trace) - 1, 0)}; "
            "persisting partial outputs"
        )

    summary = {
        "timing_frame_count": len(timing_frames),
        "avg_plan_total_ms": round(
            float(np.mean([frame["module_timing_ms"]["end_to_end_total_ms"] for frame in timing_frames])),
            3,
        ) if timing_frames else 0.0,
        "max_plan_total_ms": round(
            float(np.max([frame["module_timing_ms"]["end_to_end_total_ms"] for frame in timing_frames])),
            3,
        ) if timing_frames else 0.0,
        "realtime_ok_rate": round(
            float(sum(1 for frame in timing_frames if frame["realtime_ok"]) / len(timing_frames)),
            6,
        ) if timing_frames else 0.0,
        "planner_mode": str(args.planner_mode),
        "scene_algorithm_symbol": str(getattr(args, "scene_algorithm_symbol", "scenarios.scene_algorithm.scene_algorithm")),
        "status": status,
        "failure_reason": failure_reason,
        "map_cache": str(args.map_cache),
        "ego_start_xy": list(map(float, args.ego_start_xy)),
        "ego_goal_xy": list(map(float, args.ego_goal_xy)),
        "ego_max_speed_kph": float(getattr(args, "ego_max_speed_kph", 25.0)),
        "main_start_xy": None if main_start_xy is None else list(map(float, np.asarray(main_start_xy, dtype=float))),
        "main_goal_xy": None if main_goal_xy is None else list(map(float, np.asarray(main_goal_xy, dtype=float))),
        "main_static_xy": None if str(args.main_motion) != "static" else list(map(float, args.main_static_xy)),
        "main_motion": str(args.main_motion),
        "main_lane_change_side": str(getattr(args, "main_lane_change_side", "left")),
        "main_lane_change_start_s": float(getattr(args, "main_lane_change_start_s", 10.0)),
        "main_lane_change_duration_s": float(getattr(args, "main_lane_change_duration_s", 4.0)),
        "main_ptc_id": str(getattr(args, "main_ptc_id", "main")),
        "enable_exposed_obstacle": bool(exposed_vehicle is not None),
        "exposed_ptc_id": str(getattr(args, "exposed_ptc_id", "")) if exposed_vehicle is not None else "",
        "exposed_start_xy": None if exposed_start_xy is None else list(map(float, np.asarray(exposed_start_xy, dtype=float))),
        "exposed_goal_xy": None if exposed_goal_xy is None else list(map(float, np.asarray(exposed_goal_xy, dtype=float))),
        "exposed_static_xy": None if exposed_vehicle is None or str(getattr(args, "exposed_motion", "")) != "static" else list(map(float, args.exposed_static_xy)),
        "exposed_motion": "" if exposed_vehicle is None else str(getattr(args, "exposed_motion", "")),
        "disable_main_obstacle": bool(args.disable_main_obstacle),
        "max_steps": max_steps,
        "completed_steps": max(len(ego_trace) - 1, 0),
        "elapsed_wall_time_s": round(time.perf_counter() - wall_start, 3),
        "ego_trace": ego_trace,
        "main_trace": main_trace,
        "exposed_trace": exposed_trace,
        "scene_targets": provider.history,
        "planner_debug_markers": provider.history,
        "obstacle_input_history": obstacle_input_history,
        "planner_frames": planner_frames,
        "timing_frames": timing_frames,
        "output_live_png": None if live_output_png is None else str(live_output_png),
    }
    try:
        return persist_simulation_artifacts(
            args=args,
            xodr_parser=xodr_parser,
            ego_trace=ego_trace,
            main_trace=main_trace,
            provider=provider,
            ego_goal_xy=ego_goal_xy,
            summary=summary,
            exposed_trace=exposed_trace if exposed_vehicle is not None else None,
        )
    finally:
        viewer.close(block=False)
