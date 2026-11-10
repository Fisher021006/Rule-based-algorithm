"""多算法车辆离线仿真编排器。"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from offline_sim.actors import instantiate_ego_vehicle, serialize_obstacle_input
from offline_sim.multi_render import (
    MultiVehicleViewer,
    render_multi_vehicle_frame,
    save_multi_vehicle_gif,
)
from offline_sim.persistence import write_summary_json
from offline_sim.runner import (
    _build_route_actor,
    _build_route_extender,
    _format_step_label,
    _normalise_max_steps,
    install_interrupt_handlers,
    log_status,
)
from planning_runtime.scene.algorithm_loader import (
    load_isolated_scene_algorithm,
    scoped_scene_environment,
)
from planning_runtime.scene.map_views import initial_heading_from_line
from planning_runtime.scene.provider import SceneTargetProvider
from xodr_parser.xodr_parser import XODRParser


ALGORITHM_KINDS = {"algorithm", "scene_algorithm"}
ROUTE_KINDS = {"route", "scripted", "obstacle"}
ROUTE_MOTIONS = {"idm", "constant", "static", "scripted_lane_change"}
DEFAULT_TERMINAL_SCENE_STATUSES = {
    "brake_completed_stop",
    "cutin_success",
    "cutin_metric_miss",
    "cutin_metric_unavailable",
    "cutout_success",
    "cutout_metric_miss",
    "cutout_hard_metric_fail",
    "complete",
    "complete_thw_miss",
    "intersection_sync_missed",
    "scene_completed",
    "degraded_exit",
    "fail_safe",
}


@dataclass(frozen=True)
class MultiVehicleSpec:
    """单辆离线仿真车的配置。"""

    vehicle_id: str
    kind: str
    start_xy: tuple[float, float]
    goal_xy: tuple[float, float] | None
    static_xy: tuple[float, float] | None
    start_speed: float
    desired_speed: float
    max_speed_kph: float
    motion: str
    algorithm_symbol: str
    main_ptc_id: str
    environment: dict[str, Any] = field(default_factory=dict)
    lane_change_side: str = "left"
    lane_change_start_s: float = 10.0
    lane_change_duration_s: float = 4.0

    @property
    def is_algorithm(self) -> bool:
        """是否为运行场景算法的 GVT。"""

        return self.kind == "algorithm"


@dataclass(frozen=True)
class MultiVehicleConfig:
    """多车仿真顶层配置。"""

    vehicles: tuple[MultiVehicleSpec, ...]
    primary_vehicle_id: str
    stop_on_terminal_vehicle_ids: tuple[str, ...]
    stop_on_terminal_mode: str
    terminal_scene_statuses: frozenset[str]


@dataclass
class VehicleRuntime:
    """单辆车在多车运行器中的运行期对象。"""

    spec: MultiVehicleSpec
    actor: Any
    route_extender: Any | None
    provider: SceneTargetProvider | None = None
    trace: list[list[float]] = field(default_factory=list)
    timing_frames: list[dict[str, Any]] = field(default_factory=list)


def _parse_xy(raw_value: Any, field_name: str, *, required: bool) -> tuple[float, float] | None:
    """解析 JSON 中的二维坐标。"""

    if raw_value is None:
        if required:
            raise ValueError(f"{field_name} 必须配置")
        return None
    if not isinstance(raw_value, (list, tuple)) or len(raw_value) != 2:
        raise ValueError(f"{field_name} 必须是 [x, y]")
    return float(raw_value[0]), float(raw_value[1])


def _normalize_vehicle_spec(raw: Any, index: int) -> MultiVehicleSpec:
    """规整并校验一条车辆配置。"""

    if not isinstance(raw, dict):
        raise ValueError(f"vehicles[{index}] 必须是 JSON object")
    vehicle_id = str(raw.get("id", raw.get("vehicle_id", "")) or "").strip()
    if not vehicle_id:
        raise ValueError(f"vehicles[{index}].id 不能为空")
    raw_kind = str(raw.get("kind", "") or "").strip().lower()
    if raw_kind in ALGORITHM_KINDS:
        kind = "algorithm"
    elif raw_kind in ROUTE_KINDS:
        kind = "route"
    else:
        raise ValueError(f"{vehicle_id}.kind 必须是 algorithm 或 route")

    motion = str(raw.get("motion", "idm") or "idm").strip().lower()
    if kind == "route" and motion not in ROUTE_MOTIONS:
        raise ValueError(f"{vehicle_id}.motion 不支持: {motion}")
    static_xy = _parse_xy(raw.get("static_xy"), f"{vehicle_id}.static_xy", required=False)
    start_required = not (kind == "route" and motion == "static" and static_xy is not None)
    start_xy = _parse_xy(raw.get("start_xy"), f"{vehicle_id}.start_xy", required=start_required)
    if start_xy is None:
        start_xy = static_xy
    assert start_xy is not None
    goal_xy = _parse_xy(
        raw.get("goal_xy"),
        f"{vehicle_id}.goal_xy",
        required=bool(kind == "algorithm" or motion != "static"),
    )
    algorithm_symbol = str(
        raw.get("algorithm", raw.get("scene_algorithm_symbol", "")) or ""
    ).strip()
    main_ptc_id = str(raw.get("main_ptc_id", "") or "").strip()
    if kind == "algorithm" and not algorithm_symbol:
        raise ValueError(f"{vehicle_id}.algorithm 不能为空")
    if kind == "algorithm" and not main_ptc_id:
        raise ValueError(f"{vehicle_id}.main_ptc_id 不能为空")
    environment = raw.get("environment", raw.get("env", {}))
    if not isinstance(environment, dict):
        raise ValueError(f"{vehicle_id}.environment 必须是 JSON object")

    return MultiVehicleSpec(
        vehicle_id=vehicle_id,
        kind=kind,
        start_xy=start_xy,
        goal_xy=goal_xy,
        static_xy=static_xy,
        start_speed=max(0.0, float(raw.get("start_speed", 0.0))),
        desired_speed=max(0.0, float(raw.get("desired_speed", raw.get("start_speed", 0.0)))),
        max_speed_kph=max(0.1, float(raw.get("max_speed_kph", 25.0))),
        motion=motion,
        algorithm_symbol=algorithm_symbol,
        main_ptc_id=main_ptc_id,
        environment={str(key): value for key, value in environment.items()},
        lane_change_side=str(raw.get("lane_change_side", "left") or "left").strip().lower(),
        lane_change_start_s=max(0.0, float(raw.get("lane_change_start_s", 10.0))),
        lane_change_duration_s=max(0.1, float(raw.get("lane_change_duration_s", 4.0))),
    )


def load_multi_vehicle_config(path: str | Path) -> MultiVehicleConfig:
    """读取并校验多车 JSON 配置。"""

    config_path = Path(path).expanduser().resolve()
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("多车配置顶层必须是 JSON object")
    raw_vehicles = payload.get("vehicles")
    if not isinstance(raw_vehicles, list) or not raw_vehicles:
        raise ValueError("多车配置必须包含非空 vehicles 列表")
    vehicles = tuple(_normalize_vehicle_spec(raw, index) for index, raw in enumerate(raw_vehicles))
    vehicle_ids = [spec.vehicle_id for spec in vehicles]
    if len(set(vehicle_ids)) != len(vehicle_ids):
        raise ValueError("多车配置中的车辆 id 必须唯一")
    algorithm_ids = [spec.vehicle_id for spec in vehicles if spec.is_algorithm]
    if not algorithm_ids:
        raise ValueError("多车配置至少需要一辆 algorithm 车")
    known_ids = set(vehicle_ids)
    for spec in vehicles:
        if spec.is_algorithm and spec.main_ptc_id not in known_ids:
            raise ValueError(f"{spec.vehicle_id}.main_ptc_id 不存在: {spec.main_ptc_id}")
        if spec.is_algorithm and spec.main_ptc_id == spec.vehicle_id:
            raise ValueError(f"{spec.vehicle_id}.main_ptc_id 不能指向自身")

    primary_vehicle_id = str(payload.get("primary_vehicle_id", algorithm_ids[0]) or "").strip()
    if primary_vehicle_id not in algorithm_ids:
        raise ValueError("primary_vehicle_id 必须指向 algorithm 车")
    raw_stop_ids = payload.get("stop_on_terminal_vehicle_ids", [])
    if not isinstance(raw_stop_ids, list):
        raise ValueError("stop_on_terminal_vehicle_ids 必须是列表")
    stop_ids = tuple(str(item) for item in raw_stop_ids)
    unknown_stop_ids = sorted(set(stop_ids) - set(algorithm_ids))
    if unknown_stop_ids:
        raise ValueError(f"终止车辆必须是 algorithm 车: {unknown_stop_ids}")
    stop_mode = str(payload.get("stop_on_terminal_mode", "all") or "all").strip().lower()
    if stop_mode not in {"all", "any"}:
        raise ValueError("stop_on_terminal_mode 必须是 all 或 any")
    raw_terminal_statuses = payload.get(
        "terminal_scene_statuses",
        sorted(DEFAULT_TERMINAL_SCENE_STATUSES),
    )
    if not isinstance(raw_terminal_statuses, list):
        raise ValueError("terminal_scene_statuses 必须是列表")
    terminal_statuses = {str(item) for item in raw_terminal_statuses}
    return MultiVehicleConfig(
        vehicles=vehicles,
        primary_vehicle_id=primary_vehicle_id,
        stop_on_terminal_vehicle_ids=stop_ids,
        stop_on_terminal_mode=stop_mode,
        terminal_scene_statuses=frozenset(terminal_statuses),
    )


def _initial_trace_point(actor: Any) -> list[float]:
    """将车辆当前状态转为统一轨迹点。"""

    return [
        float(actor.x),
        float(actor.y),
        float(actor.heading),
        float(actor.v),
        float(actor.a),
        float(getattr(actor, "kappa", 0.0)),
    ]


def _snapshot_actor(actor: Any) -> SimpleNamespace:
    """冻结车辆的单帧 OBU/障碍物状态。"""

    return SimpleNamespace(**serialize_obstacle_input(actor))


def _advance_algorithm_vehicle(vehicle: Any, trajectory: np.ndarray, dt: float, advance_index: int, time_s: float) -> None:
    """执行算法车轨迹中的指定预览点。"""

    exec_index = min(max(1, int(advance_index)), len(trajectory) - 1)
    point = trajectory[exec_index]
    previous_v = float(vehicle.v)
    executed_dt = max(float(dt) * float(exec_index), 1e-3)
    vehicle.x = float(point[0])
    vehicle.y = float(point[1])
    vehicle.heading = float(point[2])
    vehicle.v = float(point[3])
    vehicle.a = float((float(point[3]) - previous_v) / executed_dt)
    vehicle.kappa = float(point[5])
    vehicle.timestamp_ms = int(round((float(time_s) + float(dt)) * 1000.0))


def _create_runtimes(
    config: MultiVehicleConfig,
    parser: XODRParser,
    args: argparse.Namespace,
) -> dict[str, VehicleRuntime]:
    """创建所有车辆，然后为每辆算法车绑定独立 provider。"""

    runtimes: dict[str, VehicleRuntime] = {}
    for spec in config.vehicles:
        start_xy = np.asarray(spec.start_xy, dtype=float)
        goal_xy = None if spec.goal_xy is None else np.asarray(spec.goal_xy, dtype=float)
        if spec.is_algorithm:
            actor = instantiate_ego_vehicle(
                parser,
                max_speed_kph=spec.max_speed_kph,
                scene_algorithm_symbol=None,
            )
            actor.ptc_id = spec.vehicle_id
            actor.ptcId = spec.vehicle_id
            assert goal_xy is not None
            actor.global_planning(start_xy, goal_xy, s_v=spec.start_speed)
            actor.x = float(start_xy[0])
            actor.y = float(start_xy[1])
            actor.heading = initial_heading_from_line(actor.refer_lines[0])
            actor.v = float(spec.start_speed)
            actor.a = 0.0
            actor.kappa = 0.0
            route_extender = _build_route_extender(parser, start_xy, goal_xy, spec.vehicle_id)
        else:
            static_xy = np.asarray(spec.static_xy or spec.start_xy, dtype=float)
            actor, actor_start, actor_goal = _build_route_actor(
                xodr_parser=parser,
                ptc_id=spec.vehicle_id,
                motion_mode=spec.motion,
                start_xy=start_xy,
                goal_xy=goal_xy,
                static_xy=static_xy,
                start_speed=spec.start_speed,
                desired_speed=spec.desired_speed,
                lane_change_side=spec.lane_change_side,
                lane_change_start_s=spec.lane_change_start_s,
                lane_change_duration_s=spec.lane_change_duration_s,
            )
            route_extender = _build_route_extender(parser, actor_start, actor_goal, spec.vehicle_id)
        runtimes[spec.vehicle_id] = VehicleRuntime(
            spec=spec,
            actor=actor,
            route_extender=route_extender,
            trace=[_initial_trace_point(actor)],
        )

    for runtime in runtimes.values():
        if not runtime.spec.is_algorithm:
            continue
        main_runtime = runtimes[runtime.spec.main_ptc_id]
        main_route_extender = _build_route_extender(
            parser,
            np.asarray(main_runtime.spec.start_xy, dtype=float),
            None
            if main_runtime.spec.goal_xy is None
            else np.asarray(main_runtime.spec.goal_xy, dtype=float),
            f"{runtime.spec.vehicle_id} 的主车 {main_runtime.spec.vehicle_id}",
        )
        scene_algorithm = load_isolated_scene_algorithm(
            runtime.spec.algorithm_symbol,
            instance_key=runtime.spec.vehicle_id,
            environment=runtime.spec.environment,
        )
        provider = SceneTargetProvider(
            xodr_parser=parser,
            main_vehicle=main_runtime.actor,
            main_lane_extender=main_route_extender,
            dt=float(args.dt),
            max_lane_segments=int(args.ref_lane_count),
            scene_algorithm_symbol=runtime.spec.algorithm_symbol,
            scene_algorithm_callable=scene_algorithm,
        )
        provider.set_main_vehicle_ptc_id(runtime.spec.main_ptc_id)
        provider.set_main_route_points(
            np.asarray(main_runtime.spec.start_xy, dtype=float),
            None if main_runtime.spec.goal_xy is None else np.asarray(main_runtime.spec.goal_xy, dtype=float),
        )
        runtime.provider = provider
        runtime.actor.scene_target_provider = provider
        runtime.actor.set_planner_mode("scene_target")
    return runtimes


def _latest_scene_status(runtime: VehicleRuntime) -> str:
    """返回算法车最近一帧场景状态。"""

    if runtime.provider is None or not runtime.provider.history:
        return ""
    return str(runtime.provider.history[-1].get("scene_status", ""))


def _stop_policy_reached(config: MultiVehicleConfig, runtimes: dict[str, VehicleRuntime]) -> bool:
    """判断配置的场景终止车是否已进入终态。"""

    if not config.stop_on_terminal_vehicle_ids:
        return False
    hits = [
        _latest_scene_status(runtimes[vehicle_id]) in config.terminal_scene_statuses
        for vehicle_id in config.stop_on_terminal_vehicle_ids
    ]
    return any(hits) if config.stop_on_terminal_mode == "any" else all(hits)


def _vehicle_metadata(runtimes: dict[str, VehicleRuntime]) -> dict[str, dict[str, Any]]:
    """生成可视化和结果文件使用的车辆元数据。"""

    return {
        vehicle_id: {
            "id": vehicle_id,
            "kind": runtime.spec.kind,
            "algorithm": runtime.spec.algorithm_symbol,
            "main_ptc_id": runtime.spec.main_ptc_id,
            "motion": runtime.spec.motion,
            "start_xy": list(runtime.spec.start_xy),
            "goal_xy": None if runtime.spec.goal_xy is None else list(runtime.spec.goal_xy),
            "length": float(getattr(runtime.actor, "length", 4.6)),
            "width": float(getattr(runtime.actor, "width", 1.8)),
        }
        for vehicle_id, runtime in runtimes.items()
    }


def run_multi_vehicle_simulation(args: argparse.Namespace) -> dict[str, Any]:
    """根据 JSON 配置运行多场景算法车辆仿真。"""

    install_interrupt_handlers()
    wall_start = time.perf_counter()
    config_path = Path(str(args.multi_vehicle_config)).expanduser().resolve()
    config = load_multi_vehicle_config(config_path)
    max_steps = _normalise_max_steps(getattr(args, "max_steps", None))
    if max_steps is None and not config.stop_on_terminal_vehicle_ids:
        raise ValueError("多车仿真需配置 --max-steps 或 stop_on_terminal_vehicle_ids")
    parser = XODRParser.load_cache(str(args.map_cache))
    runtimes = _create_runtimes(config, parser, args)
    algorithm_runtimes = {
        vehicle_id: runtime
        for vehicle_id, runtime in runtimes.items()
        if runtime.spec.is_algorithm
    }
    route_runtimes = {
        vehicle_id: runtime
        for vehicle_id, runtime in runtimes.items()
        if not runtime.spec.is_algorithm
    }
    traces = {vehicle_id: runtime.trace for vehicle_id, runtime in runtimes.items()}
    metadata = _vehicle_metadata(runtimes)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    live_output_png = output_dir / f"{args.output_prefix}.live.png" if bool(args.live_show) else None
    viewer = MultiVehicleViewer(
        xodr_parser=parser,
        vehicle_metadata=metadata,
        output_png=live_output_png,
        refresh_every=int(args.live_refresh_every),
    )
    viewer.update(traces=traces, step_index=-1, status_text="step 0 initial state", force=True)

    timing_frames: list[dict[str, Any]] = []
    planner_frames: list[dict[str, Any]] = []
    obstacle_input_history: list[dict[str, Any]] = []
    status = "running"
    failure_reason = ""
    step_index = 0
    log_every = max(int(args.log_every), 1)
    log_status(
        f"[multi-sim] start vehicles={list(runtimes)} algorithms={list(algorithm_runtimes)} "
        f"step_limit={max_steps if max_steps is not None else 'terminal-policy'}"
    )
    try:
        while True:
            if max_steps is not None and step_index >= max_steps:
                status = "max_steps_reached"
                break
            step_label = _format_step_label(step_index, max_steps)
            time_s = float(step_index) * float(args.dt)
            # 本帧所有算法和路线车只读这份冻结状态，不受规划顺序影响。
            snapshots = {
                vehicle_id: _snapshot_actor(runtime.actor)
                for vehicle_id, runtime in runtimes.items()
            }
            obstacle_input_history.append(
                {
                    "step_index": step_index,
                    "time_s": time_s,
                    "vehicles": {
                        vehicle_id: serialize_obstacle_input(snapshot)
                        for vehicle_id, snapshot in snapshots.items()
                    },
                }
            )
            planned_trajectories: dict[str, np.ndarray] = {}
            vehicle_timing: dict[str, dict[str, Any]] = {}
            step_failed = False
            for vehicle_id, runtime in algorithm_runtimes.items():
                assert runtime.provider is not None
                runtime.provider.set_runtime(step_index=step_index, time_s=time_s)
                obstacles = [
                    snapshot
                    for other_id, snapshot in snapshots.items()
                    if other_id != vehicle_id
                ]
                with scoped_scene_environment(runtime.spec.environment):
                    trajectory = runtime.actor.traj_planning(obstacles)
                scene_time_ms = float(runtime.provider.last_scene_algorithm_timing_ms or 0.0)
                local_timing = dict(
                    getattr(runtime.actor.traj_planner, "last_module_timing_ms", {}) or {}
                )
                module_timing = dict(local_timing)
                module_timing["scene_algorithm_ms"] = round(scene_time_ms, 3)
                module_timing["end_to_end_total_ms"] = round(
                    scene_time_ms + float(local_timing.get("plan_total_ms", 0.0)),
                    3,
                )
                vehicle_timing[vehicle_id] = module_timing
                if trajectory is None:
                    status = "planner_failed"
                    failure_reason = (
                        f"{vehicle_id}:"
                        f"{runtime.actor.traj_planner.attribute.get('failure_reason', 'unknown')}"
                    )
                    step_failed = True
                    break
                planned_trajectories[vehicle_id] = np.asarray(trajectory, dtype=float)
            total_plan_ms = float(
                sum(item.get("end_to_end_total_ms", 0.0) for item in vehicle_timing.values())
            )
            timing_frames.append(
                {
                    "step_index": step_index,
                    "time_s": time_s,
                    "vehicles": vehicle_timing,
                    "sequential_total_ms": round(total_plan_ms, 3),
                    "planning_budget_ms": round(float(args.dt) * 1000.0, 3),
                    "realtime_ok": bool(total_plan_ms <= float(args.dt) * 1000.0),
                }
            )
            if step_failed:
                break

            # 路线车同样只使用本帧冻结障碍物；计算完后才统一提交算法车状态。
            for vehicle_id, runtime in route_runtimes.items():
                obstacles = [
                    snapshot
                    for other_id, snapshot in snapshots.items()
                    if other_id != vehicle_id
                ]
                runtime.actor.step(obstacles, dt=float(args.dt), time_s=time_s + float(args.dt))
            for vehicle_id, runtime in algorithm_runtimes.items():
                _advance_algorithm_vehicle(
                    runtime.actor,
                    planned_trajectories[vehicle_id],
                    dt=float(args.dt),
                    advance_index=int(args.advance_index),
                    time_s=time_s,
                )
            for runtime in runtimes.values():
                runtime.trace.append(_initial_trace_point(runtime.actor))

            statuses = {
                vehicle_id: _latest_scene_status(runtime)
                for vehicle_id, runtime in algorithm_runtimes.items()
            }
            planner_frames.append(
                {
                    "step_index": step_index,
                    "time_s": time_s,
                    "vehicles": {
                        vehicle_id: serialize_obstacle_input(runtime.actor)
                        for vehicle_id, runtime in runtimes.items()
                    },
                    "scene_statuses": statuses,
                    "planner_attributes": {
                        vehicle_id: dict(runtime.actor.traj_planner.attribute)
                        for vehicle_id, runtime in algorithm_runtimes.items()
                    },
                }
            )
            viewer.update(
                traces=traces,
                step_index=step_index,
                planned_trajectories=planned_trajectories,
                status_text=f"step {step_label}  statuses={statuses}",
            )
            if step_index == 0 or (step_index + 1) % log_every == 0:
                speeds = {
                    vehicle_id: round(float(runtime.actor.v), 2)
                    for vehicle_id, runtime in runtimes.items()
                }
                log_status(
                    f"[multi-sim] step {step_label} speeds={speeds} statuses={statuses}"
                )
            if _stop_policy_reached(config, runtimes):
                status = "scene_terminal_policy_reached"
                break
            step_index += 1
    except KeyboardInterrupt as exc:
        status = "interrupted"
        failure_reason = str(exc) or "keyboard_interrupt"

    output_png = output_dir / f"{args.output_prefix}.png"
    output_gif = None if bool(args.skip_gif) else output_dir / f"{args.output_prefix}.gif"
    output_json = output_dir / f"{args.output_prefix}.json"
    output_timing_json = output_dir / f"{args.output_prefix}_timing.json"
    scene_targets = {
        vehicle_id: [] if runtime.provider is None else runtime.provider.history
        for vehicle_id, runtime in algorithm_runtimes.items()
    }
    final_scene_statuses = {
        vehicle_id: _latest_scene_status(runtime)
        for vehicle_id, runtime in algorithm_runtimes.items()
    }
    summary = {
        "multi_vehicle": True,
        "multi_vehicle_config": str(config_path),
        "primary_vehicle_id": config.primary_vehicle_id,
        "status": status,
        "failure_reason": failure_reason,
        "map_cache": str(args.map_cache),
        "dt": float(args.dt),
        "max_steps": max_steps,
        "completed_steps": max((len(trace) - 1 for trace in traces.values()), default=0),
        "elapsed_wall_time_s": round(time.perf_counter() - wall_start, 3),
        "vehicle_metadata": metadata,
        "vehicle_traces": traces,
        "scene_targets": scene_targets,
        "final_scene_statuses": final_scene_statuses,
        "planner_frames": planner_frames,
        "obstacle_input_history": obstacle_input_history,
        "timing_frames": timing_frames,
        "output_png": str(output_png),
        "output_gif": None if output_gif is None else str(output_gif),
        "output_json": str(output_json),
        "output_timing_json": str(output_timing_json),
        "output_live_png": None if live_output_png is None else str(live_output_png),
    }
    try:
        image = render_multi_vehicle_frame(
            xodr_parser=parser,
            traces=traces,
            vehicle_metadata=metadata,
            frame_index=max((len(trace) for trace in traces.values()), default=1) - 1,
            status_text=f"status={status}",
        )
        output_png.parent.mkdir(parents=True, exist_ok=True)
        image.save(output_png)
        write_summary_json(output_json, summary)
        write_summary_json(
            output_timing_json,
            {
                "multi_vehicle": True,
                "status": status,
                "failure_reason": failure_reason,
                "timing_frames": timing_frames,
            },
        )
        if output_gif is not None:
            save_multi_vehicle_gif(
                xodr_parser=parser,
                traces=traces,
                vehicle_metadata=metadata,
                dt=float(args.dt),
                output_gif=output_gif,
                log_every=log_every,
            )
        return summary
    finally:
        viewer.close(block=bool(args.show))
