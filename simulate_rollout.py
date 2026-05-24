"""离线滚动仿真脚本。

用途：
1. 在给定地图、起点、终点下先做一次全局路径规划。
2. 之后不断调用局部规划器，模拟车辆沿局部轨迹前进。
3. 将多帧局部规划拼接成一条完整执行轨迹。
4. 输出轨迹 JSON、调试信息和可视化图片。
"""

import argparse
from collections import Counter
import contextlib
import io
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from simulator.vehicle import GVT
from xodr_parser.xodr_parser import XODRParser


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CACHE_PATH = PROJECT_ROOT / "xodr_parser" / "cache" / "同济测试场.pkl"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "manual_runs"


def build_arg_parser():
    parser = argparse.ArgumentParser(description="沿局部规划滚动仿真，生成完整执行轨迹")
    parser.add_argument("--map-cache", default=str(DEFAULT_CACHE_PATH), help="地图缓存 .pkl 路径")
    parser.add_argument("--start-xy", nargs=2, type=float, required=True, metavar=("X", "Y"))
    parser.add_argument("--goal-xy", nargs=2, type=float, required=True, metavar=("X", "Y"))
    parser.add_argument("--advance-index", type=int, default=1, help="每轮执行局部轨迹中的第几个点；默认 1 表示 0.1s 重规划一次")
    parser.add_argument("--max-steps", type=int, default=400, help="最大滚动步数")
    parser.add_argument("--arrive-dist", type=float, default=1.5, help="到达判定距离阈值")
    parser.add_argument("--arrive-speed", type=float, default=0.3, help="到达判定速度阈值")
    parser.add_argument("--replan-dt", type=float, default=0.1, help="每次重规划对应的离线时间步长，默认 0.1s")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="输出目录")
    parser.add_argument("--output-prefix", default="rollout", help="输出文件名前缀")
    parser.add_argument("--verbose-planner", action="store_true", help="打印规划器内部调试输出")
    return parser


def instantiate_vehicle(xodr_parser):
    outline = {
        "height": 4640.0,
        "width": 1780.0,
        "wheelbase": 2700.0,
    }
    sport_params = {
        "maxSpeed": 25.0,
        "maxAcc": 3.0,
        "minAcc": 2.0,
    }
    return GVT("manual", outline, sport_params, xodr_parser)


def to_builtin(value):
    """将 numpy / 复杂对象递归转换为 JSON 可序列化的普通 Python 类型。"""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): to_builtin(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_builtin(v) for v in value]
    return value


def snapshot_vehicle_state(gvt):
    return {
        "x": float(gvt.x),
        "y": float(gvt.y),
        "heading": float(gvt.heading),
        "v": float(gvt.v),
        "a": float(gvt.a),
        "kappa": float(gvt.kappa),
    }


def serialize_behavior_memory(memory):
    return {
        "mode": memory.mode.value,
        "target_lane_id": memory.target_lane_id,
        "target_line_index": int(memory.target_line_index),
        "lane_change_committed": bool(memory.lane_change_committed),
        "last_selected_line_index": int(memory.last_selected_line_index),
        "cycle_count": int(memory.cycle_count),
    }


def serialize_behavior_decision(decision):
    return {
        "mode": decision.mode.value,
        "allowed_line_indices": [int(v) for v in decision.allowed_line_indices],
        "target_line_index": int(decision.target_line_index),
        "target_lane_id": decision.target_lane_id,
        "must_change": bool(decision.must_change),
        "deadline_margin_m": float(decision.deadline_margin_m),
        "reason": decision.reason,
    }


def serialize_obstacle_snapshot(obstacle):
    return {
        "ptc_id": obstacle.ptc_id,
        "x": float(obstacle.x),
        "y": float(obstacle.y),
        "v": float(obstacle.v),
        "heading": float(obstacle.heading),
        "a": float(obstacle.a),
        "length": float(obstacle.length),
        "width": float(obstacle.width),
        "height": float(obstacle.height),
        "timestamp_ms": int(obstacle.timestamp_ms),
    }


def serialize_projected_obstacle(projected):
    if projected is None:
        return None
    return {
        "obstacle": serialize_obstacle_snapshot(projected.obstacle),
        "s": float(projected.s),
        "d": float(projected.d),
        "gap": float(projected.gap),
        "relative_speed": float(projected.relative_speed),
    }


def serialize_line_context(line_context):
    return {
        "line_index": int(line_context.line_index),
        "line_id": line_context.line_id,
        "lane_sequence": list(line_context.lane_sequence),
        "lateral_step": int(line_context.lateral_step),
        "current_s": float(line_context.current_s),
        "current_d": float(line_context.current_d),
        "remaining_length": float(line_context.remaining_length),
        "front_obstacle": serialize_projected_obstacle(line_context.front_obstacle),
        "rear_obstacle": serialize_projected_obstacle(line_context.rear_obstacle),
    }


def serialize_route_intent(route_intent):
    return {
        "current_lane_id": route_intent.current_lane_id,
        "current_road_id": route_intent.current_road_id,
        "desired_route_lane_id": route_intent.desired_route_lane_id,
        "target_lane_id": route_intent.target_lane_id,
        "target_line_index": int(route_intent.target_line_index),
        "must_change": bool(route_intent.must_change),
        "deadline_margin_m": float(route_intent.deadline_margin_m),
        "distance_to_goal": float(route_intent.distance_to_goal),
        "current_road_planned_lanes": list(route_intent.current_road_planned_lanes),
    }


def serialize_sampling_template(template):
    return {
        "name": template.name,
        "line_indices": [int(v) for v in template.line_indices],
        "hold_t_values": [float(v) for v in template.hold_t_values],
        "plan_T_values": [float(v) for v in template.plan_T_values],
        "target_v_values": [float(v) for v in template.target_v_values],
        "s_bias_values": [float(v) for v in template.s_bias_values],
    }


def serialize_terminal_candidate(candidate):
    return {
        "line_index": int(candidate.line_index),
        "line_id": candidate.line_id,
        "target_s": float(candidate.target_s),
        "target_v": float(candidate.target_v),
        "plan_T": float(candidate.plan_T),
        "hold_t": float(candidate.hold_t),
        "template_name": candidate.template_name,
    }


def serialize_candidate(candidate):
    return {
        "terminal": serialize_terminal_candidate(candidate.terminal),
        "trajectory": None if candidate.trajectory is None else np.asarray(candidate.trajectory, dtype=float).tolist(),
        "smooth_path": None if candidate.smooth_path is None else np.asarray(candidate.smooth_path, dtype=float).tolist(),
        "feasible": bool(candidate.feasible),
        "score": None if not np.isfinite(candidate.score) else float(candidate.score),
        "metrics": to_builtin(candidate.metrics),
        "failure_reason": candidate.failure_reason,
    }


def select_best_candidate_index(candidates):
    best_index = None
    best_score = float("inf")
    for idx, candidate in enumerate(candidates):
        if not candidate.feasible:
            continue
        if candidate.score < best_score:
            best_score = float(candidate.score)
            best_index = idx
    return best_index


def build_sampling_frame(
    gvt,
    step_idx,
    sim_time_s,
    ego_state_before_plan,
    ego_state_after_execute,
    status,
    remain_dist,
):
    planner = gvt.traj_planner
    planner_context = planner.last_context
    planner_output = planner.last_output
    decision = None if planner_output is None else serialize_behavior_decision(planner_output.decision)
    best_candidate_index = None
    selected_line_index = None
    selected_lateral_step = None

    if planner_output is not None:
        best_candidate_index = select_best_candidate_index(planner_output.candidates)
        if best_candidate_index is not None:
            selected_line_index = int(planner_output.candidates[best_candidate_index].terminal.line_index)
            if 0 <= selected_line_index < len(gvt.refer_line_lateral_steps):
                selected_lateral_step = int(gvt.refer_line_lateral_steps[selected_line_index])

    return {
        "step": int(step_idx),
        "sim_time_s": round(float(sim_time_s), 6),
        "status": status,
        "remain_dist": float(remain_dist),
        "ego_state_before_plan": ego_state_before_plan,
        "ego_state_after_execute": ego_state_after_execute,
        "current_mode": planner.attribute.get("plan_mode", ""),
        "planner_attr": to_builtin(dict(planner.attribute)),
        "current_road_id": gvt.current_road_id,
        "refer_line_lane_ids": list(gvt.refer_line_lane_ids),
        "refer_line_lane_sequences": [list(seq) for seq in gvt.refer_line_lane_sequences],
        "refer_line_lateral_steps": [int(v) for v in gvt.refer_line_lateral_steps],
        "current_lateral_step": int(gvt.refer_line_lateral_steps[0]) if gvt.refer_line_lateral_steps else None,
        "selected_candidate_index": best_candidate_index,
        "selected_line_index": selected_line_index,
        "selected_lateral_step": selected_lateral_step,
        "planner_memory_before": None
        if planner_context is None
        else serialize_behavior_memory(planner_context.planner_memory),
        "planner_memory_after": None
        if planner_output is None
        else serialize_behavior_memory(planner_output.planner_memory),
        "decision": decision,
        "route_intent": None
        if planner_context is None
        else serialize_route_intent(planner_context.route_intent),
        "line_contexts": []
        if planner_context is None
        else [serialize_line_context(line_context) for line_context in planner_context.line_contexts],
        "obstacle_snapshots": []
        if planner_context is None
        else [serialize_obstacle_snapshot(obstacle) for obstacle in planner_context.obstacle_snapshots],
        "sampling_templates": [serialize_sampling_template(template) for template in planner.last_templates],
        "terminal_samples": [serialize_terminal_candidate(candidate) for candidate in planner.last_terminals],
        "candidates": []
        if planner_output is None
        else [serialize_candidate(candidate) for candidate in planner_output.candidates],
    }


def build_summary(result, start_xy, goal_xy):
    sampling_frames = result["sampling_frames"]
    debug_steps = result["debug_steps"]
    mode_counts = Counter(
        frame["current_mode"] for frame in sampling_frames if frame.get("current_mode")
    )
    selected_lateral_step_counts = Counter(
        str(frame["selected_lateral_step"])
        for frame in sampling_frames
        if frame.get("selected_lateral_step") is not None
    )
    candidate_total = sum(len(frame["candidates"]) for frame in sampling_frames)
    feasible_candidate_total = sum(
        sum(1 for candidate in frame["candidates"] if candidate["feasible"])
        for frame in sampling_frames
    )
    last_step = debug_steps[-1] if debug_steps else None
    first_lane_change_step = next(
        (
            frame["step"]
            for frame in sampling_frames
            if frame.get("selected_lateral_step") not in (None, 0)
        ),
        None,
    )
    return {
        "start_xy": [float(start_xy[0]), float(start_xy[1])],
        "goal_xy": [float(goal_xy[0]), float(goal_xy[1])],
        "planned_path": to_builtin(result["planned_path"]),
        "planned_actions": to_builtin(result["planned_actions"]),
        "road_sequence": to_builtin(result["road_sequence"]),
        "rollout_step_count": len(debug_steps),
        "control_frame_count": len(result["control_sequence"]),
        "trajectory_point_count": int(result["executed_trajectory"].shape[0]),
        "sampling_frame_count": len(sampling_frames),
        "candidate_total": int(candidate_total),
        "feasible_candidate_total": int(feasible_candidate_total),
        "mode_counts": dict(mode_counts),
        "selected_lateral_step_counts": dict(selected_lateral_step_counts),
        "first_lane_change_step": first_lane_change_step,
        "arrived": bool(last_step is not None and last_step["status"] == "arrived"),
        "final_state": last_step,
    }


def initial_heading_from_line(line):
    line = np.asarray(line, dtype=float)
    if len(line) < 2:
        raise ValueError("参考线点数不足，无法估计初始航向")
    vec = line[1, :2] - line[0, :2]
    return float(math.atan2(vec[1], vec[0]))


def is_arrived(state_xy, goal_xy, speed, dist_thresh, speed_thresh):
    dist = float(np.linalg.norm(np.asarray(state_xy, dtype=float) - np.asarray(goal_xy, dtype=float)))
    return dist <= dist_thresh and speed <= speed_thresh, dist


def build_hold_command():
    return {
        "acceleration": -1.0,
        "wheel_angle": 0.0,
        "gear": 4,
        "turnlight": 0,
        "brkreq": 1,
    }


def infer_turnlight(trajectory):
    traj = np.asarray(trajectory, dtype=float)
    future_idx = min(10, len(traj) - 1)
    current_pt = traj[0]
    future_pt = traj[future_idx]
    dx = future_pt[0] - current_pt[0]
    dy = future_pt[1] - current_pt[1]
    future_heading = np.arctan2(dy, dx)
    heading_delta = (future_heading - current_pt[2] + np.pi) % (2 * np.pi) - np.pi
    if heading_delta > np.radians(5.0):
        return 2
    if heading_delta < -np.radians(5.0):
        return 1
    return 0


def trajectory_to_control_cmd(gvt, trajectory):
    traj = np.asarray(trajectory, dtype=float)
    preview_idx = min(5, len(traj) - 1)
    preview = traj[preview_idx]

    acceleration = float(
        np.clip(preview[4], gvt.config["constrain"]["min_acc"], gvt.config["constrain"]["max_acc"])
    )
    wheel_angle_rad = float(np.arctan(gvt.wheelbase * preview[5]))
    wheel_angle_deg = float(np.degrees(wheel_angle_rad))
    target_speed = float(max(preview[3], 0.0))
    brkreq = 1 if target_speed < 0.05 and acceleration <= 0.0 else 0

    return {
        "acceleration": acceleration,
        "wheel_angle": wheel_angle_deg,
        "gear": 4,
        "turnlight": infer_turnlight(traj),
        "brkreq": brkreq,
        "target_speed": target_speed,
    }


def rollout(
    start_xy,
    goal_xy,
    xodr_parser,
    advance_index,
    max_steps,
    arrive_dist,
    arrive_speed,
    verbose_planner,
    replan_dt,
):
    gvt = instantiate_vehicle(xodr_parser)
    gvt.config["local_fsm"]["arrive_complete_dist"] = float(arrive_dist)
    gvt.config["local_fsm"]["arrive_complete_speed"] = float(arrive_speed)
    with contextlib.nullcontext() if verbose_planner else contextlib.redirect_stdout(io.StringIO()):
        gvt.global_planning(np.asarray(start_xy, dtype=float), np.asarray(goal_xy, dtype=float))

    gvt.x = float(start_xy[0])
    gvt.y = float(start_xy[1])
    gvt.heading = initial_heading_from_line(gvt.refer_lines[0])
    gvt.v = 0.0
    gvt.a = 0.0
    gvt.kappa = 0.0

    executed_points = []
    debug_steps = []
    control_sequence = []
    sampling_frames = []
    vehicle_states = []

    initial_point = [gvt.x, gvt.y, gvt.heading, gvt.v, gvt.a, gvt.kappa]
    executed_points.append(initial_point)

    for step_idx in range(max_steps):
        ego_state_before_plan = snapshot_vehicle_state(gvt)
        with contextlib.nullcontext() if verbose_planner else contextlib.redirect_stdout(io.StringIO()):
            trajectory = gvt.traj_planning([])
        if trajectory is None:
            remain_dist = float(np.linalg.norm(np.asarray([gvt.x, gvt.y], dtype=float) - np.asarray(goal_xy, dtype=float)))
            planner_mode = gvt.traj_planner.attribute.get("plan_mode", "")
            status = "arrived" if planner_mode == "arrived" else "planner_failed"
            sampling_frame = build_sampling_frame(
                gvt=gvt,
                step_idx=step_idx,
                sim_time_s=step_idx * replan_dt,
                ego_state_before_plan=ego_state_before_plan,
                ego_state_after_execute=ego_state_before_plan,
                status=status,
                remain_dist=remain_dist,
            )
            sampling_frames.append(sampling_frame)
            vehicle_states.append(
                {
                    "step": step_idx,
                    "sim_time_s": round(step_idx * replan_dt, 6),
                    "before_plan": ego_state_before_plan,
                    "after_execute": ego_state_before_plan,
                    "status": status,
                }
            )
            debug_steps.append(
                {
                    "step": step_idx,
                    "status": status,
                    "x": gvt.x,
                    "y": gvt.y,
                    "v": gvt.v,
                    "remain_dist": remain_dist,
                    "current_mode": sampling_frame["current_mode"],
                    "selected_lateral_step": sampling_frame["selected_lateral_step"],
                    "current_lateral_step": sampling_frame["current_lateral_step"],
                    "refer_line_lateral_steps": sampling_frame["refer_line_lateral_steps"],
                    "decision": sampling_frame["decision"],
                    "planner_attr": to_builtin(dict(gvt.traj_planner.attribute)),
                }
            )
            if planner_mode == "arrived":
                break
            break

        traj = np.asarray(trajectory, dtype=float)
        command = trajectory_to_control_cmd(gvt, traj)
        command.update(
            {
                "step": step_idx,
                "sim_time_s": round(step_idx * replan_dt, 6),
                "x": float(gvt.x),
                "y": float(gvt.y),
                "heading": float(gvt.heading),
                "speed": float(gvt.v),
            }
        )
        control_sequence.append(command)

        # 离线仿真按真实车端语义执行：每轮只消费一小段局部轨迹，持续重规划。
        exec_idx = min(max(1, advance_index), len(traj) - 1)
        exec_slice = traj[1 : exec_idx + 1]

        for pt in exec_slice:
            executed_points.append(pt.tolist())

        exec_point = traj[exec_idx]
        gvt.x = float(exec_point[0])
        gvt.y = float(exec_point[1])
        gvt.heading = float(exec_point[2])
        gvt.v = float(exec_point[3])
        gvt.a = float(exec_point[4])
        gvt.kappa = float(exec_point[5])
        ego_state_after_execute = snapshot_vehicle_state(gvt)

        remain_dist = float(np.linalg.norm(np.asarray([gvt.x, gvt.y], dtype=float) - np.asarray(goal_xy, dtype=float)))
        sampling_frame = build_sampling_frame(
            gvt=gvt,
            step_idx=step_idx,
            sim_time_s=step_idx * replan_dt,
            ego_state_before_plan=ego_state_before_plan,
            ego_state_after_execute=ego_state_after_execute,
            status="running",
            remain_dist=remain_dist,
        )
        sampling_frames.append(sampling_frame)
        vehicle_states.append(
            {
                "step": step_idx,
                "sim_time_s": round(step_idx * replan_dt, 6),
                "before_plan": ego_state_before_plan,
                "after_execute": ego_state_after_execute,
                "status": "running",
            }
        )
        debug_steps.append(
            {
                "step": step_idx,
                "status": "running",
                "x": gvt.x,
                "y": gvt.y,
                "v": gvt.v,
                "a": gvt.a,
                "kappa": gvt.kappa,
                "remain_dist": remain_dist,
                "current_mode": sampling_frame["current_mode"],
                "selected_lateral_step": sampling_frame["selected_lateral_step"],
                "current_lateral_step": sampling_frame["current_lateral_step"],
                "refer_line_lateral_steps": sampling_frame["refer_line_lateral_steps"],
                "decision": sampling_frame["decision"],
                "planner_attr": to_builtin(dict(gvt.traj_planner.attribute)),
                "refer_line_count": len(gvt.refer_lines),
            }
        )

    return {
        "planned_path": list(gvt.lanePlanner.planner.planned_path),
        "planned_actions": list(gvt.lanePlanner.planner.planned_actions),
        "road_sequence": list(gvt.lanePlanner.planner.road_sequence),
        "executed_trajectory": np.asarray(executed_points, dtype=float),
        "debug_steps": debug_steps,
        "control_sequence": control_sequence,
        "sampling_frames": sampling_frames,
        "vehicle_states": vehicle_states,
        "goal_xy": np.asarray(goal_xy, dtype=float),
    }


def plot_rollout(xodr_parser, rollout_result, start_xy, goal_xy, output_path):
    executed = rollout_result["executed_trajectory"]
    planned_path = rollout_result["planned_path"]
    sampling_frames = rollout_result["sampling_frames"]

    fig, ax = plt.subplots(figsize=(11, 11))

    for centerline in xodr_parser.lane_centerlines.values():
        if len(centerline) < 2:
            continue
        ax.plot(
            centerline[:, 0],
            centerline[:, 1],
            color="#d0d0d0",
            linewidth=0.45,
            alpha=0.35,
            zorder=1,
        )

    for lane_id in planned_path:
        centerline = xodr_parser.lane_centerlines.get(lane_id)
        if centerline is None or len(centerline) < 2:
            continue
        ax.plot(
            centerline[:, 0],
            centerline[:, 1],
            color="#4c78a8",
            linewidth=1.8,
            alpha=0.9,
            zorder=2,
        )

    sampled_label_drawn = False
    selected_label_drawn = False
    for frame in sampling_frames:
        selected_index = frame.get("selected_candidate_index")
        for candidate_idx, candidate in enumerate(frame["candidates"]):
            path_like = candidate["smooth_path"] or candidate["trajectory"]
            if path_like is None:
                continue
            sample_xy = np.asarray(path_like, dtype=float)
            if sample_xy.ndim != 2 or sample_xy.shape[0] < 2 or sample_xy.shape[1] < 2:
                continue
            ax.plot(
                sample_xy[:, 0],
                sample_xy[:, 1],
                color="#2ca25f",
                linewidth=0.7,
                alpha=0.06,
                zorder=2.4,
                label="sampled trajectories" if not sampled_label_drawn else None,
            )
            sampled_label_drawn = True
            if selected_index is not None and candidate_idx == selected_index:
                ax.plot(
                    sample_xy[:, 0],
                    sample_xy[:, 1],
                    color="#f28e2b",
                    linewidth=1.2,
                    alpha=0.22,
                    zorder=3.2,
                    label="selected per frame" if not selected_label_drawn else None,
                )
                selected_label_drawn = True

    scatter = ax.scatter(
        executed[:, 0],
        executed[:, 1],
        c=executed[:, 3],
        cmap="viridis",
        s=12,
        zorder=4,
    )
    ax.plot(
        executed[:, 0],
        executed[:, 1],
        color="#e36a1a",
        linewidth=2.6,
        alpha=0.95,
        zorder=4,
        label="executed rollout",
    )
    ax.scatter(
        executed[:, 0],
        executed[:, 1],
        color="#4c78a8",
        s=10,
        alpha=0.8,
        zorder=4.2,
        label="vehicle states",
    )
    ax.scatter(start_xy[0], start_xy[1], color="#1f9d55", s=80, zorder=5, label="start")
    ax.scatter(goal_xy[0], goal_xy[1], color="#d64545", s=80, zorder=5, label="goal")

    x_all = np.concatenate([executed[:, 0], np.array([start_xy[0], goal_xy[0]])])
    y_all = np.concatenate([executed[:, 1], np.array([start_xy[1], goal_xy[1]])])
    padding = max(20.0, 0.15 * max(x_all.max() - x_all.min(), y_all.max() - y_all.min(), 1.0))
    ax.set_xlim(x_all.min() - padding, x_all.max() + padding)
    ax.set_ylim(y_all.min() - padding, y_all.max() + padding)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.35)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("Rolling Simulation Sampling Overview")
    ax.legend(loc="best")
    colorbar = fig.colorbar(scatter, ax=ax, fraction=0.03, pad=0.02)
    colorbar.set_label("speed (m/s)")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def main():
    args = build_arg_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    xodr_parser = XODRParser.load_cache(args.map_cache)
    start_xy = np.array(args.start_xy, dtype=float)
    goal_xy = np.array(args.goal_xy, dtype=float)

    result = rollout(
        start_xy=start_xy,
        goal_xy=goal_xy,
        xodr_parser=xodr_parser,
        advance_index=args.advance_index,
        max_steps=args.max_steps,
        arrive_dist=args.arrive_dist,
        arrive_speed=args.arrive_speed,
        verbose_planner=args.verbose_planner,
        replan_dt=args.replan_dt,
    )

    traj_path = output_dir / f"{args.output_prefix}_trajectory.json"
    debug_path = output_dir / f"{args.output_prefix}_debug.json"
    control_path = output_dir / f"{args.output_prefix}_controlcmd.json"
    sampling_path = output_dir / f"{args.output_prefix}_sampling_frames.json"
    vehicle_path = output_dir / f"{args.output_prefix}_vehicle_states.json"
    summary_path = output_dir / f"{args.output_prefix}_summary.json"
    img_path = output_dir / f"{args.output_prefix}_plot.png"
    summary = build_summary(result, start_xy, goal_xy)

    with traj_path.open("w", encoding="utf-8") as f:
        json.dump(result["executed_trajectory"].tolist(), f, ensure_ascii=False, indent=2)
    with debug_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "planned_path": to_builtin(result["planned_path"]),
                "planned_actions": to_builtin(result["planned_actions"]),
                "road_sequence": to_builtin(result["road_sequence"]),
                "debug_steps": to_builtin(result["debug_steps"]),
                "mode_counts": summary["mode_counts"],
                "selected_lateral_step_counts": summary["selected_lateral_step_counts"],
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    with control_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "replan_dt": args.replan_dt,
                "control_sequence": to_builtin(result["control_sequence"]),
                "final_hold_command": build_hold_command(),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    with sampling_path.open("w", encoding="utf-8") as f:
        json.dump(result["sampling_frames"], f, ensure_ascii=False, indent=2)
    with vehicle_path.open("w", encoding="utf-8") as f:
        json.dump(result["vehicle_states"], f, ensure_ascii=False, indent=2)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    plot_rollout(xodr_parser, result, start_xy, goal_xy, img_path)

    last_step = result["debug_steps"][-1] if result["debug_steps"] else None
    print(f"TRAJ_JSON {traj_path}")
    print(f"DEBUG_JSON {debug_path}")
    print(f"CONTROL_JSON {control_path}")
    print(f"SAMPLING_JSON {sampling_path}")
    print(f"VEHICLE_JSON {vehicle_path}")
    print(f"SUMMARY_JSON {summary_path}")
    print(f"PLOT_PNG {img_path}")
    print(f"PLANNED_PATH {result['planned_path']}")
    print(f"PLANNED_ACTIONS {result['planned_actions']}")
    print(f"ROLLOUT_POINTS {result['executed_trajectory'].shape}")
    print(f"MODE_COUNTS {summary['mode_counts']}")
    print(f"SELECTED_LATERAL_STEP_COUNTS {summary['selected_lateral_step_counts']}")
    print(f"LAST_STEP {last_step}")


if __name__ == "__main__":
    main()
