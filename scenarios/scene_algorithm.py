"""Cut-in 场景算法入口。

第一版策略：
- 无主车或主车无效时，退化为沿当前参考线巡航
- 有效主车存在时，固定按配置侧寻找相邻一阶目标车道
- 以 TTC 为主控指标，对 ``hold_t`` / ``plan_T`` / 后车追近速度做小网格搜索
- 选出一帧最优切入目标，交给下层轨迹生成器生成真实轨迹
"""

from __future__ import annotations

import copy
from typing import Any

import numpy as np

from planning_runtime.scene.types import ReferenceLineView, SceneContext
from traj_planner.tools import get_path_point_at_s, normalize_angle, project_point_to_path


CFG: dict[str, Any] = {
    "cutin_side": "left",
    # 第一版默认采用更保守的 TTC 目标。
    # 在当前 min_net_gap_m=6.0m 的约束下，2.0s 会让低速主车场景长期无解；
    # 3.0s 能与当前净距下限和 closing speed 网格保持一致。
    "target_ttc_s": 1.5,
    "cruise_target_v": 5.0,
    "hold_t_grid": [0.0, 0.4, 0.8],
    "plan_T_grid": [2.2, 3.0, 4.0, 5.0],
    "rear_closing_speed_grid": [0.8, 1.2, 1.6, 2.0],
    "commit_debounce_frames": 3,
    "stabilize_duration_s": 1.5,
    "abort_cooldown_s": 2.0,
    "max_heading_diff_deg": 20.0,
    "vut_speed_min": 1.0,
    "vut_speed_max": 12.0,
    # 初版 cut-in 触发几何：主车位于自车相邻车道前方一段距离，自车沿本车道追赶。
    "search_rel_s_min": -5.0,
    "search_rel_s_max": 30.0,
    "target_lane_match_d_thresh": 1.5,
    "min_merge_remaining_m": 20.0,
    "max_ego_to_target_line_d_m": 4.5,
    "min_net_gap_m": 5.0,
    "net_gap_eps_m": 0.2,
    "ttc_tol_s": 0.3,
    "commit_ttc_tol_s": 0.5,
    "hard_min_ttc_s": 1.5,
    "min_target_progress_m": 2.0,
    "longitudinal_match_tol_m": 3.0,
    "post_merge_buffer_m": 5.0,
    "enter_target_d_thresh_m": 0.6,
    "finish_merge_d_thresh_m": 1.0,
    "execute_complete_frames": 2,
    "commit_fail_max_frames": 3,
    "commit_vut_speed_min": 0.0,
    "commit_rel_s_max": 40.0,
    "prediction_min_acc": -1.5,
    "prediction_max_acc": 1.5,
}


SCENE_MEMORY: dict[str, Any] = {
    "mode": "CRUISE",
    "search_hits": 0,
    "cooldown_until_time_s": -1.0,
    "locked_line_index": None,
    "locked_target_lane_id": "",
    "locked_source_lane_id": "",
    "locked_target_lane_sequence": [],
    "locked_source_lane_sequence": [],
    "locked_hold_t": 0.0,
    "locked_plan_T": 0.0,
    "locked_rear_closing_speed": 0.0,
    "locked_target_ttc": 0.0,
    "commit_start_time_s": -1.0,
    "execute_end_time_s": -1.0,
    "execute_target_xy": None,
    "execute_target_heading": 0.0,
    "execute_target_v": 0.0,
    "execute_target_a": 0.0,
    "execute_complete_hits": 0,
    "stable_until_time_s": -1.0,
    "commit_fail_count": 0,
    "last_best_solution": None,
}


def _clamp(value: float, lower: float, upper: float) -> float:
    return float(min(max(value, lower), upper))


def _heading_diff_deg(h1: float, h2: float) -> float:
    return float(np.degrees(abs(normalize_angle(float(h1) - float(h2)))))


def _read_ego_length(context: SceneContext) -> float:
    runtime_frame = context.runtime_frame
    if runtime_frame is None:
        return 4.5
    return float(runtime_frame.config.get("size", {}).get("L", 4.5))


def _read_main_length(context: SceneContext) -> float:
    # 当前用户已确认主车尺寸与自车一致，第一版直接复用自车长度。
    return _read_ego_length(context)


def _read_max_v(context: SceneContext) -> float:
    runtime_frame = context.runtime_frame
    if runtime_frame is None:
        return 6.94
    return float(runtime_frame.config.get("constrain", {}).get("max_v", 6.94))


def _read_min_acc(context: SceneContext) -> float:
    runtime_frame = context.runtime_frame
    if runtime_frame is None:
        return -2.0
    return float(runtime_frame.config.get("constrain", {}).get("min_acc", -2.0))


def _read_max_acc(context: SceneContext) -> float:
    runtime_frame = context.runtime_frame
    if runtime_frame is None:
        return 2.0
    return float(runtime_frame.config.get("constrain", {}).get("max_acc", 2.0))


def _reset_commit_memory(memory: dict[str, Any]) -> None:
    memory["locked_line_index"] = None
    memory["locked_target_lane_id"] = ""
    memory["locked_source_lane_id"] = ""
    memory["locked_target_lane_sequence"] = []
    memory["locked_source_lane_sequence"] = []
    memory["locked_hold_t"] = 0.0
    memory["locked_plan_T"] = 0.0
    memory["locked_rear_closing_speed"] = 0.0
    memory["locked_target_ttc"] = 0.0
    memory["commit_start_time_s"] = -1.0
    memory["execute_end_time_s"] = -1.0
    memory["execute_target_xy"] = None
    memory["execute_target_heading"] = 0.0
    memory["execute_target_v"] = 0.0
    memory["execute_target_a"] = 0.0
    memory["execute_complete_hits"] = 0
    memory["stable_until_time_s"] = -1.0
    memory["commit_fail_count"] = 0
    memory["last_best_solution"] = None


def _reset_scene_memory() -> None:
    global SCENE_MEMORY
    SCENE_MEMORY = {
        "mode": "CRUISE",
        "search_hits": 0,
        "cooldown_until_time_s": -1.0,
        "locked_line_index": None,
        "locked_target_lane_id": "",
        "locked_source_lane_id": "",
        "locked_target_lane_sequence": [],
        "locked_source_lane_sequence": [],
        "locked_hold_t": 0.0,
        "locked_plan_T": 0.0,
        "locked_rear_closing_speed": 0.0,
        "locked_target_ttc": 0.0,
        "commit_start_time_s": -1.0,
        "execute_end_time_s": -1.0,
        "execute_target_xy": None,
        "execute_target_heading": 0.0,
        "execute_target_v": 0.0,
        "execute_target_a": 0.0,
        "execute_complete_hits": 0,
        "stable_until_time_s": -1.0,
        "commit_fail_count": 0,
        "last_best_solution": None,
    }


def _enter_cooldown(memory: dict[str, Any], now: float) -> None:
    memory["mode"] = "COOLDOWN"
    memory["search_hits"] = 0
    memory["cooldown_until_time_s"] = float(now) + float(CFG["abort_cooldown_s"])
    _reset_commit_memory(memory)


def _line_by_index(
    candidate_lines: list[ReferenceLineView],
    line_index: int | None,
) -> ReferenceLineView | None:
    if line_index is None:
        return None
    for line in candidate_lines:
        if int(line.line_index) == int(line_index):
            return line
    return None


def _line_by_lane_id(
    candidate_lines: list[ReferenceLineView],
    lane_id: str,
) -> ReferenceLineView | None:
    target_lane_id = str(lane_id or "")
    if not target_lane_id:
        return None
    for line in candidate_lines:
        if str(line.lane_id) == target_lane_id:
            return line
    return None


def _normalize_lane_sequence(lane_sequence: list[str] | tuple[str, ...] | None) -> list[str]:
    if lane_sequence is None:
        return []
    return [str(lane_id) for lane_id in lane_sequence]


def _line_by_lane_sequence(
    candidate_lines: list[ReferenceLineView],
    target_lane_sequence: list[str] | tuple[str, ...] | None,
) -> ReferenceLineView | None:
    """按锁定的 lane 链在当前候选线里恢复语义最一致的线。"""

    normalized_target = _normalize_lane_sequence(target_lane_sequence)
    if not normalized_target:
        return None

    target_set = set(normalized_target)
    best_line = None
    best_score: tuple[int, int, int] | None = None
    for line in candidate_lines:
        line_sequence = _normalize_lane_sequence(line.lane_sequence)
        overlap_count = sum(1 for lane_id in line_sequence if lane_id in target_set)
        if overlap_count <= 0:
            continue

        first_match_idx = len(normalized_target)
        for idx, lane_id in enumerate(normalized_target):
            if lane_id in line_sequence:
                first_match_idx = idx
                break

        score = (
            overlap_count,
            -first_match_idx,
            -len(line_sequence),
        )
        if best_score is None or score > best_score:
            best_score = score
            best_line = line
    return best_line


def _has_valid_main_vehicle(context: SceneContext) -> bool:
    if context.main_vehicle is None:
        return False

    ego = context.ego_vehicle
    main = context.main_vehicle

    same_direction = _heading_diff_deg(main.heading, ego.heading) <= float(CFG["max_heading_diff_deg"])
    speed_ok = float(CFG["vut_speed_min"]) <= float(main.v) <= float(CFG["vut_speed_max"])
    rel_s_ok = float(CFG["search_rel_s_min"]) <= float(main.relative_s_to_ego) <= float(CFG["search_rel_s_max"])
    return bool(same_direction and speed_ok and rel_s_ok)


def _has_valid_main_vehicle_for_commit(context: SceneContext) -> bool:
    """COMMIT 阶段使用更宽松的主车有效性判定。"""

    if context.main_vehicle is None:
        return False

    ego = context.ego_vehicle
    main = context.main_vehicle

    same_direction = _heading_diff_deg(main.heading, ego.heading) <= float(CFG["max_heading_diff_deg"])
    speed_ok = float(CFG["commit_vut_speed_min"]) <= float(main.v) <= float(CFG["vut_speed_max"])
    rel_s_ok = float(main.relative_s_to_ego) <= float(CFG["commit_rel_s_max"])
    return bool(same_direction and speed_ok and rel_s_ok)


def _baseline_keep_lane(
    context: SceneContext,
    reason: str = "baseline_keep_lane",
) -> dict[str, Any]:
    ego = context.ego_vehicle
    line = ego.current_ref_line

    target_v = float(CFG["cruise_target_v"])
    plan_t = 2.0
    target_s = float(line.current_s) + target_v * plan_t + 4.0
    target_s = _clamp(
        target_s,
        float(line.current_s) + float(CFG["min_target_progress_m"]),
        float(line.cumulative_s[-1]) - 0.5,
    )
    target_xy, target_heading = get_path_point_at_s(line.line, target_s, line.cumulative_s)

    return {
        "x": float(target_xy[0]),
        "y": float(target_xy[1]),
        "v": float(target_v),
        "a": 0.0,
        "theta": float(target_heading),
        "t": float(plan_t),
        "target_line_index": 0,
        "hold_t": 0.0,
        "reason": str(reason),
    }


def _choose_target_line(context: SceneContext) -> ReferenceLineView | None:
    ego = context.ego_vehicle
    main = context.main_vehicle
    if main is None:
        return None

    side_sign = -1 if str(CFG["cutin_side"]).lower() == "left" else 1
    best_line = None
    best_score = float("inf")

    for line in ego.candidate_ref_lines:
        if abs(int(line.lateral_step)) != 1:
            continue
        if side_sign < 0 and int(line.lateral_step) >= 0:
            continue
        if side_sign > 0 and int(line.lateral_step) <= 0:
            continue

        remaining = float(line.cumulative_s[-1]) - float(line.current_s)
        if remaining < float(CFG["min_merge_remaining_m"]):
            continue

        _, main_s_on_line, main_d_on_line = project_point_to_path(np.array([main.x, main.y], dtype=float), line.line)
        if abs(float(main_d_on_line)) > float(CFG["target_lane_match_d_thresh"]):
            continue

        _, ego_s_on_line, ego_d_on_line = project_point_to_path(
            np.array([ego.x, ego.y], dtype=float),
            line.line,
        )
        if abs(float(ego_d_on_line)) > float(CFG["max_ego_to_target_line_d_m"]):
            continue

        score = abs(float(main_d_on_line)) + 0.05 * abs(float(main_s_on_line) - float(ego_s_on_line))
        if score < best_score:
            best_score = score
            best_line = line

    return best_line


def _predict_vut_at_t(
    context: SceneContext,
    target_line: ReferenceLineView,
    total_t: float,
) -> dict[str, Any]:
    main = context.main_vehicle
    if main is None:
        raise ValueError("main_vehicle is required for VUT prediction")

    main_line = main.current_ref_line
    a_use = _clamp(
        float(main.a),
        float(CFG["prediction_min_acc"]),
        float(CFG["prediction_max_acc"]),
    )
    s_pred_on_main = float(main_line.current_s) + float(main.v) * float(total_t) + 0.5 * a_use * float(total_t) ** 2
    s_pred_on_main = _clamp(s_pred_on_main, float(main_line.current_s), float(main_line.cumulative_s[-1]) - 0.5)
    xy_pred, heading_pred = get_path_point_at_s(main_line.line, s_pred_on_main, main_line.cumulative_s)
    v_pred = max(0.0, float(main.v) + a_use * float(total_t))
    _, s_pred_on_target, d_pred_on_target = project_point_to_path(xy_pred, target_line.line)

    return {
        "x": float(xy_pred[0]),
        "y": float(xy_pred[1]),
        "heading": float(heading_pred),
        "v": float(v_pred),
        "a": float(a_use),
        "s_on_main_line": float(s_pred_on_main),
        "s_on_target_line": float(s_pred_on_target),
        "d_on_target_line": float(d_pred_on_target),
    }


def _solve_cutin_target_from_ttc(
    context: SceneContext,
    target_line: ReferenceLineView,
    hold_t: float,
    plan_t: float,
    rear_closing_speed: float,
) -> dict[str, Any] | None:
    ego = context.ego_vehicle
    total_t = float(hold_t) + float(plan_t)
    if total_t <= 0.3 or rear_closing_speed <= 0.0:
        return None

    l_gvt = _read_ego_length(context)
    l_vut = _read_main_length(context)
    vut_pred = _predict_vut_at_t(context, target_line, total_t)
    v_gvt_target = float(vut_pred["v"]) - float(rear_closing_speed)
    if v_gvt_target < 0.0 or v_gvt_target > _read_max_v(context):
        return None

    net_gap_target = max(
        float(CFG["target_ttc_s"]) * float(rear_closing_speed),
        float(CFG["min_net_gap_m"]),
    )
    target_s = (
        float(vut_pred["s_on_target_line"])
        + 0.5 * float(l_vut)
        + float(net_gap_target)
        + 0.5 * float(l_gvt)
    )

    _, ego_s_on_target, ego_d_on_target = project_point_to_path(np.array([ego.x, ego.y], dtype=float), target_line.line)
    ds_needed = float(target_s) - float(ego_s_on_target)
    if ds_needed < float(CFG["min_target_progress_m"]):
        return None

    if float(target_s) > float(target_line.cumulative_s[-1]) - float(CFG["post_merge_buffer_m"]):
        return None

    ds_feasible = 0.5 * (float(ego.v) + float(v_gvt_target)) * float(total_t)
    if abs(ds_needed - ds_feasible) > float(CFG["longitudinal_match_tol_m"]):
        return None

    a_req = (float(v_gvt_target) - float(ego.v)) / float(total_t)
    if a_req < _read_min_acc(context) or a_req > _read_max_acc(context):
        return None

    if abs(float(ego_d_on_target)) > float(CFG["max_ego_to_target_line_d_m"]):
        return None

    target_xy, target_heading = get_path_point_at_s(target_line.line, target_s, target_line.cumulative_s)
    return {
        "x": float(target_xy[0]),
        "y": float(target_xy[1]),
        "v": float(v_gvt_target),
        "a": float(a_req),
        "theta": float(target_heading),
        "t": float(plan_t),
        "target_line_index": int(target_line.line_index),
        "hold_t": float(hold_t),
        "reason": "cutin_ttc_target",
        "_meta": {
            "target_s": float(target_s),
            "ego_s_on_target": float(ego_s_on_target),
            "ego_d_on_target": float(ego_d_on_target),
            "vut_pred": vut_pred,
            "rear_closing_speed": float(rear_closing_speed),
            "net_gap_target": float(net_gap_target),
            "total_T": float(total_t),
            "ds_needed": float(ds_needed),
            "ds_feasible": float(ds_feasible),
        },
    }


def _check_cutin_window_ttc(
    context: SceneContext,
    target_line: ReferenceLineView,
    solution: dict[str, Any],
    *,
    net_gap_eps_m: float = 0.0,
    ttc_tol_s: float | None = None,
) -> tuple[bool, dict[str, Any]]:
    l_gvt = _read_ego_length(context)
    l_vut = _read_main_length(context)
    rear_closing_speed = float(solution["_meta"]["rear_closing_speed"])
    vut_pred = solution["_meta"]["vut_pred"]

    if rear_closing_speed <= 0.0:
        return False, {"reason": "non_positive_closing_speed"}

    _, target_s, _ = project_point_to_path(
        np.array([solution["x"], solution["y"]], dtype=float),
        target_line.line,
    )
    net_gap = float(target_s) - float(vut_pred["s_on_target_line"]) - 0.5 * (float(l_gvt) + float(l_vut))
    min_net_gap = float(CFG["min_net_gap_m"]) - max(0.0, float(net_gap_eps_m))
    if net_gap < min_net_gap:
        return False, {"reason": "net_gap_too_small"}

    achieved_ttc = float(net_gap) / float(rear_closing_speed)
    target_ttc_tol = float(CFG["ttc_tol_s"]) if ttc_tol_s is None else float(ttc_tol_s)
    if abs(achieved_ttc - float(CFG["target_ttc_s"])) > target_ttc_tol:
        return False, {"reason": "ttc_not_match", "achieved_ttc": float(achieved_ttc)}

    if achieved_ttc < float(CFG["hard_min_ttc_s"]):
        return False, {"reason": "ttc_below_hard_min", "achieved_ttc": float(achieved_ttc)}

    remaining_after_merge = float(target_line.cumulative_s[-1]) - float(target_s)
    if remaining_after_merge < float(CFG["post_merge_buffer_m"]):
        return False, {"reason": "not_enough_post_merge_distance"}

    return True, {
        "reason": "ok",
        "achieved_ttc": float(achieved_ttc),
        "net_gap": float(net_gap),
        "rear_closing_speed": float(rear_closing_speed),
        "remaining_after_merge": float(remaining_after_merge),
        "ego_d_to_target": float(solution["_meta"]["ego_d_on_target"]),
        "ds_needed": float(solution["_meta"]["ds_needed"]),
        "a_req": float(solution["a"]),
        "total_T": float(solution["_meta"]["total_T"]),
    }


def _search_best_cutin_solution(
    context: SceneContext,
    target_line: ReferenceLineView,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    best_solution = None
    best_metrics = None
    best_cost = float("inf")

    for hold_t in CFG["hold_t_grid"]:
        for plan_t in CFG["plan_T_grid"]:
            for rear_closing_speed in CFG["rear_closing_speed_grid"]:
                solution = _solve_cutin_target_from_ttc(
                    context=context,
                    target_line=target_line,
                    hold_t=float(hold_t),
                    plan_t=float(plan_t),
                    rear_closing_speed=float(rear_closing_speed),
                )
                if solution is None:
                    continue

                ok, metrics = _check_cutin_window_ttc(context, target_line, solution)
                if not ok:
                    continue

                cost = 0.0
                cost += 1.0 * float(hold_t)
                cost += 0.4 * float(plan_t)
                cost += 0.3 * abs(float(solution["a"]))
                cost += 0.2 * float(rear_closing_speed)
                if cost < best_cost:
                    best_cost = float(cost)
                    best_solution = solution
                    best_metrics = metrics

    return best_solution, best_metrics


def _select_execute_fallback_output(
    context: SceneContext,
    memory: dict[str, Any],
    candidate_solution: dict[str, Any] | None,
    reason: str,
) -> dict[str, Any]:
    """EXECUTE 小幅失败时的输出选择。

    优先级：
    1. 当前帧重新解算出的候选
    2. 上一帧已锁定的场景目标
    3. 回退到 baseline
    """

    if candidate_solution is not None:
        return candidate_solution
    if memory.get("last_best_solution") is not None:
        return copy.deepcopy(memory["last_best_solution"])
    return _baseline_keep_lane(context, reason=reason)


def _enter_execute_from_solution(
    memory: dict[str, Any],
    source_line: ReferenceLineView,
    target_line: ReferenceLineView,
    solution: dict[str, Any],
    now: float,
) -> None:
    """把搜索阶段最优解冻结成执行态计划。"""

    memory["mode"] = "EXECUTE"
    memory["locked_line_index"] = int(solution["target_line_index"])
    memory["locked_target_lane_id"] = str(target_line.lane_id)
    memory["locked_source_lane_id"] = str(source_line.lane_id)
    memory["locked_target_lane_sequence"] = _normalize_lane_sequence(target_line.lane_sequence)
    memory["locked_source_lane_sequence"] = _normalize_lane_sequence(source_line.lane_sequence)
    memory["locked_hold_t"] = float(solution["hold_t"])
    memory["locked_plan_T"] = float(solution["t"])
    memory["locked_rear_closing_speed"] = float(solution["_meta"]["rear_closing_speed"])
    memory["locked_target_ttc"] = float(CFG["target_ttc_s"])
    memory["commit_start_time_s"] = float(now)
    memory["execute_end_time_s"] = float(now) + float(solution["t"])
    memory["execute_target_xy"] = [float(solution["x"]), float(solution["y"])]
    memory["execute_target_heading"] = float(solution["theta"])
    memory["execute_target_v"] = float(solution["v"])
    memory["execute_target_a"] = float(solution["a"])
    memory["execute_complete_hits"] = 0
    memory["commit_fail_count"] = 0
    memory["search_hits"] = 0
    memory["last_best_solution"] = copy.deepcopy(solution)


def _resolve_execute_target_line(
    context: SceneContext,
    memory: dict[str, Any],
) -> ReferenceLineView | None:
    """根据执行态缓存恢复当前帧的目标线。"""

    line = _line_by_lane_sequence(
        context.ego_vehicle.candidate_ref_lines,
        memory.get("locked_target_lane_sequence", []),
    )
    if line is not None:
        return line
    line = _line_by_lane_id(context.ego_vehicle.candidate_ref_lines, memory.get("locked_target_lane_id", ""))
    if line is not None:
        return line
    return _line_by_index(context.ego_vehicle.candidate_ref_lines, memory.get("locked_line_index"))


def _build_execute_output(
    context: SceneContext,
    memory: dict[str, Any],
    target_line: ReferenceLineView,
) -> dict[str, Any] | None:
    """基于冻结的终点目标生成执行态输出。

    执行态不再每帧重新搜索 TTC 候选，而是持续向已锁定的换道终点推进。
    """

    target_xy = memory.get("execute_target_xy")
    if target_xy is None:
        return None

    target_xy = np.asarray(target_xy, dtype=float)
    _, target_s, _ = project_point_to_path(target_xy, target_line.line)
    target_s = max(float(target_s), float(target_line.current_s) + 1.0)
    line_end_s = float(target_line.cumulative_s[-1]) - 0.5
    if target_s > line_end_s:
        target_s = line_end_s
    if target_s <= float(target_line.current_s) + 1e-6:
        return None

    projected_xy, projected_heading = get_path_point_at_s(target_line.line, target_s, target_line.cumulative_s)
    remaining_time = max(0.5, float(memory.get("execute_end_time_s", 0.0)) - float(context.time_s))

    return {
        "x": float(projected_xy[0]),
        "y": float(projected_xy[1]),
        "v": float(memory.get("execute_target_v", 0.0)),
        "a": float(memory.get("execute_target_a", 0.0)),
        "theta": float(projected_heading),
        "t": float(remaining_time),
        "target_line_index": int(target_line.line_index),
        "hold_t": 0.0,
        "reason": "cutin_execute",
    }


def _check_execute_hard_abort(
    context: SceneContext,
    target_line: ReferenceLineView,
    execute_output: dict[str, Any],
) -> tuple[bool, dict[str, Any]]:
    """执行态硬安全判定。

    与搜索阶段不同，这里不再要求精确命中 TTC 目标，只在明显不安全时中止：
    - 预测碰撞时间低于硬阈值
    - 目标点相对主车已经落到明显碰撞区
    """

    if context.main_vehicle is None:
        return False, {"reason": "no_main_vehicle"}

    l_gvt = _read_ego_length(context)
    l_vut = _read_main_length(context)
    target_t = max(float(execute_output["t"]), 0.5)
    vut_pred = _predict_vut_at_t(context, target_line, target_t)
    _, target_s, _ = project_point_to_path(
        np.array([execute_output["x"], execute_output["y"]], dtype=float),
        target_line.line,
    )
    net_gap = float(target_s) - float(vut_pred["s_on_target_line"]) - 0.5 * (float(l_gvt) + float(l_vut))
    rear_closing_speed = float(vut_pred["v"]) - float(execute_output["v"])
    if rear_closing_speed <= 1e-6:
        return False, {
            "reason": "rear_not_closing",
            "net_gap": float(net_gap),
            "rear_closing_speed": float(rear_closing_speed),
        }

    hard_ttc = float(net_gap) / float(rear_closing_speed)
    hard_gap_limit = -float(CFG["net_gap_eps_m"])
    hard_ttc_limit = float(CFG["hard_min_ttc_s"]) - float(CFG["commit_ttc_tol_s"])
    if net_gap < hard_gap_limit or hard_ttc < hard_ttc_limit:
        return True, {
            "reason": "execute_hard_abort",
            "hard_ttc": float(hard_ttc),
            "net_gap": float(net_gap),
            "rear_closing_speed": float(rear_closing_speed),
        }
    return False, {
        "reason": "execute_safe",
        "hard_ttc": float(hard_ttc),
        "net_gap": float(net_gap),
        "rear_closing_speed": float(rear_closing_speed),
    }


def _is_execute_completed(
    context: SceneContext,
    target_line: ReferenceLineView,
    memory: dict[str, Any],
) -> bool:
    """执行态完成判定：连续若干帧接近目标线中心。"""

    current_lane_id = str(context.ego_vehicle.current_ref_line.lane_id)
    target_lane_id = str(memory.get("locked_target_lane_id", ""))
    target_lane_sequence = set(_normalize_lane_sequence(memory.get("locked_target_lane_sequence", [])))
    if (target_lane_id and current_lane_id == target_lane_id) or (target_lane_sequence and current_lane_id in target_lane_sequence):
        memory["execute_complete_hits"] += 1
        return int(memory["execute_complete_hits"]) >= int(CFG["execute_complete_frames"])

    _, _, ego_d_on_target = project_point_to_path(
        np.array([context.ego_vehicle.x, context.ego_vehicle.y], dtype=float),
        target_line.line,
    )
    if abs(float(ego_d_on_target)) <= float(CFG["enter_target_d_thresh_m"]):
        memory["execute_complete_hits"] += 1
    else:
        memory["execute_complete_hits"] = 0

    return int(memory["execute_complete_hits"]) >= int(CFG["execute_complete_frames"])


def _build_stabilize_output(
    context: SceneContext,
    target_line: ReferenceLineView,
    memory: dict[str, Any],
) -> dict[str, Any]:
    """换道完成后的稳定跟随输出。"""

    target_v = min(
        max(float(getattr(context.main_vehicle, "v", memory.get("execute_target_v", 0.0))), 1.0),
        float(CFG["cruise_target_v"]),
    )
    travel_s = max(8.0, float(target_v) * 2.0 + 4.0)
    target_s = min(float(target_line.current_s) + float(travel_s), float(target_line.cumulative_s[-1]) - 0.5)
    target_xy, target_heading = get_path_point_at_s(target_line.line, target_s, target_line.cumulative_s)
    return {
        "x": float(target_xy[0]),
        "y": float(target_xy[1]),
        "v": float(target_v),
        "a": 0.0,
        "theta": float(target_heading),
        "t": 2.0,
        "target_line_index": int(target_line.line_index),
        "hold_t": 0.0,
        "reason": "cutin_stabilize",
    }


def _scene_fsm_step(
    context: SceneContext,
    memory: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    now = float(context.time_s)

    if memory["mode"] == "COOLDOWN":
        if now < float(memory["cooldown_until_time_s"]):
            return _baseline_keep_lane(context, reason="baseline_cooldown"), memory
        memory["mode"] = "CRUISE"

    if memory["mode"] not in ["EXECUTE", "STABILIZE"]:
        if not _has_valid_main_vehicle(context):
            memory["mode"] = "CRUISE"
            memory["search_hits"] = 0
            _reset_commit_memory(memory)
            return _baseline_keep_lane(context, reason="baseline_no_valid_main"), memory

    if memory["mode"] in ["CRUISE", "SEARCH"]:
        target_line = _choose_target_line(context)
        if target_line is None:
            memory["mode"] = "SEARCH"
            memory["search_hits"] = 0
            return _baseline_keep_lane(context, reason="baseline_no_target_line"), memory

        best_solution, _ = _search_best_cutin_solution(context, target_line)
        if best_solution is None:
            memory["mode"] = "SEARCH"
            memory["search_hits"] = 0
            return _baseline_keep_lane(context, reason="baseline_searching"), memory

        memory["search_hits"] += 1
        memory["last_best_solution"] = best_solution
        memory["mode"] = "SEARCH"
        if int(memory["search_hits"]) < int(CFG["commit_debounce_frames"]):
            return _baseline_keep_lane(context, reason="baseline_debounce"), memory

        _enter_execute_from_solution(memory, context.ego_vehicle.current_ref_line, target_line, best_solution, now)
        return best_solution, memory

    if memory["mode"] == "EXECUTE":
        target_line = _resolve_execute_target_line(context, memory)
        if target_line is None:
            memory["commit_fail_count"] += 1
            if int(memory["commit_fail_count"]) < int(CFG["commit_fail_max_frames"]):
                return _select_execute_fallback_output(
                    context,
                    memory,
                    candidate_solution=None,
                    reason="baseline_execute_line_lost",
                ), memory
            _enter_cooldown(memory, now)
            return _baseline_keep_lane(context, reason="baseline_execute_line_lost"), memory

        execute_output = _build_execute_output(context, memory, target_line)
        if execute_output is None:
            memory["commit_fail_count"] += 1
            if int(memory["commit_fail_count"]) < int(CFG["commit_fail_max_frames"]):
                return _select_execute_fallback_output(
                    context,
                    memory,
                    candidate_solution=None,
                    reason="baseline_execute_build_failed",
                ), memory
            _enter_cooldown(memory, now)
            return _baseline_keep_lane(context, reason="baseline_execute_build_failed"), memory

        hard_abort = False
        if _has_valid_main_vehicle_for_commit(context):
            hard_abort, _ = _check_execute_hard_abort(context, target_line, execute_output)
        if hard_abort:
            memory["commit_fail_count"] += 1
            if int(memory["commit_fail_count"]) >= int(CFG["commit_fail_max_frames"]):
                _enter_cooldown(memory, now)
                return _baseline_keep_lane(context, reason="baseline_commit_window_lost"), memory
            memory["last_best_solution"] = copy.deepcopy(execute_output)
            return _select_execute_fallback_output(
                context,
                memory,
                candidate_solution=execute_output,
                reason="baseline_execute_hold",
            ), memory

        memory["commit_fail_count"] = 0
        memory["last_best_solution"] = copy.deepcopy(execute_output)
        if _is_execute_completed(context, target_line, memory):
            memory["mode"] = "STABILIZE"
            memory["stable_until_time_s"] = float(now) + float(CFG["stabilize_duration_s"])
        return execute_output, memory

    if memory["mode"] == "STABILIZE":
        target_line = _resolve_execute_target_line(context, memory)
        if target_line is None:
            memory["mode"] = "CRUISE"
            _reset_commit_memory(memory)
            return _baseline_keep_lane(context, reason="baseline_stabilize_line_lost"), memory

        solution = _build_stabilize_output(context, target_line, memory)

        if now >= float(memory["stable_until_time_s"]):
            memory["mode"] = "CRUISE"
            _reset_commit_memory(memory)

        return solution, memory

    memory["mode"] = "CRUISE"
    return _baseline_keep_lane(context, reason="baseline_default"), memory


def scene_algorithm(context: SceneContext) -> dict[str, Any]:
    """Cut-in 场景算法主入口。

    第一版返回 ``dict``，由框架层补齐为 ``SceneAlgorithmOutput``。
    """

    global SCENE_MEMORY
    if int(context.step_index) == 0:
        _reset_scene_memory()
    output, SCENE_MEMORY = _scene_fsm_step(context, SCENE_MEMORY)
    return output
