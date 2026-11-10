"""阶段 1 局部规划总调度器。

这个模块本身不实现具体算法，而是把阶段 1 的流水线串起来：

PlanningInput
    -> build_planning_context
    -> decide_behavior
    -> build_candidate_templates
    -> sample_terminal_states
    -> build_candidate_smooth_path（对每个 candidate）
    -> quintic 时参 + evaluate_candidates
    -> best_quintic 的 OSQP refine
    -> 最优六元组轨迹

输出格式：
    ``np.ndarray``，shape ``(N, 6)``
    列顺序：``[x, y, heading, v, a, kappa]``
"""

from __future__ import annotations

import time

import numpy as np

from traj_planner.behavior import decide_behavior, update_behavior_memory
from traj_planner.context import build_planning_context
from traj_planner.evaluator import evaluate_candidates
from traj_planner.generator import (
    TIME_PARAM_DT,
    apply_time_param_to_candidate,
    build_candidate_smooth_path,
)
from traj_planner.sampling import build_candidate_templates, sample_terminal_states
from traj_planner.tools import project_point_to_path
from traj_planner.trajectory import (
    validate_velocity_profile, VelocityProfileError,
    PROFILE_EQUALITY_TOLERANCE, DYNAMIC_NUMERIC_TOLERANCE,
)
from traj_planner.types import BehaviorMemory, BehaviorMode, CandidateTrajectory, PlanningInput, PlanningOutput, TerminalCandidate


CONTINUATION_TEMPLATE_NAME = "continuation"
CONTINUATION_MATCH_DIST_M = 1.5
CONTINUATION_MATCH_HEADING_RAD = np.deg2rad(40.0)
CONTINUATION_DT = 0.1


def _normalize_angle(angle: float) -> float:
    """将角度归一化到 ``[-pi, pi]``。"""

    return (float(angle) + np.pi) % (2.0 * np.pi) - np.pi


class LocalPlanner:
    """阶段 1 局部规划总调度器类。

    上游：
        由 ``simulator/vehicle.GVT.traj_planning`` 调用，
        输入一份 :class:`PlanningInput`。

    下游：
        返回最终最优六元组轨迹给 ``GVT``。

    运行期状态：
        - ``memory``：跨帧行为承诺状态
        - ``attribute``：紧凑版调试摘要，供日志/可视化快速查看
        - ``last_output``：完整候选集合，供离线分析
    """

    def __init__(self, config):
        self.config = config
        self.memory = BehaviorMemory()
        self.attribute = {
            "plan_mode": self.memory.mode.value,
            "candidate_count": 0,
            "feasible_candidate_count": 0,
            "target_point": None,
            "selected_line_id": "",
            "selected_template": "",
            "selected_score": None,
            "failure_reason": "",
            "target_lane_id": "",
            "must_change": False,
            "deadline_margin_m": None,
            "reason": "",
            "best_mobil_incentive": 0.0,
            "current_front_ttc": None,
            "current_front_stop_margin": None,
            "current_front_gap": None,
            "desired_v": config["constrain"].get("target_v", config["constrain"]["max_v"]),
            "max_acc": self.config["constrain"]["max_acc"],
            "min_acc": self.config["constrain"]["min_acc"],
            "is_arrive": False,
            "is_on_lane": True,
        }
        self.last_output: PlanningOutput | None = None
        self.last_context = None
        self.last_templates = []
        self.last_terminals = []
        self.last_best_candidate: CandidateTrajectory | None = None
        self.last_module_timing_ms: dict[str, float] = {}
        self._frozen_external_cache = None
        self.attribute["module_timing_ms"] = {}

    def _set_last_module_timing(self, timings: dict[str, float]) -> None:
        """保存单帧模块耗时摘要，便于离线落盘与可视化。"""

        normalized = {
            key: round(float(value), 3)
            for key, value in timings.items()
        }
        self.last_module_timing_ms = normalized
        self.attribute["module_timing_ms"] = dict(normalized)

    def _reset_generator_diagnostics(self):
        self.attribute.update(
            osqp_diagnostics={}, quintic_diagnostics={}, highs_diagnostics={},
            osqp_accepted_nonconverged=False, osqp_solver_status="", highs_solver_status="",
            osqp_failure_reason="", osqp_error_type="", fallback_failure_reason="",
            fallback_error_type="", highs_failure_reason="", highs_error_type="",
            trajectory_generation_attempts=[], time_param_method="", target_point=None,
            feasible_candidate_count=0,
            frozen_suffix_rejection_reason="",
        )

    def invalidate_external_trajectory(self):
        """Discard frozen execution when a separate control mode takes over."""
        self._frozen_external_cache = None

    @staticmethod
    def _external_dynamics_valid(context, trajectory: np.ndarray | None, dt=TIME_PARAM_DT) -> bool:
        """所有输出使用同一份运行限值，含点加速度、差分加速度及可选 jerk。"""
        if trajectory is None:
            return False
        trajectory = np.asarray(trajectory)
        if (trajectory.ndim != 2 or trajectory.shape[1] != 6
                or not np.all(np.isfinite(trajectory))):
            return False
        limits = context.config["constrain"]
        try:
            validate_velocity_profile(
                trajectory[:, 3], float(context.ego_state.v), float(context.ego_state.a), dt,
                float(limits.get("min_v", 0.0)), float(limits["max_v"]),
                float(limits["min_acc"]), float(limits["max_acc"]),
                max_jerk=limits.get("max_jerk"), acceleration=trajectory[:, 4])
        except (VelocityProfileError, ValueError, TypeError):
            return False
        return True

    def plan(self, planning_input: PlanningInput):
        """执行一次完整的局部规划周期。

        输入：
            :class:`PlanningInput`

        输出：
            ``np.ndarray``，shape ``(N, 6)``
            如果所有候选都未通过硬约束，则返回 ``None``。
        """

        self._reset_generator_diagnostics()
        planning_input.planner_memory = self.memory
        if getattr(planning_input, "external_target", None) is not None:
            return self._plan_with_external_target(planning_input)
        self._frozen_external_cache = None
        plan_start = time.perf_counter()
        timings = {
            "context_ms": 0.0,
            "behavior_ms": 0.0,
            "sampling_ms": 0.0,
            "generator_ms": 0.0,
            "evaluator_ms": 0.0,
            "refine_ms": 0.0,
            "plan_total_ms": 0.0,
        }

        context_start = time.perf_counter()
        context = build_planning_context(planning_input)
        timings["context_ms"] = (time.perf_counter() - context_start) * 1000.0

        behavior_start = time.perf_counter()
        decision = decide_behavior(context)
        timings["behavior_ms"] = (time.perf_counter() - behavior_start) * 1000.0
        if decision.mode == BehaviorMode.ARRIVED:
            self.memory = update_behavior_memory(context, decision, None)
            output = PlanningOutput(
                best_trajectory=None,
                decision=decision,
                candidates=[],
                planner_memory=self.memory,
            )
            self.last_context = context
            self.last_templates = []
            self.last_terminals = []
            self.last_output = output
            self.last_best_candidate = None
            timings["plan_total_ms"] = (time.perf_counter() - plan_start) * 1000.0
            self._set_last_module_timing(timings)
            self._update_attribute(output, None)
            return None

        sampling_start = time.perf_counter()
        templates = build_candidate_templates(context, decision)
        terminals = sample_terminal_states(context, decision, templates)
        timings["sampling_ms"] = (time.perf_counter() - sampling_start) * 1000.0

        generator_start = time.perf_counter()
        candidates = [
            apply_time_param_to_candidate(
                context,
                build_candidate_smooth_path(context, terminal),
                method="quintic",
            )
            for terminal in terminals
        ]
        continuation_candidate = self._build_continuation_candidate(context)
        if continuation_candidate is not None:
            candidates.append(continuation_candidate)
            terminals.append(continuation_candidate.terminal)
        timings["generator_ms"] = (time.perf_counter() - generator_start) * 1000.0

        evaluator_start = time.perf_counter()
        checked_candidates = []
        for candidate in candidates:
            if self._external_dynamics_valid(context, candidate.trajectory):
                checked_candidates.append(candidate)
            else:
                candidate.feasible = False
                candidate.failure_reason = candidate.failure_reason or "trajectory_dynamic_limit_violation"
        best_candidate, _ = evaluate_candidates(context, decision, checked_candidates)
        timings["evaluator_ms"] = (time.perf_counter() - evaluator_start) * 1000.0

        refine_start = time.perf_counter()
        best_candidate, candidates = self._refine_best_candidate(context, decision, best_candidate, candidates)
        timings["refine_ms"] = (time.perf_counter() - refine_start) * 1000.0
        if best_candidate is not None and not self._external_dynamics_valid(context, best_candidate.trajectory):
            best_candidate.feasible = False
            best_candidate.failure_reason = "trajectory_dynamic_limit_violation"
            best_candidate = None
        self.memory = update_behavior_memory(context, decision, best_candidate)

        output = PlanningOutput(
            best_trajectory=None if best_candidate is None else np.asarray(best_candidate.trajectory, dtype=float),
            decision=decision,
            candidates=candidates,
            planner_memory=self.memory,
        )
        self.last_context = context
        self.last_templates = list(templates)
        self.last_terminals = list(terminals)
        self.last_output = output
        self.last_best_candidate = best_candidate
        timings["plan_total_ms"] = (time.perf_counter() - plan_start) * 1000.0
        self._set_last_module_timing(timings)
        self._update_attribute(output, best_candidate)

        if best_candidate is None:
            return None
        return np.asarray(best_candidate.trajectory, dtype=float)

    @staticmethod
    def _frozen_external_contract(context, target):
        """Only an explicit, time-indexed frozen request can reuse an execution path."""
        meta = target.metadata or {}
        if not meta.get("preserve_trajectory", False):
            return None
        try:
            now = float(meta["trajectory_execution_time_s"])
            deadline = float(meta["trajectory_deadline_s"])
            duration = float(target.plan_T)
            route = tuple(str(item) for item in meta["trajectory_route_lane_ids"])
            xy = np.asarray(target.point_xy, dtype=float)
            line = context.line_contexts[target.target_line_index]
            line_id = line.line_id
            heading = None if target.target_heading is None else float(target.target_heading)
            values = [now, deadline, duration, target.target_v, target.target_a, target.hold_t, *xy]
            if heading is not None:
                values.append(heading)
            if (xy.shape != (2,) or not np.isfinite(values).all() or not route
                    or line_id not in route or duration <= 0.0 or target.hold_t != 0.0
                    or abs(deadline - now - duration) > 1e-6
                    or abs(duration / TIME_PARAM_DT - round(duration / TIME_PARAM_DT)) > 1e-6):
                return None
            key = (deadline, *xy, float(target.target_v), float(target.target_a), heading, route)
            limit_names = ("min_v", "max_v", "min_acc", "max_acc", "max_jerk", "max_kappa", "max_acc_h")
            limits = tuple((name, context.config["constrain"].get(name)) for name in limit_names)
            return {"key": key, "now": now, "limits": limits,
                    "route_sequence": tuple(line.lane_sequence), "line_id": line_id}
        except (KeyError, TypeError, ValueError, IndexError):
            return None

    def _frozen_external_suffix(self, context, terminal, contract):
        cache = self._frozen_external_cache
        if contract is None or cache is None:
            self._frozen_external_cache = None
            return None

        def reject(reason):
            self.attribute["frozen_suffix_rejection_reason"] = reason
            self._frozen_external_cache = None
            return None

        if contract["key"] != cache["key"]:
            return reject("request_changed")
        if contract["limits"] != cache["limits"]:
            return reject("limits_changed")
        old_route, new_route = cache["route_sequence"], contract["route_sequence"]
        occurrences = [i for i, lane in enumerate(old_route) if lane == contract["line_id"]]
        if (len(occurrences) != 1 or not new_route or new_route[0] != contract["line_id"]):
            return reject("route_changed")
        known_suffix = old_route[occurrences[0]:]
        if new_route[:len(known_suffix)] != known_suffix:
            return reject("route_changed")
        elapsed_steps = (contract["now"] - cache["epoch"]) / TIME_PARAM_DT
        index = int(round(elapsed_steps))
        trajectory = cache["trajectory"]
        previous_index = cache["last_index"]
        if (abs(elapsed_steps - index) > 1e-6 or index <= previous_index
                or index >= len(trajectory) - 1
                or abs((len(trajectory) - 1 - index) * TIME_PARAM_DT - terminal.plan_T) > 1e-6):
            return reject("time_mismatch")
        point = trajectory[index]
        ego = context.ego_state
        # Runtime feedback reports acceleration over the interval just executed;
        # trajectory[row,4] describes the next interval, so those are not interchangeable.
        executed_a = (point[3] - cache["last_observed_v"]) / ((index - previous_index) * TIME_PARAM_DT)
        tol = PROFILE_EQUALITY_TOLERANCE
        if (not np.isfinite([ego.x, ego.y, ego.heading, ego.v, ego.a]).all()
                or np.linalg.norm(point[:2] - [ego.x, ego.y]) > tol
                or abs(_normalize_angle(point[2] - ego.heading)) > 1e-3
                or abs(point[3] - ego.v) > tol
                or abs(executed_a - ego.a) > DYNAMIC_NUMERIC_TOLERANCE):
            return reject("state_mismatch")
        suffix = trajectory[index:].copy()
        if not self._external_dynamics_valid(context, suffix):
            return reject("dynamic_limits")
        cache.update(last_index=index, last_observed_v=float(ego.v))
        return CandidateTrajectory(
            terminal=terminal, trajectory=suffix, smooth_path=suffix[:, :3].copy(), feasible=True,
            metrics={"time_param_method": "frozen_suffix", "constraint_check_passed": True,
                     "frozen_source_method": cache["method"], "frozen_source_time_s": cache["epoch"],
                     "frozen_suffix_index": index},
        )

    def _plan_with_external_target(self, planning_input: PlanningInput):
        """使用外部终端目标直接生成一条轨迹。"""
        external_target = planning_input.external_target
        if external_target is None:
            return None

        plan_start = time.perf_counter()
        timings = {
            "context_ms": 0.0,
            "behavior_ms": 0.0,
            "sampling_ms": 0.0,
            "generator_ms": 0.0,
            "evaluator_ms": 0.0,
            "refine_ms": 0.0,
            "plan_total_ms": 0.0,
        }

        context_start = time.perf_counter()
        context = build_planning_context(planning_input)
        timings["context_ms"] = (time.perf_counter() - context_start) * 1000.0
        target_line_index = int(external_target.target_line_index)
        if target_line_index < 0 or target_line_index >= len(context.line_contexts):
            self._frozen_external_cache = None
            timings["plan_total_ms"] = (time.perf_counter() - plan_start) * 1000.0
            self._set_last_module_timing(timings)
            return None

        target_line = context.line_contexts[target_line_index]
        metadata = dict(getattr(external_target, "metadata", {}) or {})
        if "time_param_solver_tolerance" in metadata:
            context.config = {**context.config, "constrain": {
                **context.config["constrain"],
                "time_param_solver_tolerance": float(metadata["time_param_solver_tolerance"]),
            }}
        self.attribute.update(plan_mode="external_target", failure_reason="",
                              osqp_failure_reason="", fallback_failure_reason="",
                              osqp_error_type="", osqp_solver_status="", fallback_error_type="",
                              osqp_diagnostics={}, osqp_accepted_nonconverged=False,
                              external_dynamic_limits_enforced=True,
                              candidate_count=1, feasible_candidate_count=0,
                              target_point=None, time_param_method="",
                              selected_line_id=target_line.line_id, target_lane_id=target_line.line_id,
                              requested_target_point=np.asarray(external_target.point_xy, dtype=float).tolist(),
                              attempted_time_param_method="")
        min_target_progress = max(0.0, float(metadata.get("min_external_target_progress_m", 1.0)))
        min_plan_t = max(0.0, float(metadata.get("min_external_plan_t_s", 0.5)))
        target_xy = np.asarray(external_target.point_xy, dtype=float)
        _, target_s, _ = project_point_to_path(target_xy, target_line.line)
        target_s = max(float(target_s), float(target_line.current_s) + min_target_progress)

        generator_start = time.perf_counter()
        terminal = TerminalCandidate(
            line_index=target_line_index,
            line_id=target_line.line_id,
            target_s=float(target_s),
            target_v=float(external_target.target_v),
            plan_T=max(min_plan_t, float(external_target.plan_T)),
            hold_t=max(0.0, float(external_target.hold_t)),
            template_name="external_target",
            target_xy_override=(target_xy.copy() if target_line_index == 0
                                and bool(metadata.get("maintain_lateral_offset", False)) else None),
            target_path_length=(float(metadata["lane_path_distance_m"]) if target_line_index == 0
                                and bool(metadata.get("maintain_lateral_offset", False))
                                and "lane_path_distance_m" in metadata else None),
        )
        self.attribute.update(external_target_v=float(terminal.target_v),
                              external_target_plan_T=float(terminal.plan_T))
        contract = self._frozen_external_contract(context, external_target)
        candidate = self._frozen_external_suffix(context, terminal, contract)
        if candidate is not None:
            self.attribute["attempted_time_param_method"] = "frozen_suffix"
            self.attribute["trajectory_generation_attempts"].append({
                "method": "frozen_suffix", "accepted": True, "failure_reason": "",
                "wall_ms": (time.perf_counter() - generator_start) * 1000.0,
                "diagnostics": dict(candidate.metrics),
            })
        else:
            geometry_candidate = build_candidate_smooth_path(context, terminal)
            for method in ("osqp", "quintic", "highs"):
                self.attribute["attempted_time_param_method"] = method
                attempt_start = time.perf_counter()
                attempt = apply_time_param_to_candidate(context, geometry_candidate, method=method)
                accepted = self._external_dynamics_valid(context, attempt.trajectory)
                diagnostics = dict(attempt.metrics)
                reason = "" if accepted else (attempt.failure_reason or "external_trajectory_dynamic_limit_violation")
                self.attribute["trajectory_generation_attempts"].append({
                    "method": method, "accepted": bool(accepted), "failure_reason": reason,
                    "wall_ms": (time.perf_counter() - attempt_start) * 1000.0,
                    "diagnostics": diagnostics,
                })
                self.attribute[f"{method}_diagnostics"] = diagnostics
                if method in ("osqp", "highs"):
                    self.attribute[f"{method}_solver_status"] = diagnostics.get("solver_status", "")
                if accepted:
                    candidate = attempt
                    self.attribute["osqp_accepted_nonconverged"] = bool(
                        method == "osqp" and diagnostics.get("accepted_nonconverged", False))
                    break
                prefix = "fallback" if method == "quintic" else method
                self.attribute[f"{prefix}_failure_reason"] = reason
                self.attribute[f"{prefix}_error_type"] = diagnostics.get("time_param_error_type", "constraint_violation")
        timings["generator_ms"] = (time.perf_counter() - generator_start) * 1000.0
        if candidate is None:
            self.attribute["failure_reason"] = "external_trajectory_dynamic_limit_violation"
            timings["plan_total_ms"] = (time.perf_counter() - plan_start) * 1000.0
            self._set_last_module_timing(timings)
            return None

        if contract is not None and candidate.metrics.get("time_param_method") != "frozen_suffix":
            self._frozen_external_cache = {
                **contract, "epoch": contract["now"], "last_index": 0,
                "last_observed_v": float(context.ego_state.v),
                "trajectory": np.asarray(candidate.trajectory, dtype=float).copy(),
                "method": str(candidate.metrics.get("time_param_method", "")),
            }

        self.last_context = context
        self.last_templates = []
        self.last_terminals = [terminal]
        self.last_best_candidate = candidate
        self.last_output = None
        timings["plan_total_ms"] = (time.perf_counter() - plan_start) * 1000.0
        self._set_last_module_timing(timings)
        self.attribute.update(
            {
                "plan_mode": "external_target",
                "candidate_count": 1,
                "feasible_candidate_count": 1,
                "target_point": np.asarray(candidate.trajectory[-1, :2], dtype=float).tolist(),
                "external_target_v": float(terminal.target_v),
                "external_target_plan_T": float(terminal.plan_T),
                "time_param_method": str(candidate.metrics.get("time_param_method", "")),
                "selected_line_id": terminal.line_id,
                "selected_template": terminal.template_name,
                "selected_score": None,
                "failure_reason": candidate.failure_reason,
                "target_lane_id": terminal.line_id,
                "must_change": False,
                "deadline_margin_m": None,
                "reason": str(getattr(external_target, "source", "external_target")),
                "best_mobil_incentive": 0.0,
                "current_front_ttc": None,
                "current_front_stop_margin": None,
                "current_front_gap": None,
                "is_arrive": False,
                "is_on_lane": terminal.line_index == 0,
            }
        )
        return np.asarray(candidate.trajectory, dtype=float)

    def _refine_best_candidate(
        self,
        context,
        decision,
        best_candidate: CandidateTrajectory | None,
        candidates: list[CandidateTrajectory],
    ) -> tuple[CandidateTrajectory | None, list[CandidateTrajectory]]:
        """对 quintic 阶段选出的最优候选做一次 OSQP refine。"""

        if best_candidate is None:
            return None, candidates
        if best_candidate.terminal.template_name == CONTINUATION_TEMPLATE_NAME:
            best_candidate.metrics["refine_status"] = "skipped_continuation"
            return best_candidate, candidates

        best_candidate.metrics["refine_status"] = "attempted"
        refined_candidate = apply_time_param_to_candidate(context, best_candidate, method="osqp")
        if refined_candidate.trajectory is None or len(refined_candidate.trajectory) == 0:
            best_candidate.metrics["refine_status"] = "solver_failed"
            best_candidate.metrics["refine_failure_reason"] = refined_candidate.failure_reason
            return best_candidate, candidates

        refined_candidate.metrics["coarse_score"] = float(best_candidate.score)
        refined_best, _ = evaluate_candidates(context, decision, [refined_candidate])
        if refined_best is None:
            best_candidate.metrics["refine_status"] = "rejected_after_refine"
            best_candidate.metrics["refine_failure_reason"] = refined_candidate.failure_reason
            return best_candidate, candidates

        refined_best.metrics["refine_status"] = "accepted"
        winner_idx = self._find_candidate_index(candidates, best_candidate)
        if winner_idx is not None:
            candidates[winner_idx] = refined_best
        return refined_best, candidates

    def _build_continuation_candidate(self, context) -> CandidateTrajectory | None:
        """把上一帧最优轨迹的剩余后缀包装成一个额外候选。"""

        previous = self.last_best_candidate
        if previous is None or previous.trajectory is None:
            return None

        prev_traj = np.asarray(previous.trajectory, dtype=float)
        if prev_traj.ndim != 2 or prev_traj.shape[1] < 6 or len(prev_traj) < 3:
            return None

        ego_state = context.ego_state
        ego_xy = np.array([ego_state.x, ego_state.y], dtype=float)
        point_distances = np.linalg.norm(prev_traj[:, :2] - ego_xy, axis=1)
        nearest_idx = int(np.argmin(point_distances))
        if nearest_idx >= len(prev_traj) - 1:
            return None
        if float(point_distances[nearest_idx]) > CONTINUATION_MATCH_DIST_M:
            return None

        heading_error = abs(_normalize_angle(prev_traj[nearest_idx, 2] - float(ego_state.heading)))
        if heading_error > CONTINUATION_MATCH_HEADING_RAD:
            return None

        line_context = self._find_line_context_by_lane_id(context, previous.terminal.line_id)
        if line_context is None:
            return None

        suffix = np.asarray(prev_traj[nearest_idx:], dtype=float).copy()
        suffix[0, :] = np.array(
            [
                float(ego_state.x),
                float(ego_state.y),
                float(ego_state.heading),
                float(ego_state.v),
                float(ego_state.a),
                float(ego_state.kappa),
            ],
            dtype=float,
        )
        suffix = self._deduplicate_trajectory_points(suffix)
        if len(suffix) < 2:
            return None

        _, target_s, _ = project_point_to_path(suffix[-1, :2], line_context.line)
        terminal = TerminalCandidate(
            line_index=int(line_context.line_index),
            line_id=line_context.line_id,
            target_s=float(target_s),
            target_v=float(suffix[-1, 3]),
            plan_T=max(CONTINUATION_DT, CONTINUATION_DT * (len(suffix) - 1)),
            hold_t=0.0,
            template_name=CONTINUATION_TEMPLATE_NAME,
        )
        return CandidateTrajectory(
            terminal=terminal,
            trajectory=suffix,
            smooth_path=np.asarray(suffix[:, :3], dtype=float),
            feasible=True,
            metrics={"continuation_source_template": previous.terminal.template_name},
        )

    @staticmethod
    def _deduplicate_trajectory_points(traj: np.ndarray) -> np.ndarray:
        """移除对后续评估没有意义的重复状态点。"""

        if len(traj) <= 1:
            return traj

        kept_points = [traj[0]]
        for point in traj[1:]:
            prev = kept_points[-1]
            same_xy = np.linalg.norm(point[:2] - prev[:2]) <= 1e-6
            same_heading = abs(_normalize_angle(point[2] - prev[2])) <= 1e-6
            same_dyn = np.max(np.abs(point[3:] - prev[3:])) <= 1e-6
            if same_xy and same_heading and same_dyn:
                continue
            kept_points.append(point)
        return np.asarray(kept_points, dtype=float)

    @staticmethod
    def _find_line_context_by_lane_id(context, lane_id: str):
        """在当前可用参考线中找回指定 lane_id 对应的 line_context。"""

        if not lane_id:
            return None
        for line_context in context.line_contexts:
            if line_context.line_id == lane_id:
                return line_context
        return None

    @staticmethod
    def _find_candidate_index(candidates: list[CandidateTrajectory], target: CandidateTrajectory) -> int | None:
        """按对象身份在候选列表中定位目标。"""

        for idx, candidate in enumerate(candidates):
            if candidate is target:
                return idx
        return None

    def _update_attribute(self, output: PlanningOutput, best_candidate):
        """在单帧规划结束后更新紧凑调试摘要。"""

        feasible_count = sum(1 for candidate in output.candidates if candidate.feasible)
        best_point = None
        selected_line_id = ""
        selected_template = ""
        selected_score = None
        failure_reason = ""
        is_on_lane = True

        if best_candidate is not None and best_candidate.trajectory is not None and len(best_candidate.trajectory) > 0:
            best_point = np.asarray(best_candidate.trajectory[-1, :2], dtype=float).tolist()
            selected_line_id = best_candidate.terminal.line_id
            selected_template = best_candidate.terminal.template_name
            selected_score = float(best_candidate.score)
            failure_reason = best_candidate.failure_reason
            is_on_lane = best_candidate.terminal.line_index == 0
        elif output.candidates:
            failure_reason = ",".join(
                sorted({
                    candidate.failure_reason
                    for candidate in output.candidates
                    if candidate.failure_reason
                })
            )

        self.attribute.update(
            {
                "plan_mode": output.decision.mode.value,
                "candidate_count": len(output.candidates),
                "feasible_candidate_count": feasible_count,
                "target_point": best_point,
                "selected_line_id": selected_line_id,
                "selected_template": selected_template,
                "selected_score": selected_score,
                "failure_reason": failure_reason,
                "target_lane_id": output.decision.target_lane_id,
                "must_change": output.decision.must_change,
                "deadline_margin_m": output.decision.deadline_margin_m,
                "reason": output.decision.reason,
                "best_mobil_incentive": float(output.decision.best_mobil_incentive),
                "current_front_ttc": float(output.decision.current_front_ttc),
                "current_front_stop_margin": float(output.decision.current_front_stop_margin),
                "current_front_gap": float(output.decision.current_front_gap),
                "is_arrive": output.decision.mode.value in ("approach_goal", "arrived"),
                "is_on_lane": is_on_lane,
            }
        )
