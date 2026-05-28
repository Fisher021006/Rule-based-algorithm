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
    apply_time_param_to_candidate,
    build_candidate_smooth_path,
)
from traj_planner.sampling import build_candidate_templates, sample_terminal_states
from traj_planner.tools import project_point_to_path
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
        self.attribute["module_timing_ms"] = {}

    def _set_last_module_timing(self, timings: dict[str, float]) -> None:
        """保存单帧模块耗时摘要，便于离线落盘与可视化。"""

        normalized = {
            key: round(float(value), 3)
            for key, value in timings.items()
        }
        self.last_module_timing_ms = normalized
        self.attribute["module_timing_ms"] = dict(normalized)

    def plan(self, planning_input: PlanningInput):
        """执行一次完整的局部规划周期。

        输入：
            :class:`PlanningInput`

        输出：
            ``np.ndarray``，shape ``(N, 6)``
            如果所有候选都未通过硬约束，则返回 ``None``。
        """

        planning_input.planner_memory = self.memory
        if getattr(planning_input, "external_target", None) is not None:
            return self._plan_with_external_target(planning_input)
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
        best_candidate, candidates = evaluate_candidates(context, decision, candidates)
        timings["evaluator_ms"] = (time.perf_counter() - evaluator_start) * 1000.0

        refine_start = time.perf_counter()
        best_candidate, candidates = self._refine_best_candidate(context, decision, best_candidate, candidates)
        timings["refine_ms"] = (time.perf_counter() - refine_start) * 1000.0
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
            timings["plan_total_ms"] = (time.perf_counter() - plan_start) * 1000.0
            self._set_last_module_timing(timings)
            return None

        target_line = context.line_contexts[target_line_index]
        target_xy = np.asarray(external_target.point_xy, dtype=float)
        _, target_s, _ = project_point_to_path(target_xy, target_line.line)
        target_s = max(float(target_s), float(target_line.current_s) + 1.0)

        generator_start = time.perf_counter()
        terminal = TerminalCandidate(
            line_index=target_line_index,
            line_id=target_line.line_id,
            target_s=float(target_s),
            target_v=float(external_target.target_v),
            plan_T=max(0.5, float(external_target.plan_T)),
            hold_t=max(0.0, float(external_target.hold_t)),
            template_name="external_target",
        )
        candidate = build_candidate_smooth_path(context, terminal)
        candidate_osqp = apply_time_param_to_candidate(context, candidate, method="osqp")
        if candidate_osqp.trajectory is not None and len(candidate_osqp.trajectory) > 0:
            candidate = candidate_osqp
        else:
            candidate = apply_time_param_to_candidate(context, candidate, method="quintic")
        timings["generator_ms"] = (time.perf_counter() - generator_start) * 1000.0
        if candidate.trajectory is None or len(candidate.trajectory) == 0:
            timings["plan_total_ms"] = (time.perf_counter() - plan_start) * 1000.0
            self._set_last_module_timing(timings)
            return None

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
