"""场景目标距离与终点速度的动力学可行性解析。

场景算法每帧输出 ``(x, y, v, t)``：``(x, y)`` 决定几何路径长度 ``L``，
``v`` 是终点速度 ``vT``，``t`` 是规划时域 ``T``。时间参数化 QP 要求

    dt * sum(v[:-1]) = L

并且速度、加速度都满足平台限值。用 ``vT * T`` 估算 L 只在 ``v0 == vT``
时成立；减速时会低估刹车距离，加速时会高估可达距离，导致 QP 因 L 落在
可达区间外而报 infeasible。

本模块提供统一的目标解析，只负责“让场景目标自洽”，不放宽规划器的
动态硬门：

- 生成模式：给定 ``v0、vT、T``，裁出可达的 ``vT``，用
  ``L = (v0 + vT) / 2 * T`` 算自洽距离；
- 校验模式：给定固定 L、vT、T，检查是否存在可行速度曲线，必要时延长 T。

``max_plan_t`` 是允许延长时域的上限，不是每帧的规划时域本身。正常情况
下 T 原样返回；只有 vT 不可达或最小前进量不够时才会用到它。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

MIN_EXTERNAL_PLAN_T_S = 0.5
DEFAULT_MAX_PLAN_T_S = 2.0
FEASIBILITY_TIME_LIMIT_S = 0.01
DISTANCE_EPS_M = 1e-6
_SPEED_EPS_MPS = 1e-9

_SOLVER_CACHE: tuple[Any, Any] | None = None
_SOLVER_IMPORT_ERROR: Exception | None = None


@dataclass(frozen=True)
class TargetResolution:
    """一帧场景目标解析结果。"""

    v_end: float
    travel_s: float
    plan_t: float
    feasible: bool
    reason: str
    v_end_clipped: bool
    plan_t_extended: bool
    diagnostics: dict[str, Any]


def _clip(value: float, lower: float, upper: float) -> float:
    """把数值裁剪到上下界；上下界倒挂时返回下界。"""

    lower = float(lower)
    upper = float(upper)
    if upper < lower:
        return lower
    return float(min(max(float(value), lower), upper))


def _snap_plan_t(plan_t: float, dt: float) -> float:
    """把时域吸附到统一采样周期上。"""

    dt = max(float(dt), 1e-3)
    return max(dt, round(float(plan_t) / dt) * dt)


def _reachable_end_speed(
    v_start: float,
    plan_t: float,
    dt: float,
    a_min: float,
    a_max: float,
    v_min: float,
    v_max: float,
) -> tuple[float, float]:
    """返回 T 内可达的终点速度区间。

    QP 要求末端加速度为 0，最后两个采样点速度都等于 vT，因此真正可用的
    加速窗口是 ``T - dt``。
    """

    effective_t = max(float(dt), float(plan_t) - float(dt))
    lower = max(float(v_min), float(v_start) + float(a_min) * effective_t)
    upper = min(float(v_max), float(v_start) + float(a_max) * effective_t)
    return lower, upper


def _nominal_travel_s(v_start: float, v_end: float, plan_t: float) -> float:
    """匀变速自洽距离：平均速度乘时域。"""

    return max(0.0, 0.5 * (float(v_start) + float(v_end)) * float(plan_t))


def _distance_bounds(
    v_start: float,
    v_end: float,
    plan_t: float,
    a_min: float,
    a_max: float,
) -> tuple[float, float]:
    """给定端点速度时，连续匀变速下的最小/最大行驶距离。

    用于在固定距离的校验模式里反推可行的终点速度区间：距离对终点速度
    单调递增，因此可以二分出满足 ``L_min(vT) <= L <= L_max(vT)`` 的区间。
    """

    plan_t = max(0.0, float(plan_t))
    if plan_t <= 1e-9:
        return 0.0, 0.0
    v_start = float(v_start)
    v_end = float(v_end)
    if v_end >= v_start:
        if a_max <= 1e-9:
            return v_start * plan_t, v_start * plan_t
        accel_t = min(plan_t, (v_end - v_start) / float(a_max))
        mean_part = 0.5 * (v_start + v_end) * accel_t
        return (
            v_start * (plan_t - accel_t) + mean_part,
            mean_part + v_end * (plan_t - accel_t),
        )
    if a_min >= -1e-9:
        return v_start * plan_t, v_start * plan_t
    decel_t = min(plan_t, (v_start - v_end) / abs(float(a_min)))
    mean_part = 0.5 * (v_start + v_end) * decel_t
    return (
        mean_part + v_end * (plan_t - decel_t),
        v_start * (plan_t - decel_t) + mean_part,
    )


def _feasible_end_speed_candidates(
    *,
    v_start: float,
    travel_s: float,
    plan_t: float,
    lower: float,
    upper: float,
    requested: float,
    a_min: float,
    a_max: float,
) -> list[float]:
    """在可达速度区间内，按“离请求值最近”排序候选终点速度。

    请求的 ``vT`` 可能落在速度可达区间内、但对应距离不可达（例如参考线
    被截短后，``L`` 比匀速走完 ``T`` 还短）。这时不能直接判定目标不可行，
    而要在区间内换一个终点速度。先用连续距离界把区间收紧，再交给离散
    求解器逐一确认。
    """

    lower = float(lower)
    upper = float(upper)
    if upper < lower:
        return []
    target = float(travel_s)

    def distance_at(v_end: float) -> tuple[float, float]:
        return _distance_bounds(v_start, v_end, plan_t, a_min, a_max)

    # 距离上界随 vT 增大；找到使 L_max(vT) >= L 的最小 vT。
    lo, hi = lower, upper
    if distance_at(hi)[1] < target - DISTANCE_EPS_M:
        return []
    if distance_at(lo)[1] < target - DISTANCE_EPS_M:
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            if distance_at(mid)[1] < target - DISTANCE_EPS_M:
                lo = mid
            else:
                hi = mid
        v_lo = hi
    else:
        v_lo = lower

    # 距离下界随 vT 增大；找到使 L_min(vT) <= L 的最大 vT。
    lo, hi = lower, upper
    if distance_at(lo)[0] > target + DISTANCE_EPS_M:
        return []
    if distance_at(hi)[0] > target + DISTANCE_EPS_M:
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            if distance_at(mid)[0] > target + DISTANCE_EPS_M:
                hi = mid
            else:
                lo = mid
        v_hi = lo
    else:
        v_hi = upper

    if v_hi < v_lo - _SPEED_EPS_MPS:
        return []

    # 解析界与离散约束（末端加速度为 0 吃掉一步）有偏差，区间内取几个点
    # 让离散求解器确认；请求值本身优先。
    span = max(0.0, v_hi - v_lo)
    samples = [v_lo + span * frac for frac in (0.0, 0.25, 0.5, 0.75, 1.0)]
    clipped_request = _clip(requested, v_lo, v_hi)
    candidates = [clipped_request, *samples]
    unique: list[float] = []
    for value in candidates:
        value = _clip(value, v_lo, v_hi)
        if all(abs(value - seen) > 1e-6 for seen in unique):
            unique.append(value)
    unique.sort(key=lambda value: abs(value - float(requested)))
    return unique


def _analytic_interval(
    v_start: float,
    v_end: float,
    plan_t: float,
    a_min: float,
    a_max: float,
) -> tuple[float, float]:
    """连续匀变速下的可达距离区间；仅作为离散求解器不可用时的保守回退。"""

    if v_end >= v_start:
        delta = float(v_end) - float(v_start)
        if a_max > 1e-9:
            return (
                float(v_start) * plan_t + delta * delta / (2.0 * a_max),
                float(v_end) * plan_t - delta * delta / (2.0 * a_max),
            )
        return float(v_start) * plan_t, float(v_start) * plan_t

    delta = float(v_start) - float(v_end)
    if a_min < -1e-9:
        decel = abs(float(a_min))
        return (
            float(v_end) * plan_t + delta * delta / (2.0 * decel),
            float(v_start) * plan_t - delta * delta / (2.0 * decel),
        )
    return float(v_start) * plan_t, float(v_start) * plan_t


def _load_solver() -> tuple[Any, Any] | None:
    """惰性加载离散可行性求解器，避免场景模块导入时拉起优化栈。"""

    global _SOLVER_CACHE, _SOLVER_IMPORT_ERROR
    if _SOLVER_CACHE is not None or _SOLVER_IMPORT_ERROR is not None:
        return _SOLVER_CACHE
    try:
        from traj_planner.trajectory import (  # noqa: PLC0415
            VelocityProfileError,
            solve_feasible_velocity_profile,
        )
    except Exception as exc:  # pragma: no cover - 仅在依赖缺失时触发
        _SOLVER_IMPORT_ERROR = exc
        return None
    _SOLVER_CACHE = (solve_feasible_velocity_profile, VelocityProfileError)
    return _SOLVER_CACHE


def _discrete_feasible(
    *,
    travel_s: float,
    plan_t: float,
    dt: float,
    v_start: float,
    v_end: float,
    v_min: float,
    v_max: float,
    a_min: float,
    a_max: float,
) -> bool:
    """用规划器同一组约束检查 ``(L, T, v0, vT)`` 是否存在可行速度曲线。"""

    solver = _load_solver()
    if solver is None:
        lower, upper = _analytic_interval(v_start, v_end, plan_t, a_min, a_max)
        return lower - DISTANCE_EPS_M <= float(travel_s) <= upper + DISTANCE_EPS_M

    solve, profile_error = solver
    try:
        solve(
            float(travel_s),
            float(plan_t),
            float(dt),
            float(v_start),
            0.0,
            float(v_end),
            0.0,
            v_min=float(v_min),
            v_max=float(v_max),
            a_min=float(a_min),
            a_max=float(a_max),
            max_jerk=None,
            time_limit_s=FEASIBILITY_TIME_LIMIT_S,
        )
        return True
    except profile_error:
        return False
    except Exception:
        lower, upper = _analytic_interval(v_start, v_end, plan_t, a_min, a_max)
        return lower - DISTANCE_EPS_M <= float(travel_s) <= upper + DISTANCE_EPS_M


def _build_diagnostics(
    *,
    mode: str,
    requested_v_end: float,
    resolved_v_end: float,
    requested_plan_t: float,
    resolved_plan_t: float,
    requested_travel_s: float,
    resolved_travel_s: float,
    extra_reduced_m: float,
    reason: str,
) -> dict[str, Any]:
    """生成可直接写入 scene_debug 的诊断字段。"""

    return {
        "target_mode": str(mode),
        "target_requested_v_end_mps": float(requested_v_end),
        "target_resolved_v_end_mps": float(resolved_v_end),
        "target_v_end_clip_mps": float(resolved_v_end) - float(requested_v_end),
        "target_requested_plan_t_s": float(requested_plan_t),
        "target_resolved_plan_t_s": float(resolved_plan_t),
        "target_plan_t_extension_s": float(resolved_plan_t) - float(requested_plan_t),
        "target_requested_travel_s_m": float(requested_travel_s),
        "target_resolved_travel_s_m": float(resolved_travel_s),
        "target_extra_reduced_m": float(extra_reduced_m),
        "target_feasible": str(reason) in {"ok", "v_end_clipped"},
        "target_reason": str(reason),
    }


def _resolve_generated(
    *,
    v_start: float,
    v_end: float,
    plan_t: float,
    a_min: float,
    a_max: float,
    v_min: float,
    v_max: float,
    dt: float,
    min_progress: float,
    extra_progress: float,
    max_plan_t: float,
    min_plan_t: float,
) -> TargetResolution:
    """生成模式：先保住 vT，再算自洽 L，最后检查最小前进量。"""

    t0 = _snap_plan_t(max(float(plan_t), float(min_plan_t)), dt)
    max_t = max(t0, _snap_plan_t(max(float(max_plan_t), t0), dt))

    # 1. 优先延长 T 保住请求的终点速度；只有上限内都不可达才裁剪。
    t_use = None
    candidate = t0
    while candidate <= max_t + 1e-9:
        lower, upper = _reachable_end_speed(v_start, candidate, dt, a_min, a_max, v_min, v_max)
        if lower - _SPEED_EPS_MPS <= v_end <= upper + _SPEED_EPS_MPS:
            t_use = candidate
            break
        candidate = _snap_plan_t(candidate + dt, dt)

    if t_use is None:
        t_use = t0
        lower, upper = _reachable_end_speed(v_start, t_use, dt, a_min, a_max, v_min, v_max)
        v_res = _clip(v_end, lower, upper)
    else:
        v_res = float(v_end)
    v_clipped = abs(v_res - float(v_end)) > _SPEED_EPS_MPS

    # 2. 自洽距离 = 平均速度 * 时域，再加场景要求的前视余量。
    nominal = _nominal_travel_s(v_start, v_res, t_use)
    travel = nominal + float(extra_progress)
    extra_reduced = 0.0

    # 3. 前视余量可能把 L 推出可行区间；逐步回收，而不是让 QP 失败。
    if extra_progress > 0.0:
        while not _discrete_feasible(
            travel_s=travel,
            plan_t=t_use,
            dt=dt,
            v_start=v_start,
            v_end=v_res,
            v_min=v_min,
            v_max=v_max,
            a_min=a_min,
            a_max=a_max,
        ):
            if extra_reduced >= float(extra_progress) - 1e-9:
                travel = nominal
                break
            extra_reduced = min(float(extra_progress), extra_reduced + 0.1)
            travel = max(0.0, nominal + float(extra_progress) - extra_reduced)

    # 4. 最小前进量不够时继续延长 T，而不是输出一个走不到的点。
    while travel + DISTANCE_EPS_M < float(min_progress) and t_use < max_t - 1e-9:
        t_use = _snap_plan_t(t_use + dt, dt)
        lower, upper = _reachable_end_speed(v_start, t_use, dt, a_min, a_max, v_min, v_max)
        if v_clipped:
            v_res = _clip(v_res, lower, upper)
        else:
            v_res = _clip(v_end, lower, upper)
            v_clipped = abs(v_res - float(v_end)) > _SPEED_EPS_MPS
        nominal = _nominal_travel_s(v_start, v_res, t_use)
        travel = nominal + float(extra_progress) - extra_reduced

    feasible = travel + DISTANCE_EPS_M >= float(min_progress)
    reason = "ok" if feasible else "min_progress_unreachable"
    diagnostics = _build_diagnostics(
        mode="generate",
        requested_v_end=v_end,
        resolved_v_end=v_res,
        requested_plan_t=plan_t,
        resolved_plan_t=t_use,
        requested_travel_s=_nominal_travel_s(v_start, v_end, t0) + float(extra_progress),
        resolved_travel_s=travel,
        extra_reduced_m=extra_reduced,
        reason=reason,
    )
    return TargetResolution(
        v_end=float(v_res),
        travel_s=float(travel),
        plan_t=float(t_use),
        feasible=bool(feasible),
        reason=str(reason),
        v_end_clipped=bool(v_clipped),
        plan_t_extended=bool(t_use > t0 + 1e-9),
        diagnostics=diagnostics,
    )


def _resolve_fixed(
    *,
    v_start: float,
    v_end: float,
    plan_t: float,
    fixed_travel_s: float,
    a_min: float,
    a_max: float,
    v_min: float,
    v_max: float,
    dt: float,
    max_plan_t: float,
    min_plan_t: float,
) -> TargetResolution:
    """校验模式：目标点已锁定，搜索能让该 L 可行的时域。"""

    t0 = _snap_plan_t(max(float(plan_t), float(min_plan_t)), dt)
    max_t = max(t0, _snap_plan_t(max(float(max_plan_t), t0), dt))
    target = max(0.0, float(fixed_travel_s))

    # 距离被截短时可能需要更短的时域（短时域才能少走），也可能需要更长
    # 的时域（末端速度不可达）。先试请求时域，再按偏离量向外搜索。
    t_values: list[float] = []
    candidate = _snap_plan_t(max(float(min_plan_t), dt), dt)
    while candidate <= max_t + 1e-9:
        t_values.append(candidate)
        candidate = _snap_plan_t(candidate + dt, dt)
    if not t_values:
        t_values = [t0]
    t_values.sort(key=lambda value: (abs(value - float(plan_t)), value))

    def attempt(allow_clip: bool) -> TargetResolution | None:
        for candidate in t_values:
            lower, upper = _reachable_end_speed(v_start, candidate, dt, a_min, a_max, v_min, v_max)
            if allow_clip:
                v_values = [float(v_end)]
                if lower - _SPEED_EPS_MPS <= v_end <= upper + _SPEED_EPS_MPS:
                    # 请求速度在可达区间内但距离不可达时，换一个终点速度。
                    v_values.extend(
                        _feasible_end_speed_candidates(
                            v_start=v_start,
                            travel_s=target,
                            plan_t=candidate,
                            lower=lower,
                            upper=upper,
                            requested=v_end,
                            a_min=a_min,
                            a_max=a_max,
                        )
                    )
                else:
                    v_values.append(_clip(v_end, lower, upper))
            elif lower - _SPEED_EPS_MPS <= v_end <= upper + _SPEED_EPS_MPS:
                v_values = [float(v_end)]
            else:
                continue

            for v_res in v_values:
                if not _discrete_feasible(
                    travel_s=target,
                    plan_t=candidate,
                    dt=dt,
                    v_start=v_start,
                    v_end=v_res,
                    v_min=v_min,
                    v_max=v_max,
                    a_min=a_min,
                    a_max=a_max,
                ):
                    continue
                clipped = abs(v_res - float(v_end)) > _SPEED_EPS_MPS
                reason = "v_end_clipped" if clipped else "ok"
                diagnostics = _build_diagnostics(
                    mode="validate",
                    requested_v_end=v_end,
                    resolved_v_end=v_res,
                    requested_plan_t=plan_t,
                    resolved_plan_t=candidate,
                    requested_travel_s=target,
                    resolved_travel_s=target,
                    extra_reduced_m=0.0,
                    reason=reason,
                )
                return TargetResolution(
                    v_end=float(v_res),
                    travel_s=float(target),
                    plan_t=float(candidate),
                    feasible=True,
                    reason=reason,
                    v_end_clipped=bool(clipped),
                    plan_t_extended=bool(candidate > t0 + 1e-9),
                    diagnostics=diagnostics,
                )
        return None

    # 1. 先保终点速度；2. 保不住再在可达区间内寻找可行的终点速度。
    resolution = attempt(allow_clip=False) or attempt(allow_clip=True)
    if resolution is not None:
        return resolution

    diagnostics = _build_diagnostics(
        mode="validate",
        requested_v_end=v_end,
        resolved_v_end=v_end,
        requested_plan_t=plan_t,
        resolved_plan_t=t0,
        requested_travel_s=target,
        resolved_travel_s=target,
        extra_reduced_m=0.0,
        reason="target_unreachable",
    )
    return TargetResolution(
        v_end=float(v_end),
        travel_s=float(target),
        plan_t=float(t0),
        feasible=False,
        reason="target_unreachable",
        v_end_clipped=False,
        plan_t_extended=False,
        diagnostics=diagnostics,
    )


def resolve_target(
    *,
    v_start: float,
    v_end: float,
    plan_t: float,
    a_min: float,
    a_max: float,
    v_max: float,
    dt: float,
    v_min: float = 0.0,
    min_progress: float = 0.0,
    extra_progress: float = 0.0,
    max_plan_t: float = DEFAULT_MAX_PLAN_T_S,
    fixed_travel_s: float | None = None,
    min_plan_t: float = MIN_EXTERNAL_PLAN_T_S,
) -> TargetResolution:
    """解析一帧场景目标，返回动力学自洽的 ``(vT, L, T)``。

    ``fixed_travel_s`` 为 ``None`` 时走生成模式；否则走校验模式，保持该
    距离不变，只调整时域/终点速度。
    """

    dt = max(float(dt), 1e-3)
    v_start = max(0.0, float(v_start))
    v_min = max(0.0, float(v_min))
    v_end = max(v_min, float(v_end))
    v_max = max(float(v_max), v_start)
    min_plan_t = max(float(dt), float(min_plan_t))
    max_plan_t = max(float(plan_t), float(max_plan_t))

    if fixed_travel_s is not None:
        return _resolve_fixed(
            v_start=v_start,
            v_end=v_end,
            plan_t=float(plan_t),
            fixed_travel_s=float(fixed_travel_s),
            a_min=float(a_min),
            a_max=float(a_max),
            v_min=v_min,
            v_max=v_max,
            dt=dt,
            max_plan_t=max_plan_t,
            min_plan_t=min_plan_t,
        )

    return _resolve_generated(
        v_start=v_start,
        v_end=v_end,
        plan_t=float(plan_t),
        a_min=float(a_min),
        a_max=float(a_max),
        v_min=v_min,
        v_max=v_max,
        dt=dt,
        min_progress=max(0.0, float(min_progress)),
        extra_progress=max(0.0, float(extra_progress)),
        max_plan_t=max_plan_t,
        min_plan_t=min_plan_t,
    )
