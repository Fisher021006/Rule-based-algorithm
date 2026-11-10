import numpy as np
import time
from traj_planner.tools import normalize_angle
import osqp
from scipy.sparse import csc_matrix, diags
# 启动时加载 HiGHS 所属模块，避免首次兜底把导入耗时计入规划周期。
from scipy import optimize as scipy_optimize


DYNAMIC_NUMERIC_TOLERANCE = 0.01
PROFILE_EQUALITY_TOLERANCE = 1e-3
DEFAULT_HIGHS_TIME_LIMIT_S = 0.05


class VelocityProfileError(RuntimeError):
    """区分当前候选不可行、数值未收敛及结果违规，不推断整个场景无解。"""

    def __init__(self, kind, message, solver_status=""):
        super().__init__(message)
        self.kind = kind
        self.solver_status = solver_status


def _validate_dynamic_limits(v_min, v_max, a_min, a_max, max_jerk):
    if (not np.all(np.isfinite([v_min, v_max, a_min, a_max]))
            or v_min > v_max or a_min > a_max):
        raise VelocityProfileError("invalid_input", "Invalid velocity/acceleration limits")
    if max_jerk is not None and (np.isnan(max_jerk) or max_jerk < 0.0):
        raise VelocityProfileError("invalid_input", "max_jerk must be non-negative or positive infinity")
    # 正无穷表示显式关闭 jerk 硬约束与验收；有限值仍按配置生效。
    return None if max_jerk is None or np.isposinf(max_jerk) else float(max_jerk)


def validate_velocity_profile(v, start_v, start_a, dt, v_min, v_max,
                              a_min, a_max, max_jerk=None, acceleration=None):
    """按调用方提供的同一组限值复核速度、差分加速度和 jerk。"""
    max_jerk = _validate_dynamic_limits(v_min, v_max, a_min, a_max, max_jerk)
    if not np.all(np.isfinite([start_v, start_a, dt])) or dt <= 0.0:
        raise VelocityProfileError("invalid_input", "Invalid initial state or sample period")
    v = np.asarray(v, dtype=float)
    if v.ndim != 1 or len(v) < 2 or not np.all(np.isfinite(v)):
        raise VelocityProfileError("constraint_violation", "Invalid velocity profile")
    tol = DYNAMIC_NUMERIC_TOLERANCE
    if abs(float(v[0]) - start_v) > tol:
        raise VelocityProfileError("constraint_violation", "Initial velocity mismatch")
    if np.min(v) < v_min - tol or np.max(v) > v_max + tol:
        raise VelocityProfileError("constraint_violation", "Velocity limit exceeded")
    finite_acc = np.diff(v) / dt
    accelerations = [finite_acc]
    if acceleration is not None:
        point_acc = np.asarray(acceleration, dtype=float)
        if point_acc.shape != v.shape or not np.all(np.isfinite(point_acc)):
            raise VelocityProfileError("constraint_violation", "Invalid point acceleration")
        accelerations.append(point_acc)
    for acc in accelerations:
        if np.min(acc) < a_min - tol or np.max(acc) > a_max + tol:
            raise VelocityProfileError("constraint_violation", "Acceleration limit exceeded")
        if max_jerk is not None:
            jerk = np.diff(np.r_[start_a, acc]) / dt
            if np.max(np.abs(jerk)) > max_jerk + tol:
                raise VelocityProfileError("constraint_violation", "Jerk limit exceeded (including initial splice)")


def calculate_path_cumulative_distance(path):
    """
    计算路径点的累积弧长
    path: (N, 3) 或 (N, 2) 数组，若为3列则第三列为航向角（仅用于长度计算时不使用）
    返回：累积弧长数组 (N,)
    """
    points = path[:, :2]  # 只取坐标
    diffs = np.diff(points, axis=0)
    seg_lengths = np.sqrt(np.sum(diffs ** 2, axis=1))
    cum_s = np.concatenate(([0.0], np.cumsum(seg_lengths)))
    return cum_s


def solve_quintic_coeffs(x0, v0, a0, xf, vf, af, T):
    """
    求解五次多项式 s(t) = c0 + c1*t + c2*t^2 + c3*t^3 + c4*t^4 + c5*t^5 的系数
    满足边界条件：
        s(0)=x0, s'(0)=v0, s''(0)=a0
        s(T)=xf, s'(T)=vf, s''(T)=af
    """
    if T <= 0:
        raise ValueError("Total time T must be positive")

    c0 = x0
    c1 = v0
    c2 = a0 / 2.0

    T2 = T * T
    T3 = T2 * T
    T4 = T3 * T
    T5 = T4 * T

    A = np.array([
        [T3, T4, T5],
        [3.0 * T2, 4.0 * T3, 5.0 * T4],
        [6.0 * T, 12.0 * T2, 20.0 * T3]
    ], dtype=float)

    b = np.array([
        xf - c0 - c1 * T - c2 * T2,
        vf - c1 - 2.0 * c2 * T,
        af - 2.0 * c2
    ], dtype=float)

    c3, c4, c5 = np.linalg.solve(A, b)
    return np.array([c0, c1, c2, c3, c4, c5])


def eval_quintic(coeffs, t):
    """
    计算五次多项式在时间 t 处的值 s(t), 速度 v(t), 加速度 a(t)
    t 可以是标量或数组
    """
    c0, c1, c2, c3, c4, c5 = coeffs
    t = np.asarray(t)
    s = c0 + c1 * t + c2 * t ** 2 + c3 * t ** 3 + c4 * t ** 4 + c5 * t ** 5
    v = c1 + 2.0 * c2 * t + 3.0 * c3 * t ** 2 + 4.0 * c4 * t ** 3 + 5.0 * c5 * t ** 4
    a = 2.0 * c2 + 6.0 * c3 * t + 12.0 * c4 * t ** 2 + 20.0 * c5 * t ** 3
    return s, v, a


def interp_pose_on_path(path, cum_s, query_s):
    """
    根据弧长 query_s 在路径上线性插值，返回 (x, y, heading)
    path: (N, 3) 数组 [x, y, heading]
    cum_s: 累积弧长 (N,)
    query_s: 目标弧长（标量）
    """
    query_s = np.clip(query_s, cum_s[0], cum_s[-1])
    idx = np.searchsorted(cum_s, query_s, side='right') - 1
    idx = np.clip(idx, 0, len(path) - 2)

    s0, s1 = cum_s[idx], cum_s[idx + 1]
    if s1 - s0 < 1e-6:
        ratio = 0.0
    else:
        ratio = (query_s - s0) / (s1 - s0)

    p0, p1 = path[idx], path[idx + 1]

    x = p0[0] + ratio * (p1[0] - p0[0])
    y = p0[1] + ratio * (p1[1] - p0[1])

    # 航向角插值，注意角度跳变处理
    dtheta = normalize_angle(p1[2] - p0[2])
    theta = normalize_angle(p0[2] + ratio * dtheta)

    return x, y, theta


def calculate_trajectory_kappa(trajectory):
    """
    根据轨迹点计算曲率 kappa（三点法）
    trajectory: list of [x, y, heading, v, a] 或类似格式，至少包含 x, y
    返回：添加了 kappa 的轨迹列表（每个元素为 [x, y, heading, v, a, kappa]）
    """
    traj = np.array(trajectory)
    if len(traj) < 3:
        # 点数不足时曲率置零
        kappa = np.zeros(len(traj))
    else:
        x = traj[:, 0]
        y = traj[:, 1]
        dx = np.gradient(x)
        dy = np.gradient(y)
        ddx = np.gradient(dx)
        ddy = np.gradient(dy)
        # 曲率公式: (x' y'' - y' x'') / (x'^2 + y'^2)^(3/2)
        denominator = (dx ** 2 + dy ** 2) ** 1.5
        kappa = np.zeros_like(denominator)
        mask = denominator > 1e-6
        kappa[mask] = (dx[mask] * ddy[mask] - dy[mask] * ddx[mask]) / denominator[mask]

    # 将曲率添加到轨迹中
    result = []
    for i, pt in enumerate(trajectory):
        # 假设 pt 为 [x, y, heading, v, a] 形式
        result.append(list(pt) + [float(kappa[i])])
    return result


def quintic_time_param_traj(smooth_path, start_v, start_a, end_v, end_a, T, dt=0.1):
    """
    对平滑路径进行五次多项式时间参数化，生成轨迹点（包含起点）。

    参数:
        smooth_path: (N,3) 数组，每行为 [x, y, heading]，heading 为路径切线方向角（弧度）
        start_v, start_a: 起点的速度 (m/s) 和加速度 (m/s²)（沿路径切向）
        end_v, end_a: 终点的速度 (m/s) 和加速度 (m/s²)
        T: 总时间 (s)，应为 dt 的整数倍
        dt: 采样间隔 (s)，默认 0.1

    返回:
        trajectory: list of [x, y, heading, v, a, kappa]，时间从 0 到 T，步长 dt
    """
    path = np.asarray(smooth_path, dtype=float)
    if path.ndim != 2 or path.shape[1] < 3:
        raise ValueError(f"smooth_path must have shape (N, 3), got {path.shape}")
    if len(path) < 2:
        return []  # 路径点数不足，返回空

    # 检查时间倍数关系
    steps_f = T / dt
    steps = int(round(steps_f))
    if abs(steps_f - steps) > 1e-6:
        raise ValueError(f"T must be an integer multiple of dt={dt}, got T={T}")

    # 计算路径累积弧长
    cum_s = calculate_path_cumulative_distance(path)
    total_s = float(cum_s[-1])

    # 如果路径总长几乎为零，则所有点重复终点（无运动）
    if total_s < 1e-6:
        x, y, heading = float(path[-1, 0]), float(path[-1, 1]), float(path[-1, 2])
        # 生成从 0 到 T 的每个时间点（速度加速度设为 0）
        t_points = np.linspace(0.0, float(T), steps + 1)  # 包含起点
        trajectory = []
        for t in t_points:
            # 速度加速度为零
            v = 0.0
            a = 0.0
            trajectory.append([x, y, heading, v, a])
        # 计算曲率
        trajectory = calculate_trajectory_kappa(trajectory)
        return trajectory

    # 解五次多项式（弧长 s 作为时间的函数）
    coeffs = solve_quintic_coeffs(
        x0=0.0,  # s(0) = 0
        v0=float(start_v),
        a0=float(start_a),
        xf=total_s,  # s(T) = total_s
        vf=float(end_v),
        af=float(end_a),
        T=float(T)
    )

    # 生成时间序列（包含起点和终点）
    t_points = np.linspace(0.0, float(T), steps + 1)  # 步长 dt，共 steps+1 个点
    s_profile, v_profile, a_profile = eval_quintic(coeffs, t_points)

    # 数值保护：确保 s 不超出 [0, total_s]
    s_profile = np.clip(s_profile, 0.0, total_s)

    # 插值得到每个时间点的位置和航向
    trajectory = []
    for s, v, a in zip(s_profile, v_profile, a_profile):
        x, y, heading = interp_pose_on_path(path, cum_s, s)
        # 注意：v 可能为负（若允许倒车），但此处假设非负；若需支持倒车可去掉 max
        v = max(0.0, float(v))  # 强制非负（根据同学代码的习惯）
        a = float(a)
        trajectory.append([x, y, heading, v, a])

    # 计算曲率并返回最终轨迹
    trajectory = calculate_trajectory_kappa(trajectory)
    return trajectory


# ===================================== 优化的方法 ==========================================
def _build_velocity_constraints(N, dt, L, v_start, a_start, v_end, a_end,
                                v_min, v_max, a_min, a_max, max_jerk):
    """QP/LP 共用同一组名义纵向约束，不能在回退时悄悄放宽。"""
    n_vars = N
    D2 = np.diff(np.eye(N), n=2, axis=0) if max_jerk is not None else None
    start_acc_row = np.zeros(N)
    start_acc_row[:2] = [-1.0 / dt, 1.0 / dt]
    # ---------- 约束 ----------
    # 我们将所有约束组装成 A x = b 或 A x <= b

    # 先收集所有等式约束
    A_eq_list = []
    b_eq_list = []

    # 1. 起点速度
    eq1 = np.zeros(n_vars)
    eq1[0] = 1.0
    A_eq_list.append(eq1)
    b_eq_list.append(v_start)

    # 2. 终点速度
    eq3 = np.zeros(n_vars)
    eq3[-1] = 1.0
    A_eq_list.append(eq3)
    b_eq_list.append(v_end)

    # 3. 终点加速度：a_{N-2} = (v_{N-1} - v_{N-2})/dt = a_end  => v_{N-1} - v_{N-2} = a_end * dt
    eq4 = np.zeros(n_vars)
    eq4[-2] = -1.0
    eq4[-1] = 1.0
    A_eq_list.append(eq4)
    b_eq_list.append(a_end * dt)

    # 4. 总路程约束：s_{N-1} = L
    # s_i = sum_{j=0}^{i-1} v_j * dt,  s_0 = 0
    # 构造累积和矩阵
    M = np.tril(np.ones((N, N)), k=-1)  # 下三角，主对角线为0，因为 s_i 不包括当前 v_i
    # s = M @ v * dt
    # 终点约束： (M[N-1, :] @ v) * dt = L
    eq5 = M[-1, :] * dt
    A_eq_list.append(eq5)
    b_eq_list.append(L)

    # 合并等式约束
    if A_eq_list:
        A_eq = np.vstack(A_eq_list)
        b_eq = np.array(b_eq_list)
    else:
        A_eq = np.zeros((0, n_vars))
        b_eq = np.array([])

    # 不等式约束：速度上下限
    A_ineq_list = []
    b_ineq_list = []

    # 速度范围
    for i in range(n_vars):
        # v_i <= v_max
        row_ub = np.zeros(n_vars)
        row_ub[i] = 1.0
        A_ineq_list.append(row_ub)
        b_ineq_list.append(v_max)
        # -v_i <= -v_min
        row_lb = np.zeros(n_vars)
        row_lb[i] = -1.0
        A_ineq_list.append(row_lb)
        b_ineq_list.append(-v_min)

    # 加速度范围
    # a_i = (v_{i+1} - v_i)/dt,  i = 0 ... N-2
    for i in range(N - 1):
        # a_i <= a_max
        row_ub = np.zeros(n_vars)
        row_ub[i] = -1.0 / dt
        row_ub[i + 1] = 1.0 / dt
        A_ineq_list.append(row_ub)
        b_ineq_list.append(a_max)
        # -a_i <= -a_min  => -(v_{i+1} - v_i)/dt <= -a_min
        row_lb = np.zeros(n_vars)
        row_lb[i] = 1.0 / dt
        row_lb[i + 1] = -1.0 / dt
        A_ineq_list.append(row_lb)
        b_ineq_list.append(-a_min)

    if max_jerk is not None:
        # 内部 jerk=(v[i+2]-2*v[i+1]+v[i])/dt^2，配置值作为硬约束。
        for row in D2 / dt ** 2:
            A_ineq_list.extend([row, -row])
            b_ineq_list.extend([max_jerk, max_jerk])
        # 首区间加速度可调整，但相对当前实测加速度不能瞬间跳变。
        splice_row = start_acc_row / dt
        A_ineq_list.extend([splice_row, -splice_row])
        b_ineq_list.extend([max_jerk + a_start / dt, max_jerk - a_start / dt])

    # 合并不等式约束
    A_ineq = np.vstack(A_ineq_list)
    b_ineq = np.array(b_ineq_list)

    return A_eq, b_eq, A_ineq, b_ineq


def optimize_velocity_profile(L, T, dt, v_start, a_start, v_end, a_end, v_min=0.0, v_max=7.0, a_min=-3.0, a_max=5.0,
                              w_jerk=1.0, w_acc=0.0, w_start_acc=1.0, solver_tolerance=1e-2, max_jerk=None, solver_diagnostics=None):
    """
    通过二次规划优化速度曲线，满足运动学约束和边界条件。

    参数：
        L: 路径总长度 (m)
        T: 总时间 (s)
        dt: 采样间隔 (s)
        v_start, a_start: 起点速度、加速度
        v_end, a_end: 终点速度、加速度
        v_min, v_max: 速度上下限
        a_min, a_max: 加速度上下限
        w_jerk, w_acc: 目标函数中 jerk 和加速度项的权重
        w_start_acc: 首个离散区间加速度跟踪当前加速度的软约束权重
        max_jerk: 调用方配置的 jerk 硬上限；正无穷或 None 表示不限制
        solver_diagnostics: 可选输出字典，保留原始状态、迭代、残差和接受质量

    返回：
        t: 时间数组 (N,)
        s: 位置数组 (N,)
        v: 速度数组 (N,)
        a: 加速度数组 (N-1,)
    """
    max_jerk = _validate_dynamic_limits(v_min, v_max, a_min, a_max, max_jerk)
    if (not np.all(np.isfinite([L, T, dt, v_start, a_start, v_end, a_end, solver_tolerance]))
            or L < 0.0 or T <= 0.0 or dt <= 0.0 or solver_tolerance <= 0.0):
        raise VelocityProfileError("invalid_input", "Invalid distance, horizon, state or solver tolerance")
    N = int(round(T / dt)) + 1  # 包含起点和终点的点数
    if N < 2:
        raise VelocityProfileError("invalid_input", "Planning horizon must contain at least one interval")
    T_actual = (N - 1) * dt
    # if abs(T_actual - T) > 1e-6:
    #     raise ValueError(f"T must be an integer multiple of dt, got T={T}, dt={dt}")

    # 变量：v[0] ... v[N-1] 共 N 个
    n_vars = N

    # ---------- 构建目标函数 ----------
    # 我们希望最小化 jerk 平方积分（离散形式）
    # jerk_i = (a_{i+1} - a_i)/dt, 其中 a_i = (v_{i+1} - v_i)/dt
    # 所以 jerk_i = (v_{i+2} - 2v_{i+1} + v_i) / dt^2
    # 目标项：sum (jerk_i^2) * dt  近似积分
    # 构建矩阵 Q_jerk，使得 v^T Q_jerk v = sum jerk_i^2 * dt

    # 采用有限差分矩阵构造二阶差分
    D2 = np.diff(np.eye(N), n=2, axis=0)  # 每个 jerk 涉及三个 v；单区间时为空
    # jerk = D2 @ v / dt^2
    # sum jerk_i^2 = v^T (D2.T @ D2) v / dt^4
    # 积分乘以 dt: 目标项 = v^T (D2.T @ D2 / dt^3) v
    Q_jerk = (D2.T @ D2) / (dt ** 3)

    # 如果希望同时最小化加速度，可以加入加速度项
    # a_i = (v_{i+1} - v_i)/dt, 构造差分矩阵 D1 (N-1, N)
    D1 = diags([-1, 1], [0, 1], shape=(N - 1, N)).toarray()
    # sum a_i^2 * dt = v^T (D1.T @ D1 / dt) v
    Q_acc = (D1.T @ D1) / dt

    Q = w_jerk * Q_jerk + w_acc * Q_acc
    # 滚动规划每帧只执行轨迹第一个区间。如果把起始加速度写成硬等式，
    # 上一帧加速度会在后续每帧被重复执行，无法向新的终端速度收敛。
    # 这里保留加速度连续性偏好，但允许优化器在当前帧调整首区间加速度。
    start_acc_row = np.zeros(n_vars)
    start_acc_row[0] = -1.0 / dt
    start_acc_row[1] = 1.0 / dt
    Q = Q + float(w_start_acc) * np.outer(start_acc_row, start_acc_row)
    # OSQP 需要稀疏矩阵形式
    Q = csc_matrix(Q)

    # OSQP 目标为 0.5*x'Px + q'x，线性项与上面的起始加速度软约束配对。
    q = -float(w_start_acc) * float(a_start) * start_acc_row

    A_eq, b_eq, A_ineq, b_ineq = _build_velocity_constraints(
        N, dt, L, v_start, a_start, v_end, a_end,
        v_min, v_max, a_min, a_max, max_jerk)

    # 组装成 OSQP 所需格式
    P = Q
    q = q
    A = csc_matrix(np.vstack([A_eq, A_ineq]))
    l = np.concatenate([b_eq, -np.inf * np.ones(len(b_ineq))])
    u = np.concatenate([b_eq, b_ineq])

    # 创建求解器实例
    prob = osqp.OSQP()
    # 导数硬约束加入后，自动缩放会让已知可行的小型问题在残差平衡上停滞。
    # 使用原始物理单位求解；不改变任何约束、容差或迭代上限。
    numeric_settings = {"scaling": 0} if max_jerk is not None else {}
    prob.setup(P, q, A, l, u, verbose=False, polish=False, eps_abs=solver_tolerance,
               eps_rel=solver_tolerance, check_termination=100, **numeric_settings)

    # 求解
    res = prob.solve()

    status = str(res.info.status)
    diagnostics = {} if solver_diagnostics is None else solver_diagnostics
    diagnostics.clear()
    def finite_residual(name, legacy_name):
        value = getattr(res.info, name, getattr(res.info, legacy_name, None))
        return float(value) if value is not None and np.isfinite(value) else None
    diagnostics.update(
        solver_status=status, solver_iterations=int(getattr(res.info, "iter", 0)),
        solver_primal_residual=finite_residual("prim_res", "pri_res"),
        solver_dual_residual=finite_residual("dual_res", "dua_res"),
        solution_quality="rejected", accepted_nonconverged=False,
        constraint_check_passed=False, dynamic_numeric_tolerance=DYNAMIC_NUMERIC_TOLERANCE,
    )
    usable_statuses = {"solved", "solved inaccurate", "maximum iterations reached", "run time limit reached"}
    if status not in usable_statuses:
        kind = ("candidate_infeasible" if status == "primal infeasible" else
                "solver_not_converged" if status in (
                    "maximum iterations reached", "run time limit reached", "solved inaccurate",
                    "primal infeasible inaccurate", "unsolved") else "solver_failure")
        raise VelocityProfileError(
            kind, "OSQP failed to solve: "
            f"{status}; L={L:.3f}, T={T:.3f}, "
            f"v_start={v_start:.3f}, v_end={v_end:.3f}", solver_status=status,
        )

    # 有限的完整候选通过相同验收即可交付；未收敛不等于约束违规。
    candidate_kind = "constraint_violation" if status == "solved" else "solver_not_converged"
    raw_velocity = getattr(res, "x", None)
    try:
        v_opt = np.asarray(raw_velocity, dtype=float)
    except (TypeError, ValueError) as exc:
        raise VelocityProfileError(candidate_kind, "OSQP returned no complete finite candidate", status) from exc
    if v_opt.shape != (N,) or not np.all(np.isfinite(v_opt)):
        raise VelocityProfileError(candidate_kind, "OSQP returned no complete finite candidate", status)
    try:
        validate_velocity_profile(v_opt, v_start, a_start, dt,
                                  v_min, v_max, a_min, a_max, max_jerk)
        if np.max(np.abs(A_eq @ v_opt - b_eq)) > PROFILE_EQUALITY_TOLERANCE:
            raise VelocityProfileError("constraint_violation", "Endpoint/distance residual exceeded tolerance")
    except VelocityProfileError as exc:
        exc.solver_status = status
        raise
    diagnostics.update(
        solution_quality="optimized" if status == "solved" else "feasible_not_optimal",
        accepted_nonconverged=status != "solved", constraint_check_passed=True,
        initial_speed_error_mps=float(v_opt[0] - v_start),
        terminal_speed_error_mps=float(v_opt[-1] - v_end),
        terminal_acceleration_error_mps2=float((v_opt[-1] - v_opt[-2]) / dt - a_end),
        distance_error_m=float(dt * np.sum(v_opt[:-1]) - L),
        max_speed_violation_mps=float(max(0.0, v_min - np.min(v_opt), np.max(v_opt) - v_max)),
        max_acceleration_violation_mps2=float(max(
            0.0, a_min - np.min(np.diff(v_opt) / dt), np.max(np.diff(v_opt) / dt) - a_max)),
    )

    # 计算 s 和 a
    s_opt = np.zeros(N)
    for i in range(1, N):
        s_opt[i] = s_opt[i - 1] + v_opt[i - 1] * dt

    a_opt = np.diff(v_opt) / dt  # 加速度，长度 N-1

    # 生成时间轴
    t = np.arange(N) * dt

    return t, s_opt, v_opt, a_opt


def solve_feasible_velocity_profile(L, T, dt, v_start, a_start, v_end, a_end,
                                    *, v_min, v_max, a_min, a_max, max_jerk=None,
                                    time_limit_s=DEFAULT_HIGHS_TIME_LIMIT_S,
                                    solver_diagnostics=None):
    """HiGHS 仅求同一约束下的可行速度序列，不改变路径、目标或限值。"""
    max_jerk = _validate_dynamic_limits(v_min, v_max, a_min, a_max, max_jerk)
    if (not np.all(np.isfinite([L, T, dt, v_start, a_start, v_end, a_end, time_limit_s]))
            or L < 0.0 or T <= 0.0 or dt <= 0.0 or time_limit_s <= 0.0):
        raise VelocityProfileError("invalid_input", "Invalid HiGHS input or time limit")
    N = int(round(T / dt)) + 1
    if N < 2:
        raise VelocityProfileError("invalid_input", "Planning horizon must contain at least one interval")
    A_eq, b_eq, A_ineq, b_ineq = _build_velocity_constraints(
        N, dt, L, v_start, a_start, v_end, a_end,
        v_min, v_max, a_min, a_max, max_jerk)
    diagnostics = {} if solver_diagnostics is None else solver_diagnostics
    diagnostics.clear()
    diagnostics.update(solver_backend="highs", solution_quality="rejected",
                       accepted_nonconverged=False, constraint_check_passed=False,
                       dynamic_numeric_tolerance=DYNAMIC_NUMERIC_TOLERANCE,
                       solver_time_limit_s=float(time_limit_s))
    started = time.perf_counter()
    result = scipy_optimize.linprog(np.zeros(N), A_ub=A_ineq, b_ub=b_ineq, A_eq=A_eq, b_eq=b_eq,
                     bounds=[(None, None)] * N, method="highs",
                     options={"time_limit": float(time_limit_s)})
    code = int(result.status)
    status = {0: "optimal", 1: "time_or_iteration_limit", 2: "infeasible",
              3: "unbounded", 4: "solver_error"}.get(code, "unknown")
    diagnostics.update(solver_status=status, solver_status_code=code,
                       solver_message=str(result.message), solver_iterations=int(getattr(result, "nit", 0)),
                       solver_wall_ms=(time.perf_counter() - started) * 1000.0)
    if code not in (0, 1):
        kind = "candidate_infeasible" if code == 2 else "solver_failure"
        raise VelocityProfileError(kind, f"HiGHS failed: {status}; {result.message}", status)
    kind = "constraint_violation" if code == 0 else "solver_not_converged"
    try:
        v = np.asarray(getattr(result, "x", None), dtype=float)
    except (ValueError, TypeError) as exc:
        raise VelocityProfileError(kind, "HiGHS returned no complete finite candidate", status) from exc
    if v.shape != (N,) or not np.all(np.isfinite(v)):
        raise VelocityProfileError(kind, "HiGHS returned no complete finite candidate", status)
    try:
        validate_velocity_profile(v, v_start, a_start, dt, v_min, v_max, a_min, a_max, max_jerk)
        if np.max(np.abs(A_eq @ v - b_eq)) > PROFILE_EQUALITY_TOLERANCE:
            raise VelocityProfileError("constraint_violation", "Endpoint/distance residual exceeded tolerance")
    except VelocityProfileError as exc:
        exc.solver_status = status
        raise
    acc = np.diff(v) / dt
    diagnostics.update(solution_quality="feasible" if code == 0 else "feasible_not_optimal",
                       accepted_nonconverged=code == 1, constraint_check_passed=True,
                       initial_speed_error_mps=float(v[0] - v_start),
                       terminal_speed_error_mps=float(v[-1] - v_end),
                       terminal_acceleration_error_mps2=float(acc[-1] - a_end),
                       distance_error_m=float(dt * np.sum(v[:-1]) - L),
                       max_speed_violation_mps=float(max(0., v_min - v.min(), v.max() - v_max)),
                       max_acceleration_violation_mps2=float(max(0., a_min - acc.min(), acc.max() - a_max)))
    return np.arange(N) * dt, np.r_[0., np.cumsum(v[:-1]) * dt], v, acc


def optimize_time_param_traj(smooth_path, T, v_start, a_start, v_end, a_end, dt=0.1, v_min=0.0, v_max=7.0,
                                          a_min=-3.0, a_max=5.0, solver_tolerance=1e-2, max_jerk=None, solver_diagnostics=None,
                                          solver_backend="osqp", highs_time_limit_s=DEFAULT_HIGHS_TIME_LIMIT_S):
    """
    综合函数：先优化速度，再插值得到轨迹点（包含起点）。
    """
    path = np.asarray(smooth_path)
    cum_s = calculate_path_cumulative_distance(path)
    L = cum_s[-1]

    if solver_backend == "osqp":
        t, s_opt, v_opt, a_opt = optimize_velocity_profile(
            L, T, dt, v_start, a_start, v_end, a_end,
            v_min, v_max, a_min, a_max, solver_tolerance=solver_tolerance,
            max_jerk=max_jerk, solver_diagnostics=solver_diagnostics)
    elif solver_backend == "highs":
        t, s_opt, v_opt, a_opt = solve_feasible_velocity_profile(
            L, T, dt, v_start, a_start, v_end, a_end,
            v_min=v_min, v_max=v_max, a_min=a_min, a_max=a_max,
            max_jerk=max_jerk, time_limit_s=highs_time_limit_s,
            solver_diagnostics=solver_diagnostics)
    else:
        raise VelocityProfileError("invalid_input", f"Unknown velocity solver: {solver_backend}")

    # 插值得到位置和航向
    trajectory = []
    for i in range(len(t)):
        s = s_opt[i]
        x, y, heading = interp_pose_on_path(path, cum_s, s)
        v = v_opt[i]
        # 加速度需要对应时刻：a_opt 长度 N-1，最后一个点没有加速度（可用 a_opt[-1] 近似）
        a = a_opt[i] if i < len(a_opt) else a_opt[-1]
        trajectory.append([x, y, heading, v, a])

    # 计算曲率
    trajectory = calculate_trajectory_kappa(trajectory)
    return np.array(trajectory)
