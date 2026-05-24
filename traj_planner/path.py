import numpy as np
from scipy.spatial import KDTree


CURVATURE_WARNING_PRINT_LIMIT = 5
_curvature_warning_count = 0


def _log_curvature_warning() -> None:
    """限制重复曲率告警的输出频率，避免刷屏掩盖主流程日志。"""

    global _curvature_warning_count
    _curvature_warning_count += 1
    if _curvature_warning_count <= CURVATURE_WARNING_PRINT_LIMIT:
        print(
            f"⚠: 无法满足曲率约束！使用最小scale控制贝塞尔曲线（第{_curvature_warning_count}次）",
            flush=True,
        )
    elif _curvature_warning_count == CURVATURE_WARNING_PRINT_LIMIT + 1:
        print("⚠: 后续相同曲率告警已省略，请结合 step 日志观察仿真进度", flush=True)


def compute_path_curvature(path_xy):
    """
    计算路径点 (N,2) 的曲率（三点法），返回曲率数组
    """
    if len(path_xy) < 3:
        return np.zeros(len(path_xy))
    x = path_xy[:, 0]
    y = path_xy[:, 1]
    dx = np.gradient(x)
    dy = np.gradient(y)
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    denominator = (dx ** 2 + dy ** 2) ** 1.5
    kappa = np.zeros_like(denominator)
    mask = denominator > 1e-6
    kappa[mask] = (dx[mask] * ddy[mask] - dy[mask] * ddx[mask]) / denominator[mask]
    return kappa


def compute_heading_from_path(path_xy, start_theta, end_theta):
    """
    根据路径点计算航向角（弧度），并强制首尾角度为给定值。
    path_xy: (N,2) 路径点
    start_theta: 起点航向角
    end_theta: 终点航向角
    返回: (N,3) [x, y, heading]
    """
    n = len(path_xy)
    if n == 0:
        return np.empty((0, 3))
    if n == 1:
        return np.array([[path_xy[0, 0], path_xy[0, 1], start_theta]])
    # 使用中心差分计算 heading（边界使用前向/后向）
    heading = np.zeros(n)
    for i in range(n):
        if i == 0:
            dx = path_xy[1, 0] - path_xy[0, 0]
            dy = path_xy[1, 1] - path_xy[0, 1]
        elif i == n - 1:
            dx = path_xy[-1, 0] - path_xy[-2, 0]
            dy = path_xy[-1, 1] - path_xy[-2, 1]
        else:
            dx = path_xy[i + 1, 0] - path_xy[i - 1, 0]
            dy = path_xy[i + 1, 1] - path_xy[i - 1, 1]
        heading[i] = np.arctan2(dy, dx)
    # 强制端点角度
    heading[0] = start_theta
    heading[-1] = end_theta
    return np.column_stack([path_xy, heading])


def compute_reference_properties(target_line):
    """
    计算参考线的基础属性
    返回：
        s_points: 每个参考点的累积弧长 (n,)
        segments: 字典，包含每个线段的信息
            - start_idx: 线段起点索引
            - end_idx:   线段终点索引
            - p1, p2:    线段端点坐标
            - vec:       线段向量
            - length:    线段长度
            - unit:      单位方向向量
            - normal:    单位法向量（指向左侧）
    """
    n = len(target_line)
    # 计算累积弧长
    diff = np.diff(target_line, axis=0)
    ds = np.sqrt(np.sum(diff ** 2, axis=1))
    s_points = np.concatenate([[0], np.cumsum(ds)])

    segments = []
    for i in range(n - 1):
        p1 = target_line[i]
        p2 = target_line[i + 1]
        vec = p2 - p1
        length = ds[i]
        if length > 1e-6:
            unit = vec / length
        else:
            unit = np.array([1.0, 0.0])  # 退化为点，方向任意
        # 左手法向量：逆时针旋转90度
        normal = np.array([-unit[1], unit[0]])
        segments.append({
            'start_idx': i,
            'end_idx': i + 1,
            'p1': p1,
            'p2': p2,
            'vec': vec,
            'length': length,
            'unit': unit,
            'normal': normal,
            's_start': s_points[i],
            's_end': s_points[i + 1]
        })
    return s_points, segments


def project_to_reference(pt, target_line, segments, kdtree=None):
    """
    将单个点精确投影到参考线上
    返回：
        s: 投影点的弧长
        l: 有符号横向偏移（左正右负）
        proj_pt: 投影点坐标
        tangent: 投影点处的单位切线向量
        normal:  投影点处的单位法向量
    """
    if kdtree is None:
        kdtree = KDTree(target_line)

    # 1. 找到最近点索引
    dist, idx = kdtree.query(pt)

    # 2. 候选线段：idx-1 和 idx 两条（注意边界）
    candidate_idxs = []
    if idx > 0:
        candidate_idxs.append(idx - 1)
    if idx < len(target_line) - 1:
        candidate_idxs.append(idx)

    best_dist2 = np.inf
    best_s = None
    best_l = None
    best_proj = None
    best_tangent = None
    best_normal = None

    for seg_idx in candidate_idxs:
        seg = segments[seg_idx]
        # 投影参数 t (0~1)
        if seg['length'] < 1e-6:
            # 线段退化为点，投影点就是端点
            proj = seg['p1']
            t = 0.0
        else:
            t = np.dot(pt - seg['p1'], seg['unit']) / seg['length']
            t = np.clip(t, 0, 1)  # 限制在线段内
            proj = seg['p1'] + t * seg['vec']

        d2 = np.sum((pt - proj) ** 2)
        if d2 < best_dist2:
            best_dist2 = d2
            # 计算 s: 线段起点弧长 + t * 线段长度
            best_s = seg['s_start'] + t * seg['length']
            # 计算 l: 有符号横向偏移（使用该线段的法向量）
            offset_vec = pt - proj
            best_l = np.dot(offset_vec, seg['normal'])
            best_proj = proj
            best_tangent = seg['unit']  # 用线段方向作为切线
            best_normal = seg['normal']

    return best_s, best_l, best_proj, best_tangent, best_normal


def cartesian_to_frenet_exact(points, target_line, segments):
    """
    将多个点精确转换到Frenet坐标系
    返回：
        s_arr: 弧长数组
        l_arr: 横向偏移数组
        tangents: 每个点处的切线向量数组
        normals:  每个点处的法线向量数组
    """
    kdtree = KDTree(target_line)
    s_arr = np.zeros(len(points))
    l_arr = np.zeros(len(points))
    tangents = np.zeros((len(points), 2))
    normals = np.zeros((len(points), 2))

    for i, pt in enumerate(points):
        s, l, proj, tangent, normal = project_to_reference(pt, target_line, segments, kdtree)
        s_arr[i] = s
        l_arr[i] = l
        tangents[i] = tangent
        normals[i] = normal

    return s_arr, l_arr, tangents, normals


def absolute_to_frenet_theta(abs_theta, tangent):
    """绝对航向角 -> 相对于参考线切线的角度"""
    ref_theta = np.arctan2(tangent[1], tangent[0])
    rel_theta = abs_theta - ref_theta
    # 归一化到 [-pi, pi]
    rel_theta = np.arctan2(np.sin(rel_theta), np.cos(rel_theta))
    return rel_theta


def generate_bezier_control_points_frenet(start_s, start_l, start_rel_theta, end_s, end_l, end_rel_theta, scale1=0.3,
                                          scale2=None):
    P0 = np.array([start_s, start_l])
    P3 = np.array([end_s, end_l])
    dir_start = np.array([np.cos(start_rel_theta), np.sin(start_rel_theta)])
    dir_end = np.array([np.cos(end_rel_theta), np.sin(end_rel_theta)])
    delta_s = end_s - start_s
    if scale2 is None:
        dist = delta_s * scale1
        P1 = P0 + dir_start * dist
        P2 = P3 - dir_end * dist
    else:
        P1 = P0 + dir_start * delta_s * scale1
        P2 = P3 - dir_end * delta_s * scale2
    # 可选：限制P1/P2的s不超出区间
    P1[0] = np.clip(P1[0], start_s, end_s)
    P2[0] = np.clip(P2[0], start_s, end_s)
    return np.array([P0, P1, P2, P3])


def evaluate_bezier_frenet(control_points, num_points=50):
    t = np.linspace(0, 1, num_points)
    curve = np.zeros((num_points, 2))
    for i, ti in enumerate(t):
        B0 = (1 - ti) ** 3
        B1 = 3 * (1 - ti) ** 2 * ti
        B2 = 3 * (1 - ti) * ti ** 2
        B3 = ti ** 3
        curve[i] = (B0 * control_points[0] + B1 * control_points[1] +
                    B2 * control_points[2] + B3 * control_points[3])
    return curve


def frenet_to_cartesian_exact(curve_sl, segments, s_points):
    """
    将Frenet曲线点 (s, l) 转换回笛卡尔坐标
    """
    xy = np.zeros((len(curve_sl), 2))
    for i, (s, l) in enumerate(curve_sl):
        # 根据s找到所在线段
        # 使用二分查找定位 s 在 s_points 中的位置
        idx = np.searchsorted(s_points, s) - 1
        idx = np.clip(idx, 0, len(segments) - 1)
        seg = segments[idx]
        # 在线段上插值得到参考点
        if seg['length'] > 1e-6:
            t = (s - seg['s_start']) / seg['length']
            t = np.clip(t, 0, 1)
            ref_pt = seg['p1'] + t * seg['vec']
        else:
            ref_pt = seg['p1']
        # 加上横向偏移（使用该线段的法向量）
        xy[i] = ref_pt + l * seg['normal']
    return xy


def bezier_curve(target_line, start_point, end_point, start_theta, end_theta=None):
    # 1. 参考线预处理
    s_points, segments = compute_reference_properties(target_line)
    # 2. 精确转换起终点到Frenet
    points = np.vstack([start_point, end_point])
    s, l, tangents, normals = cartesian_to_frenet_exact(points, target_line, segments)
    start_s, end_s = s[0], s[1]
    start_l, end_l = l[0], l[1]
    start_tangent, end_tangent = tangents[0], tangents[1]
    if end_theta is None:
        # 使用参考线上投影点的切线方向作为终点的绝对朝向
        end_theta = np.arctan2(end_tangent[1], end_tangent[0])
    # 3. 转换朝向到Frenet相对角度
    start_rel_theta = absolute_to_frenet_theta(start_theta, start_tangent)
    end_rel_theta = absolute_to_frenet_theta(end_theta, end_tangent)
    # 4. 生成Frenet贝塞尔控制点
    ctrl_pts = generate_bezier_control_points_frenet(
        start_s, start_l, start_rel_theta,
        end_s, end_l, end_rel_theta,
        scale=0.3
    )
    # 5. 采样Frenet曲线
    curve_sl = evaluate_bezier_frenet(ctrl_pts, num_points=50)
    # 6. 转换回笛卡尔
    path_xy = frenet_to_cartesian_exact(curve_sl, segments, s_points)
    path_with_heading = compute_heading_from_path(path_xy, start_theta, end_theta)
    return path_with_heading


def bezier_curve_kappa(target_line, start_point, end_point, start_theta, end_theta=None, max_curvature=0.3,
                       scale_list=(0.35, 0.3, 0.25, 0.2, 0.15, 0.1), num_path_points=50):
    """
    尝试不同的 scale，直到生成的路径最大曲率绝对值 < max_curvature。
    返回满足条件的 scale 和对应的笛卡尔路径 (num_path_points, 3)。
    如果无法满足，返回 None, None。
    """
    # 预计算参考线属性（假设已有 compute_reference_properties, cartesian_to_frenet_exact 等函数）
    s_points, segments = compute_reference_properties(target_line)
    points = np.vstack([start_point, end_point])
    s, l, tangents, normals = cartesian_to_frenet_exact(points, target_line, segments)
    start_s, end_s = s[0], s[1]
    start_l, end_l = l[0], l[1]
    start_tangent, end_tangent = tangents[0], tangents[1]
    # 处理终点朝向 None 的情况（已在 main 中处理，这里假设已给定）
    if end_theta is None:
        end_theta = np.arctan2(end_tangent[1], end_tangent[0])
    start_rel_theta = absolute_to_frenet_theta(start_theta, start_tangent)
    end_rel_theta = absolute_to_frenet_theta(end_theta, end_tangent)

    for scale in scale_list:
        # 生成 Frenet 贝塞尔控制点
        ctrl_pts = generate_bezier_control_points_frenet(start_s, start_l, start_rel_theta, end_s, end_l, end_rel_theta,
                                                         scale1=scale)
        # 采样 Frenet 曲线
        curve_sl = evaluate_bezier_frenet(ctrl_pts, num_points=num_path_points)
        # 转换到笛卡尔
        path_xy = frenet_to_cartesian_exact(curve_sl, segments, s_points)
        # 计算最大曲率绝对值
        kappa = compute_path_curvature(path_xy)
        max_kappa = np.max(np.abs(kappa))
        # print(f"scale={scale:.3f}, max curvature={max_kappa:.4f}")
        if max_kappa < max_curvature:
            path_with_heading = compute_heading_from_path(path_xy, start_theta, end_theta)
            return path_with_heading
    _log_curvature_warning()
    ctrl_pts = generate_bezier_control_points_frenet(start_s, start_l, start_rel_theta, end_s, end_l, end_rel_theta,
                                                     scale1=0.33, scale2=0.33)
    curve_sl = evaluate_bezier_frenet(ctrl_pts, num_points=num_path_points)
    path_xy = frenet_to_cartesian_exact(curve_sl, segments, s_points)
    return compute_heading_from_path(path_xy, start_theta, end_theta)
