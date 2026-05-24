import numpy as np

MAX_PROTOCOL_SPEED_MPS = 25.0 / 3.6


def build_path_curvature_profile(smooth_path, cum_s, min_seg_len=1e-3):
    path = np.asarray(smooth_path, dtype=float)
    cum_s = np.asarray(cum_s, dtype=float)

    n = len(path)
    if n < 3:
        return np.zeros(n)

    diffs = np.diff(path[:, :2], axis=0)
    seg_len = np.linalg.norm(diffs, axis=1)

    keep = np.ones(n, dtype=bool)
    keep[1:] = seg_len > min_seg_len

    path_f = path[keep]
    cum_s_f = cum_s[keep]

    if len(path_f) < 3:
        return np.zeros(n)

    ds = np.diff(cum_s_f)
    seg_vec = np.diff(path_f[:, :2], axis=0)
    seg_heading = np.arctan2(seg_vec[:, 1], seg_vec[:, 0])
    dtheta = np.array([
        normalize_angle(seg_heading[i + 1] - seg_heading[i])
        for i in range(len(seg_heading) - 1)
    ])
    ds_mid = 0.5 * (ds[:-1] + ds[1:])
    kappa_mid = np.abs(dtheta) / np.maximum(ds_mid, 1e-3)

    kappa_f = np.zeros(len(path_f))
    if len(kappa_mid) > 0:
        kappa_f[0] = kappa_mid[0]
        kappa_f[-1] = kappa_mid[-1]
        kappa_f[1:-1] = kappa_mid

    kappa_abs = np.zeros(n)
    kappa_abs[keep] = kappa_f
    return kappa_abs



def project_point_to_path(point, path):
    """
    将点投影到路径上
    返回:
        (最近点索引, 沿路径距离s, 横向距离d - 带符号，左正右负)
    """
    if len(path) < 2:
        print("⚠️ 参考线点数不足")
        return -1, -1.0, 0.0

    # 计算累积距离
    diffs = np.diff(path[:, :2], axis=0)
    seg_lengths = np.sqrt(np.sum(diffs ** 2, axis=1))
    cum_dist = np.zeros(len(path))
    cum_dist[1:] = np.cumsum(seg_lengths)

    # 找到最近的线段
    min_dist = float('inf')
    best_s = 0.0
    best_d = 0.0
    best_idx = 0

    for i in range(len(path) - 1):
        p1 = path[i, :2]
        p2 = path[i + 1, :2]
        seg = p2 - p1
        seg_len = seg_lengths[i]

        if seg_len < 1e-6:
            continue

        # 计算投影参数t
        t = np.dot(point[:2] - p1, seg) / (seg_len ** 2)
        t = np.clip(t, 0, 1)

        # 投影点
        proj = p1 + t * seg
        dist = np.linalg.norm(point[:2] - proj)

        if dist < min_dist:
            min_dist = dist
            best_idx = i
            best_s = cum_dist[i] + t * seg_len

            # 计算带符号的横向距离（叉积判断左右）
            cross = seg[0] * (point[1] - p1[1]) - seg[1] * (point[0] - p1[0])
            best_d = dist * np.sign(cross) if abs(cross) > 1e-6 else dist

    return best_idx, best_s, best_d


def calculate_path_cumulative_distance(path: np.ndarray) -> np.ndarray:
    """计算路径累积距离"""
    if len(path) < 2:
        return np.array([0.0])
    diffs = np.diff(path[:, :2], axis=0)
    distances = np.sqrt(np.sum(diffs ** 2, axis=1))
    cum_dist = np.zeros(len(path))
    cum_dist[1:] = np.cumsum(distances)
    return cum_dist


def get_path_point_at_s(path, s, cum_dist=None):
    """获取路径上s位置的点和航向"""
    if cum_dist is None:
        cum_dist = calculate_path_cumulative_distance(path)
    s = np.clip(s, 0, cum_dist[-1])
    idx = np.searchsorted(cum_dist, s, side='right') - 1
    idx = np.clip(idx, 0, len(path) - 2)
    seg_start_s = cum_dist[idx]
    seg_end_s = cum_dist[idx + 1]
    seg_len = seg_end_s - seg_start_s
    if seg_len > 1e-6:
        t = (s - seg_start_s) / seg_len
    else:
        t = 0.0
    p1 = path[idx, :2]
    p2 = path[idx + 1, :2]
    pos = p1 + t * (p2 - p1)
    heading = np.arctan2(p2[1] - p1[1], p2[0] - p1[0])
    return pos, heading


def normalize_angle(angle: float) -> float:
    """角度归一化到 [-π, π]"""
    return (angle + np.pi) % (2 * np.pi) - np.pi


def slice_path_from_projection(path, ego_pos):
    """
    从 ego_pos 在 path 上的投影点开始截取路径，返回“从当前位置往前”的路径。
    """
    path = np.asarray(path, dtype=float)
    ego_pos = np.asarray(ego_pos, dtype=float)

    if path.ndim != 2 or len(path) == 0:
        return path
    if len(path) == 1:
        return path.copy()

    best_idx = 0
    best_proj = path[0, :2].copy()
    min_dist = float("inf")

    for i in range(len(path) - 1):
        p1 = path[i, :2]
        p2 = path[i + 1, :2]
        seg = p2 - p1
        seg_len_sq = float(np.dot(seg, seg))
        if seg_len_sq < 1e-12:
            continue

        t = float(np.dot(ego_pos[:2] - p1, seg) / seg_len_sq)
        t = float(np.clip(t, 0.0, 1.0))
        proj = p1 + t * seg
        dist = float(np.linalg.norm(ego_pos[:2] - proj))

        if dist < min_dist:
            min_dist = dist
            best_idx = i
            best_proj = proj

    sliced = [best_proj.tolist()]
    append_start_idx = best_idx + 1
    if np.linalg.norm(path[best_idx + 1, :2] - best_proj) < 1e-6:
        append_start_idx = best_idx + 2

    for j in range(append_start_idx, len(path)):
        sliced.append(path[j, :2].tolist())

    if len(sliced) < 2:
        if best_idx >= 0 and best_idx < len(path):
            anchor = path[min(best_idx + 1, len(path) - 1), :2].tolist()
            if np.linalg.norm(np.asarray(anchor) - np.asarray(sliced[0])) > 1e-6:
                sliced.append(anchor)
        if len(sliced) < 2 and len(path) >= 2:
            fallback = path[-1, :2].tolist()
            if np.linalg.norm(np.asarray(fallback) - np.asarray(sliced[0])) > 1e-6:
                sliced.append(fallback)

    return np.asarray(sliced, dtype=float)


def align_path_direction_to_heading(path, ego_pos, ego_heading):
    """
    根据自车当前位置附近的切线方向，必要时反转参考线，使其与自车航向基本一致。
    """
    path = np.asarray(path, dtype=float)
    if path.ndim != 2 or len(path) < 2:
        return path

    idx, _, _ = project_point_to_path(np.asarray(ego_pos, dtype=float), path)
    if idx < 0:
        return path
    idx = int(np.clip(idx, 0, len(path) - 2))

    seg_vec = None
    search_order = [idx]
    for offset in range(1, len(path) - 1):
        left = idx - offset
        right = idx + offset
        if left >= 0:
            search_order.append(left)
        if right <= len(path) - 2:
            search_order.append(right)

    for seg_idx in search_order:
        vec = path[seg_idx + 1, :2] - path[seg_idx, :2]
        if np.linalg.norm(vec) > 1e-6:
            seg_vec = vec
            break

    if seg_vec is None:
        return path

    seg_heading = np.arctan2(seg_vec[1], seg_vec[0])
    if abs(normalize_angle(seg_heading - ego_heading)) > np.pi / 2:
        path = path[::-1].copy()

    return slice_path_from_projection(path, ego_pos)


def find_closest_line_to_point(refer_lines, point):
    """找到离point最近的参考线"""
    min_dist = float('inf')
    closest_line = refer_lines[0]
    for line in refer_lines:
        if len(line) == 0:
            continue
        distances = np.linalg.norm(line[:, :2] - point, axis=1)
        min_d = np.min(distances)
        if min_d < min_dist:
            min_dist = min_d
            closest_line = line
    return closest_line


def select_target_and_old_line(refer_lines, target_point):
    """
    根据target_point选择target_line和old_line
    参数:
        refer_lines: 参考线列表
        target_point: 目标点坐标
    返回:
        (target_line, old_line): 离target_point近的为target_line，远的为old_line
    """
    if len(refer_lines) < 2:
        return refer_lines[0], refer_lines[0]

    # 计算每条线到target_point的最小距离
    dist0 = np.min(np.linalg.norm(refer_lines[0][:, :2] - target_point, axis=1))
    dist1 = np.min(np.linalg.norm(refer_lines[1][:, :2] - target_point, axis=1))

    if dist0 < dist1:
        return refer_lines[0], refer_lines[1]
    else:
        return refer_lines[1], refer_lines[0]


def select_target_point_on_line(line, c_dist, start_s, current_v, tplan, desired_v, max_acc, min_acc, start_l=0.0):
    """

    :param line: 参考线
    :param c_dist: 参考线段累计长度列表
    :param start_s: 初始位置的纵向距离
    :param current_v: 当前速度
    :param tplan: 规划时长
    :param desired_v: 期望速度
    :param max_acc: 最大加速度
    :param min_acc: 最大减速度
    :param start_l: 初始位置的横向距离
    """
    if current_v > desired_v:
        # 需要减速
        if current_v + min_acc * 0.8 * tplan > desired_v:
            desired_v_eff = current_v + min_acc * 0.8 * tplan
            a_eff = min_acc * max(1 - desired_v / desired_v_eff, 0)
        else:
            desired_v_eff = desired_v
            a_eff = 0.0
    else:
        # 需要加速
        if current_v + max_acc * 0.7 * tplan > desired_v:
            desired_v_eff = desired_v
            a_eff = 0.0
        else:
            desired_v_eff = current_v + max_acc * 0.7 * tplan
            a_eff = max_acc *(1- min(desired_v_eff / desired_v, 1))

    dist_by_desired_avg = 0.5 * (current_v + desired_v_eff) * tplan
    if dist_by_desired_avg > start_l:
        dist_s = max(np.sqrt(dist_by_desired_avg ** 2 - start_l ** 2), 1.5 * start_l)
    else:
        dist_s = 1.5 * start_l

    target_s = start_s + dist_s
    if target_s > c_dist[-1]:
        print("⚠ 规划目标点距离超出参考线长度")
        remaining_s = max(float(c_dist[-1] - start_s), 0.0)
        avg_speed_feasible_end_v = max(0.0, 2.0 * remaining_s / max(float(tplan), 1e-3) - float(current_v))
        accel_feasible_end_v = np.sqrt(max(float(current_v) ** 2 + 2.0 * max(float(max_acc), 0.0) * remaining_s, 0.0))
        desired_v_eff = min(float(desired_v_eff), float(avg_speed_feasible_end_v), float(accel_feasible_end_v))
        a_eff = 0.0
    pos, _ = get_path_point_at_s(line, target_s, c_dist)
    return desired_v_eff, a_eff, pos


def find_front_vehicle(ego_pos, ego_s, obstacles, line, threshold=2.0):
    """
    在参考线上找前车
    参数:
        ego_pos: 自车位置 [x, y]
        ego_s: 自车在参考线上的s坐标（可选，传None则自动计算）
        obstacles: 障碍物列表，每个障碍物需有 x, y, v 属性
        line: 参考线
        threshold: 横向距离阈值
    返回: 
        {'dist': 距离, 'v': 速度} 或 None
    """

    if len(obstacles) == 0 or len(line) < 2:
        return None

    if ego_s is None:
        _, ego_s, _ = project_point_to_path(np.array(ego_pos), line)

    min_dist = float('inf')
    front_v = None

    for obs in obstacles:
        obs_pos = np.array([obs.x, obs.y])
        _, s_obs, d_obs = project_point_to_path(obs_pos, line)

        if s_obs > ego_s and abs(d_obs) < threshold:
            dist = s_obs - ego_s
            if dist < min_dist:
                min_dist = dist
                front_v = obs.v

    return {'dist': min_dist, 'v': front_v} if front_v is not None else None


def find_rear_vehicle(ego_pos, ego_s, obstacles, line, threshold=2.0):
    """
    在参考线上找后车
    参数:
        ego_pos: 自车位置 [x, y]
        ego_s: 自车在参考线上的s坐标（可选，传None则自动计算）
        obstacles: 障碍物列表
        line: 参考线
        threshold: 横向距离阈值
    返回: 
        {'dist': 距离, 'v': 速度} 或 None
    """
    if len(obstacles) == 0 or len(line) < 2:
        return None

    if ego_s is None:
        _, ego_s, _ = project_point_to_path(np.array(ego_pos), line)

    min_dist = float('inf')
    rear_v = None

    for obs in obstacles:
        obs_pos = np.array([obs.x, obs.y])
        _, s_obs, d_obs = project_point_to_path(obs_pos, line)

        if s_obs < ego_s and abs(d_obs) < threshold:
            dist = ego_s - s_obs
            if dist < min_dist:
                min_dist = dist
                rear_v = obs.v

    return {'dist': min_dist, 'v': rear_v} if rear_v is not None else None
