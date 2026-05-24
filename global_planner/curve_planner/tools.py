import numpy as np


def calculate_angle(dx, dy):
    """计算并返回在[0, 2*pi]范围内的角度"""
    angle = np.arctan2(dy, dx)  # 返回值在[-pi, pi]
    if angle < 0:
        angle += 2 * np.pi  # 调整到[0, 2*pi]范围内
    return angle


def project_and_cut_path(path, start_point, end_point):
    path = np.array(path)
    start_idx = np.argmin(np.sum((path - start_point) ** 2, axis=1))
    end_idx = np.argmin(np.sum((path - end_point) ** 2, axis=1))

    if start_idx > end_idx:  # 确保开始索引小于结束索引
        start_idx, end_idx = end_idx, start_idx

    new_path = path[start_idx:end_idx + 1]

    angles = []
    for i in range(1, len(new_path) - 1):
        dx = new_path[i + 1][0] - new_path[i - 1][0]
        dy = new_path[i + 1][1] - new_path[i - 1][1]
        angles.append(calculate_angle(dx, dy))

    # 添加起始点和结束点的角度
    # 对于起始点，我们可以用它与下一个点形成的向量来计算角度
    first_dx = new_path[1][0] - new_path[0][0]
    first_dy = new_path[1][1] - new_path[0][1]
    angles.insert(0, calculate_angle(first_dx, first_dy))

    # 对于结束点，可以用前一个点与它的向量来计算角度
    last_dx = new_path[-1][0] - new_path[-2][0]
    last_dy = new_path[-1][1] - new_path[-2][1]
    angles.append(calculate_angle(last_dx, last_dy))

    # 将角度作为第三列添加到new_path中
    new_path_with_angles = np.column_stack((new_path, np.array(angles)))

    return new_path_with_angles
