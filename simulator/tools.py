import math
import numpy as np


def interpolate_angle(theta1, theta2, t):
    """
    在 theta1 和 theta2 之间按比例 t 插值角度（单位：弧度）

    Parameters:
        theta1, theta2: float, 起始和结束角度（建议在 [0, 2π)）
        t: float ∈ [0, 1], 插值比例

    Returns:
        interpolated_theta: float, 插值后的角度 ∈ [0, 2π)
    """
    theta1 = (theta1 + np.pi) % (2 * np.pi) - np.pi
    theta2 = (theta2 + np.pi) % (2 * np.pi) - np.pi
    diff = theta2 - theta1
    diff = (diff + np.pi) % (2 * np.pi) - np.pi
    theta = theta1 + t * diff
    theta = theta % (2 * np.pi)
    return theta




