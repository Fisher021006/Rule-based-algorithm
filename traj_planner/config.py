from dataclasses import dataclass


@dataclass
class VehicleState:
    """车辆状态"""
    x: float = 0.0  # 位置x
    y: float = 0.0  # 位置y
    heading: float = 0.0  # 航向角 (rad)
    v: float = 0.0  # 速度 (m/s)
    a: float = 0.0  # 加速度 (m/s^2)
    kappa: float = 0.0  # 曲率 (1/m)
