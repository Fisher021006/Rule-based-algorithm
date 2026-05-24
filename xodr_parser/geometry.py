"""
几何计算工具模块
处理OpenDRIVE中的各种几何形状计算
"""

import math


class GeometryCalculator:
    """处理OpenDRIVE中的各种几何形状计算"""
    
    @staticmethod
    def calculate_line(x0: float, y0: float, hdg: float, s: float) -> tuple:
        """
        计算直线段上的点
        
        参数:
            x0, y0: 起点坐标
            hdg: 航向角（弧度）
            s: 沿线距离
        
        返回:
            (x, y, hdg): 新位置和航向角
        """
        x = x0 + s * math.cos(hdg)
        y = y0 + s * math.sin(hdg)
        return x, y, hdg
    
    @staticmethod
    def calculate_arc(x0: float, y0: float, hdg: float, 
                      curvature: float, s: float) -> tuple:
        """
        计算圆弧段上的点
        
        参数:
            x0, y0: 起点坐标
            hdg: 起始航向角
            curvature: 曲率（1/半径）
            s: 沿弧距离
        
        返回:
            (x, y, new_hdg): 新位置和航向角
        """
        if abs(curvature) < 1e-10:
            return GeometryCalculator.calculate_line(x0, y0, hdg, s)
        
        # 圆弧计算
        radius = 1.0 / curvature
        # 圆心位置
        cx = x0 - radius * math.sin(hdg)
        cy = y0 + radius * math.cos(hdg)
        
        # 新的航向角
        new_hdg = hdg + s * curvature
        
        # 新的位置
        x = cx + radius * math.sin(new_hdg)
        y = cy - radius * math.cos(new_hdg)
        
        return x, y, new_hdg
    
    @staticmethod
    def calculate_spiral(x0: float, y0: float, hdg: float,
                         curv_start: float, curv_end: float, 
                         length: float, s: float) -> tuple:
        """
        计算螺旋线（clothoid）上的点 - 使用数值积分
        
        参数:
            x0, y0: 起点坐标
            hdg: 起始航向角
            curv_start: 起始曲率
            curv_end: 终点曲率
            length: 螺旋线总长度
            s: 沿线距离
        
        返回:
            (x, y, h): 新位置和航向角
        """
        if s <= 0:
            return x0, y0, hdg
        
        # 螺旋线参数
        curv_dot = (curv_end - curv_start) / length if length > 0 else 0
        
        # 数值积分
        n_steps = max(int(s / 0.1), 10)
        ds = s / n_steps
        
        x, y, h = x0, y0, hdg
        curv = curv_start
        
        for _ in range(n_steps):
            x += ds * math.cos(h)
            y += ds * math.sin(h)
            h += ds * curv
            curv += ds * curv_dot
        
        return x, y, h
    
    @staticmethod
    def calculate_poly3(x0: float, y0: float, hdg: float,
                        a: float, b: float, c: float, d: float, 
                        s: float) -> tuple:
        """
        计算三次多项式上的点
        
        参数:
            x0, y0: 起点坐标
            hdg: 起始航向角
            a, b, c, d: 多项式系数
            s: 沿线距离
        
        返回:
            (x, y, new_hdg): 新位置和航向角
        """
        # 局部坐标
        u = s
        v = a + b*s + c*s**2 + d*s**3
        
        # 转换到全局坐标
        cos_h = math.cos(hdg)
        sin_h = math.sin(hdg)
        
        x = x0 + u * cos_h - v * sin_h
        y = y0 + u * sin_h + v * cos_h
        
        # 切线方向
        dv = b + 2*c*s + 3*d*s**2
        local_hdg = math.atan2(dv, 1.0)
        new_hdg = hdg + local_hdg
        
        return x, y, new_hdg
    
    @staticmethod
    def calculate_parampoly3(x0: float, y0: float, hdg: float,
                             aU: float, bU: float, cU: float, dU: float,
                             aV: float, bV: float, cV: float, dV: float,
                             p_range: str, s: float, length: float) -> tuple:
        """
        计算参数化三次多项式上的点
        
        参数:
            x0, y0: 起点坐标
            hdg: 起始航向角
            aU, bU, cU, dU: U方向多项式系数
            aV, bV, cV, dV: V方向多项式系数
            p_range: 参数范围类型 ('normalized' 或 'arcLength')
            s: 沿线距离
            length: 几何段总长度
        
        返回:
            (x, y, new_hdg): 新位置和航向角
        """
        # 参数p
        if p_range == 'normalized':
            p = s / length if length > 0 else 0
        else:
            p = s
        
        # 局部坐标
        u = aU + bU*p + cU*p**2 + dU*p**3
        v = aV + bV*p + cV*p**2 + dV*p**3
        
        # 转换到全局坐标
        cos_h = math.cos(hdg)
        sin_h = math.sin(hdg)
        
        x = x0 + u * cos_h - v * sin_h
        y = y0 + u * sin_h + v * cos_h
        
        # 计算切线方向
        du = bU + 2*cU*p + 3*dU*p**2
        dv = bV + 2*cV*p + 3*dV*p**2
        local_hdg = math.atan2(dv, du) if abs(du) > 1e-10 else 0
        new_hdg = hdg + local_hdg
        
        return x, y, new_hdg
