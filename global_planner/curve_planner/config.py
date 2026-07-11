# 允许换道的车道线类型
CHANGEABLE_MARKS = ['broken', 'broken broken', 'broken solid', 'none']

# ==================== 规划参数 ====================
# 换道惩罚系数（越大越不倾向换道）
LANE_CHANGE_COST = 5.0

# 最小换道长度（米）
MIN_LANE_CHANGE_LENGTH = 10.0

# 最大搜索距离（米）- 用于KDTree查找最近车道
MAX_SEARCH_DISTANCE = 10.0

# ==================== 换道曲线配置（可调整）====================
LANE_CHANGE_CONFIG = {
    # === 单次换道参数 ===
    'single_start_ratio': 0.1,  # 换道开始位置 (范围: 0.1~0.4，越小越早开始)
    'single_end_ratio': 0.7,  # 换道结束位置 (范围: 0.6~0.9，越大越晚结束)
    'single_num_points': 50,  # 曲线采样点数 (范围: 30~100，越大越平滑)
    'single_control_factor': 3.0,  # 贝塞尔控制点系数 (范围: 2.0~5.0，越小曲线越圆润)

}
