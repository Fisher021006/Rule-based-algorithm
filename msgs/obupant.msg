string ptc_id        # 交通参与者 ID，平台背景车 ID 会映射到这里
int32 ptc_type       # 交通参与者类型，1 通常表示机动车
int32 source         # 数据来源，Adapter 默认用 1 表示平台 Redis 背景车
string source_id     # 来源通道 ID，用于追踪该车来自哪个平台背景车通道
int32 sec_mark       # 秒内毫秒时间戳，当前本地推演暂填 0
float64 pos_lon      # 经度，平台字段 longitude/lon/lng 映射到这里
float64 pos_lat      # 纬度，平台字段 latitude/lat 映射到这里
float64 lon          # 经度冗余字段，兼容现有算法读取习惯
float64 lat          # 纬度冗余字段，兼容现有算法读取习惯
float64 speed        # 速度，单位 m/s，平台 km/h 会在 Adapter 中转换
float64 heading      # 航向角，单位 degree，沿用平台 courseAngle/heading 定义
float64 accel        # 纵向加速度，单位 m/s^2
float64 width        # 车辆宽度，单位 m
float64 length       # 车辆长度，单位 m
float64 height       # 车辆高度，单位 m
uint8 isvalid        # 有效位，1 表示该背景车记录可用
