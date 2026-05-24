float64 lon                # 经度
float64 lat                # 纬度
float64 alt                # 高度
uint8 roadtype             # 道路属性
uint8 lanetype             # 当前车道|总车道(4|4)
float64 heading            # 航向角 degree
float64 pitch              # 俯仰角
float64 roll               # 横滚角
float64 pitchrate          # deg/s
float64 rollrate           # deg/s
float64 yawrate            # deg/s
float64 accx               # m/s^2
float64 accy               # m/s^2
float64 accz               # m/s^2
float64 mile               # 里程 m
float64 velocity           # 速度 m/s
uint8 status               # 导航状态
uint8 rawstatus            # 导航状态，定位正常为 50
uint8 satenum              # 卫星个数
int64 gpstime              # gps时间
uint8 isvalid              # 有效位
int64 timestamp            # 时间戳
float64 speedN             # X轴速度
float64 speedE             # Y轴速度
float64 speedD             # Z轴速度
