int64 timestamp            # 时间戳
float64 acceleration       # 速度指令，km/h
float64 wheel_angle        # 前轮转角指令，度，左偏为正，右偏为负
uint8 gear                 # 档位指令：1-P，2-R，3-N，4-D
uint8 turnlight            # 灯光指令：0-无，1-右转灯亮，2-左转灯亮，4-双闪灯亮
uint8 brkreq               # 刹车请求：0-无，1-急刹
