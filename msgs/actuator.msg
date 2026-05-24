uint8 epsmethod            # 方向盘使能位
uint8 espmethod            # esp使能位
float32 escbrakepress      # 刹车压力
uint8 gaspedal             # 油门开度
uint8 sysstatus            # 系统状态
float32 speed              # 车速
uint8 lights               # 灯光
uint8 turnLight
uint8 gear
uint8 epb                  # 0:驻车 1:释放
uint8 door
uint8 isvalid              # 有效位
int64 timestamp            # 时间戳
uint8 sendsuccess
uint8 brakepedal
uint8 warning
uint8 battery
uint8 controlover
int32 accelpos
int32 breakflag
int32 breakpos
int32 yaw
int32 mil
float32 soc
float32 batvol
int32 acc
int32 oilperhour
int32 oilhundredkmconsume
int32 oilconsume
int32 autoctrlsig
float32 totalvoltage
float32 totalcurrent
int32 motorspeed
int32 motortorque
uint8 wireControlStatus    # 线控状态，0关闭，1打开
uint8 blinkerStatus        # 双闪状态，0关闭，1打开
float32 accx               # 纵向加速度 m/s2
float32 gaspedalcar        # 油门踏板开度
uint8 brkMode              # 制动状态，0：正常 1：紧急
uint8 ehbWorkMode          # 行车制动系统工作模式，0：待机 1：终端 4：遥控 5：自动
uint8 BrkPressure          # 当前实际制动压力 0-255
uint8 epd                  # 驻车状态
uint8 epdWorkMode          # 驻车工作模式，0：待机 1：终端 4：遥控 5：自动
uint8 epsWorkMode          # 转向工作模式，0：待机 1：终端 4：遥控 5：自动
float32 epsangle           # 实际方向盘转角 -1638～1638
uint8 vehWorkMode          # 车辆工作模式，0：待机 1：终端 4：遥控 5：自动
float32 SteerSpeed         # 车辆纵向速度 0～256 km/h
float32 velAngle           # 车辆运动方向，0:停止 1：向前 2：向后
uint8 error                # 系统故障位
int64 errorlevel           # 系统故障等级
