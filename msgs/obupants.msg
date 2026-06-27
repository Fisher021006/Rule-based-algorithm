int64 timestamp                  # OBU 消息组时间戳，单位 ms
uint8 isvalid                    # 整组消息有效位，1 表示 pants 数组可用
int32 msg_cnt                    # pants 数组中的背景车数量
common_msgs/obupant[] pants      # 背景车/交通参与者列表，供场景算法作为障碍物输入
