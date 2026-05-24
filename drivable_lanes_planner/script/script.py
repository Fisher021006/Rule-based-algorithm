from xodr_parser import XODRParser
from drivable_lanes_planner.config import LaneChangeConfig
from drivable_lanes_planner.planner import DrivableLanesExtender

pkl_dir = r"C:\Users\Administrator\Desktop\TrajectoryPlanning\xodr_parser\cache\农安测试场.pkl"

x_p = XODRParser.load_cache(pkl_dir)
cfg = LaneChangeConfig()
extender = DrivableLanesExtender(x_p, cfg)
flag = extender.plan_shortest_path(-37, -46, -96, -143)
import time

start = time.perf_counter()
result = extender.plan_extended_centerlines(-77, -106).extended_centerlines
print(f"程序运行时间: {(time.perf_counter() - start) * 1000:.2f} ms")
