# 改代码用于将地图信息进行预处理
# 地图储存于data中，预处理后的数据储存于cache

import sys
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# 构建 data 和 cache 的绝对路径
data_dir = os.path.join(parent_dir, "data", "国汽智联通州测试场map终版.xodr")
cache_dir = os.path.join(parent_dir, "cache", "通州测试场.pkl")

# 导入解析器
from xodr_parser import XODRParser

if __name__ == "__main__":
    map_parser = XODRParser(data_dir)
    map_parser.save_cache(cache_dir)

