import re
import time

import numpy as np
from simulator.vehicle import GVT, VUT

PROTOCOL_MAX_SPEED_MPS = 25.0 / 3.6


def generate_veh_list(task_data, traj_data, xodr_tools):
    xodr_parser, transformer_to_proj, transformer_to_latlon = xodr_tools
    vut_list, gvt_list = [], []
    task_scene_tP = task_data['scene']['trafficParticipant']  # 所有车的交通参与者信息，devices是从中提取的
    for participant in task_scene_tP:
        ptcId = participant["bindDevice"]["ptcId"]
        outline = participant["bindDevice"]["outline"]
        sportParams = participant["bindDevice"]["sportParams"]

        # 如果是被测车
        if participant["subType"] == "TestCar":
            # TODO.暂时不考虑被测车
            continue
            vut = VUT(ptcId, outline, sportParams, xodr_parser)
            vut.pth = pre_vut_traj_process(traj_data[ptcId], transformer_to_proj)
            vut_list.append(vut)
        # 如果是平板车
        elif participant["subType"] in ["Automobile", "Motorbike"]:
            gvt = GVT(ptcId, outline, sportParams, xodr_parser)
            gvt.pre_traj, info_s, info_g = pre_gvt_traj_process(traj_data[ptcId], transformer_to_proj)
            gvt.s_heading = np.radians(traj_data[ptcId]['origin']["heading"])
            gvt.global_planning(info_s, info_g)
            gvt_list.append(gvt)
    return vut_list, gvt_list


def clean_proj_string(proj_str):
    """
    从 proj4 字符串中移除与高程/大地水准面相关的参数，
    避免因缺少 .gtx 文件导致 pyproj 报错。
    """
    # 移除 +geoidgrids=... （包括有值或无值的情况）
    proj_str = re.sub(r'\s*\+\s*geoidgrids\s*=?[^ ]*', '', proj_str)
    # 移除 +vunits=...
    proj_str = re.sub(r'\s*\+\s*vunits\s*=?[^ ]*', '', proj_str)
    # 可选：移除多余的空格和连续的空格
    proj_str = re.sub(r'\s+', ' ', proj_str).strip()
    return proj_str


def pre_vut_traj_process(json_data, transformer_to_proj):
    # 获取原始点的经纬度
    heading_s = np.radians(json_data['origin']["heading"])
    origin_lat = json_data['origin']['latitude']
    origin_lon = json_data['origin']['longitude']
    # 将原始点的经纬度转换为指定投影下的局部坐标
    origin_x, origin_y = transformer_to_proj.transform(origin_lon, origin_lat)
    # 存储去重后的 xy 序列
    xy_sequence = []
    segments = json_data['segments']
    for i, segment in enumerate(segments):
        points = segment['points']
        if i == 0:
            # 第一个 segment：取全部点
            selected_points = points
        else:
            # 后续 segment：跳过第一个点（与上一段末尾重复）
            selected_points = points[1:]
        for point in selected_points:
            local_x, local_y = veh_to_proj(heading_s, point['position']['pointX'], point['position']['pointY'])
            real_x = origin_x + local_x
            real_y = origin_y + local_y
            xy_sequence.append([real_x, real_y])
    return np.array(xy_sequence)


def pre_gvt_traj_process(json_data, transformer_to_proj):
    # 获取 origin 的经纬度
    heading_s = np.radians(json_data['origin']['heading'])
    origin_lat = json_data['origin']['latitude']
    origin_lon = json_data['origin']['longitude']

    # 注意：transform 参数顺序应为 (经度, 纬度)
    origin_x, origin_y = transformer_to_proj.transform(origin_lon, origin_lat)

    # 存储结果：{segment_id: [[x1, y1], [x2, y2], ...]},todo.添加了速度信息
    segmented_xyv = {}

    for segment in json_data['segments']:
        raw_seg_id = segment['id']
        seg_id = raw_seg_id
        points_xyv = []

        for point in segment['points']:
            local_x, local_y = veh_to_proj(heading_s, point['position']['pointX'], point['position']['pointY'])
            real_x = local_x + origin_x
            real_y = local_y + origin_y
            points_xyv.append([real_x, real_y, point["speed"]])
            segmented_xyv[seg_id] = np.array(points_xyv)
    # 返回所有路段的xy字典，此时未裁剪，起点局部坐标，终点局部坐标
    return segmented_xyv, np.array([origin_x, origin_y]), segmented_xyv[seg_id][-1, :2]


def pack_platform_json(gvt_id, gvt_s_heading, s_x_y_lng_lat, segments_dict):
    json_segments = []

    # 按 Segment ID 排序
    sorted_seg_ids = sorted(segments_dict.keys())

    for seg_id in sorted_seg_ids:
        points_array = segments_dict[seg_id]

        if points_array is None or len(points_array) == 0:
            continue

        # 转换点数据 (Local -> Protocol Format)
        json_points = []
        for pt in points_array:
            # 1. 航向转换 (rad -> deg [0, 360])
            h_deg = (450 - np.degrees(pt[2])) % 360

            # 2. 曲率截断 [-0.3, 0.3]
            kappa = np.clip(pt[5], -0.3, 0.3)

            veh_x, veh_y = proj_to_veh(gvt_s_heading, pt[0] - s_x_y_lng_lat[0], pt[1] - s_x_y_lng_lat[1])

            # 3. 构建 Point (协议 4.3)
            p_data = {
                "position": {
                    "pointX": round(float(veh_x), 6),
                    "pointY": round(float(veh_y), 6)
                },
                "heading": round(float(h_deg), 6),
                "speed": round(float(np.clip(pt[3], 0.0, PROTOCOL_MAX_SPEED_MPS)), 6),
                "acceleration": round(float(pt[4]), 6),
                "curvature": round(float(kappa), 6)
            }
            json_points.append(p_data)

        # 构建 Segment (协议 4.2)
        segment_obj = {
            "id": int(seg_id),
            "interval": 100,  # 100ms
            "segmentType": 0,  # 1: 离散点集
            "points": json_points
        }
        json_segments.append(segment_obj)
    sx = json_segments[0]["points"][0]["position"]["pointX"]
    sy = json_segments[0]["points"][0]["position"]["pointY"]
    p1x = json_segments[0]["points"][1]["position"]["pointX"]
    p1y = json_segments[0]["points"][1]["position"]["pointY"]
    sv = json_segments[0]["points"][0]["speed"]
    p1v = json_segments[0]["points"][1]["speed"]
    json_segments[0]["interval"] = 1000 * np.sqrt((sx - p1x) ** 2 + (sy - p1y) ** 2) / (sv + p1v) * 2
    # 构建 Trajectory (协议 4.1)
    trajectory_obj = {
        "ptcId": gvt_id,
        "trajectoryType": 2,  # 2: 增量下发
        "origin": {
            "latitude": s_x_y_lng_lat[3],
            "longitude": s_x_y_lng_lat[2],
            "speed": 0.0,
            "heading": np.degrees(gvt_s_heading)
        },
        "updateStartIndex": json_segments[0]["id"],
        "segments": json_segments
    }

    # 构建 Root Message (协议 3.7)
    message = {
        "uuid": gvt_id,
        "timestamp": int(time.time() * 1000),
        "trajectory": trajectory_obj
    }
    return message


def veh_to_proj(theta, x1, y1):
    # 车身坐标系转局部坐标系
    # theta是正北顺时针增加的弧度，x1y1是以theta向前为x轴，向左为y轴的车身坐标系坐标
    sin_t = np.sin(theta)
    cos_t = np.cos(theta)
    x2 = x1 * sin_t - y1 * cos_t
    y2 = x1 * cos_t + y1 * sin_t
    return x2, y2


def proj_to_veh(theta, x2, y2):
    sin_t = np.sin(theta)
    cos_t = np.cos(theta)
    x1 = x2 * sin_t + y2 * cos_t
    y1 = -x2 * cos_t + y2 * sin_t
    return x1, y1
