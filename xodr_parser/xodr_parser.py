"""
XODR解析器模块
解析OpenDRIVE文件，提取道路和车道信息
⭐ 这个模块只需要运行一次，结果可以缓存
"""

import math
import numpy as np
import lxml.etree as ET
from typing import List, Tuple, Dict, Optional
import pickle

try:
    from . import geometry, config
except ImportError:
    import geometry, config


class XODRParser:
    """OpenDRIVE文件解析器"""

    def __init__(self, xodr_path: str = None):
        """
        初始化解析器
        
        参数:
            xodr_path: XODR文件路径，如果为None则创建空解析器（用于从缓存加载）
        """
        # 存储解析结果
        self.roads = {}  # 道路信息
        self.junctions = {}  # 路口信息
        self.lanes = {}  # 车道信息 (详细)
        self.road_centerlines = {}  # 道路中心线坐标
        self.lane_centerlines = {}  # 车道中心线坐标
        self.geo_reference = {'lon0': 0, 'lat0': 0, 'proj': None}

        if xodr_path:
            self.xodr_path = xodr_path
            self.tree = ET.parse(xodr_path)
            self.root = self.tree.getroot()

            # 解析头文件获取投影信息
            self.geo_reference = self._parse_geo_reference()

            # 解析地图
            self._parse_all()

    def _parse_geo_reference(self) -> dict:
        """解析地理参考信息"""
        header = self.root.find('header')
        geo_ref = {'lon0': 0, 'lat0': 0, 'proj': None}

        if header is not None:
            geo_reference = header.find('geoReference')
            if geo_reference is not None and geo_reference.text:
                proj_str = geo_reference.text.strip()
                # 解析投影字符串
                parts = proj_str.split()
                for part in parts:
                    if part.startswith('+lon_0='):
                        geo_ref['lon0'] = float(part.split('=')[1])
                    elif part.startswith('+lat_0='):
                        geo_ref['lat0'] = float(part.split('=')[1])
                geo_ref['proj'] = proj_str

        return geo_ref

    def _parse_all(self):
        """解析所有数据"""
        print("正在解析XODR文件...")

        # 1. 解析道路
        self._parse_roads()

        # 2. 解析路口
        self._parse_junctions()

        # 3. 构建车道详细信息
        self._build_lane_info()

        print(f"解析完成! 道路数: {len(self.roads)}, 车道数: {len(self.lanes)}")

    def _parse_roads(self):
        """解析所有道路"""
        for road in self.root.findall('road'):
            road_id = road.get('id')
            length = float(road.get('length'))
            junction_id = road.get('junction')

            road_data = {
                'id': road_id,
                'length': length,
                'junction': junction_id,
                'geometries': [],
                'lane_sections': [],
                'link': {'predecessor': None, 'successor': None}
            }

            # 解析几何
            plan_view = road.find('planView')
            if plan_view is not None:
                for geo in plan_view.findall('geometry'):
                    geo_data = self._parse_geometry(geo)
                    road_data['geometries'].append(geo_data)

            # 按s排序几何
            road_data['geometries'].sort(key=lambda x: x['s'])

            # 解析链接
            link = road.find('link')
            if link is not None:
                pred = link.find('predecessor')
                if pred is not None:
                    road_data['link']['predecessor'] = {
                        'type': pred.get('elementType'),
                        'id': pred.get('elementId'),
                        'contact': pred.get('contactPoint')
                    }
                succ = link.find('successor')
                if succ is not None:
                    road_data['link']['successor'] = {
                        'type': succ.get('elementType'),
                        'id': succ.get('elementId'),
                        'contact': succ.get('contactPoint')
                    }

            # 解析车道段
            lanes = road.find('lanes')
            if lanes is not None:
                # 解析laneOffset
                lane_offsets = []
                for offset in lanes.findall('laneOffset'):
                    lane_offsets.append({
                        's': float(offset.get('s')),
                        'a': float(offset.get('a')),
                        'b': float(offset.get('b')),
                        'c': float(offset.get('c')),
                        'd': float(offset.get('d'))
                    })
                lane_offsets.sort(key=lambda x: x['s'])

                for section in lanes.findall('laneSection'):
                    section_data = self._parse_lane_section(section, lane_offsets)
                    road_data['lane_sections'].append(section_data)

            self.roads[road_id] = road_data

            # 生成道路中心线
            self.road_centerlines[road_id] = self._generate_road_centerline(road_id)

    def _parse_geometry(self, geo) -> dict:
        """解析单个几何段"""
        geo_data = {
            's': float(geo.get('s')),
            'x': float(geo.get('x')),
            'y': float(geo.get('y')),
            'hdg': float(geo.get('hdg')),
            'length': float(geo.get('length')),
            'type': 'line',
            'params': {}
        }

        # 检查几何类型
        if geo.find('line') is not None:
            geo_data['type'] = 'line'
        elif geo.find('arc') is not None:
            geo_data['type'] = 'arc'
            geo_data['params']['curvature'] = float(geo.find('arc').get('curvature'))
        elif geo.find('spiral') is not None:
            spiral = geo.find('spiral')
            geo_data['type'] = 'spiral'
            geo_data['params']['curvStart'] = float(spiral.get('curvStart'))
            geo_data['params']['curvEnd'] = float(spiral.get('curvEnd'))
        elif geo.find('poly3') is not None:
            poly3 = geo.find('poly3')
            geo_data['type'] = 'poly3'
            geo_data['params']['a'] = float(poly3.get('a'))
            geo_data['params']['b'] = float(poly3.get('b'))
            geo_data['params']['c'] = float(poly3.get('c'))
            geo_data['params']['d'] = float(poly3.get('d'))
        elif geo.find('paramPoly3') is not None:
            pp3 = geo.find('paramPoly3')
            geo_data['type'] = 'paramPoly3'
            geo_data['params']['aU'] = float(pp3.get('aU'))
            geo_data['params']['bU'] = float(pp3.get('bU'))
            geo_data['params']['cU'] = float(pp3.get('cU'))
            geo_data['params']['dU'] = float(pp3.get('dU'))
            geo_data['params']['aV'] = float(pp3.get('aV'))
            geo_data['params']['bV'] = float(pp3.get('bV'))
            geo_data['params']['cV'] = float(pp3.get('cV'))
            geo_data['params']['dV'] = float(pp3.get('dV'))
            geo_data['params']['pRange'] = pp3.get('pRange', 'normalized')

        return geo_data

    def _parse_lane_section(self, section, lane_offsets) -> dict:
        """解析车道段"""
        section_data = {
            's': float(section.get('s')),
            'lane_offsets': lane_offsets,
            'left': [],
            'center': [],
            'right': []
        }

        # 解析中心车道
        center = section.find('center')
        if center is not None:
            for lane in center.findall('lane'):
                lane_data = self._parse_lane(lane)
                section_data['center'].append(lane_data)

        # 解析左侧车道
        left = section.find('left')
        if left is not None:
            for lane in left.findall('lane'):
                lane_data = self._parse_lane(lane)
                section_data['left'].append(lane_data)
            section_data['left'].sort(key=lambda x: x['id'])  # ID升序

        # 解析右侧车道
        right = section.find('right')
        if right is not None:
            for lane in right.findall('lane'):
                lane_data = self._parse_lane(lane)
                section_data['right'].append(lane_data)
            section_data['right'].sort(key=lambda x: x['id'], reverse=True)  # ID降序

        return section_data

    def _parse_lane(self, lane) -> dict:
        """解析单个车道"""
        lane_data = {
            'id': int(lane.get('id')),
            'type': lane.get('type'),
            'widths': [],
            'road_marks': [],
            'link': {'predecessor': None, 'successor': None}
        }

        # 解析宽度
        for width in lane.findall('width'):
            lane_data['widths'].append({
                's': float(width.get('sOffset')),
                'a': float(width.get('a')),
                'b': float(width.get('b')),
                'c': float(width.get('c')),
                'd': float(width.get('d'))
            })

        # 解析道路标记（车道线）
        for mark in lane.findall('roadMark'):
            lane_data['road_marks'].append({
                's': float(mark.get('sOffset', 0)),
                'type': mark.get('type', 'none'),
                'color': mark.get('color', 'white'),
                'width': float(mark.get('width', 0.13)),
                'laneChange': mark.get('laneChange', 'none')
            })

        # 解析链接
        link = lane.find('link')
        if link is not None:
            pred = link.find('predecessor')
            if pred is not None:
                lane_data['link']['predecessor'] = int(pred.get('id'))
            succ = link.find('successor')
            if succ is not None:
                lane_data['link']['successor'] = int(succ.get('id'))

        return lane_data

    def _parse_junctions(self):
        """解析路口"""
        for junction in self.root.findall('junction'):
            junc_id = junction.get('id')
            junc_data = {
                'id': junc_id,
                'connections': []
            }

            for conn in junction.findall('connection'):
                conn_data = {
                    'id': conn.get('id'),
                    'incoming_road': conn.get('incomingRoad'),
                    'connecting_road': conn.get('connectingRoad'),
                    'contact_point': conn.get('contactPoint'),
                    'lane_links': []
                }

                for lane_link in conn.findall('laneLink'):
                    conn_data['lane_links'].append({
                        'from': int(lane_link.get('from')),
                        'to': int(lane_link.get('to'))
                    })

                junc_data['connections'].append(conn_data)

            self.junctions[junc_id] = junc_data

    def _calculate_position(self, road_id: str, s: float, t: float = 0) -> Tuple[float, float, float]:
        """
        计算道路上给定s,t位置的全局坐标
        
        参数:
            road_id: 道路ID
            s: 沿道路的距离
            t: 横向偏移（正为左，负为右）
        
        返回:
            (x, y, hdg): 全局坐标和航向角
        """
        road = self.roads.get(road_id)
        if not road or not road['geometries']:
            return 0, 0, 0

        # 找到对应的几何段
        target_geo = road['geometries'][0]
        for geo in road['geometries']:
            if s >= geo['s']:
                target_geo = geo
            else:
                break

        # 计算在几何段内的相对位置
        ds = s - target_geo['s']
        x0, y0, hdg = target_geo['x'], target_geo['y'], target_geo['hdg']

        # 根据几何类型计算
        if target_geo['type'] == 'line':
            x, y, h = geometry.GeometryCalculator.calculate_line(x0, y0, hdg, ds)
        elif target_geo['type'] == 'arc':
            curv = target_geo['params'].get('curvature', 0)
            x, y, h = geometry.GeometryCalculator.calculate_arc(x0, y0, hdg, curv, ds)
        elif target_geo['type'] == 'spiral':
            curv_start = target_geo['params'].get('curvStart', 0)
            curv_end = target_geo['params'].get('curvEnd', 0)
            x, y, h = geometry.GeometryCalculator.calculate_spiral(
                x0, y0, hdg, curv_start, curv_end, target_geo['length'], ds
            )
        elif target_geo['type'] == 'poly3':
            a = target_geo['params'].get('a', 0)
            b = target_geo['params'].get('b', 0)
            c = target_geo['params'].get('c', 0)
            d = target_geo['params'].get('d', 0)
            x, y, h = geometry.GeometryCalculator.calculate_poly3(x0, y0, hdg, a, b, c, d, ds)
        elif target_geo['type'] == 'paramPoly3':
            p = target_geo['params']
            x, y, h = geometry.GeometryCalculator.calculate_parampoly3(
                x0, y0, hdg,
                p.get('aU', 0), p.get('bU', 0), p.get('cU', 0), p.get('dU', 0),
                p.get('aV', 0), p.get('bV', 0), p.get('cV', 0), p.get('dV', 0),
                p.get('pRange', 'normalized'), ds, target_geo['length']
            )
        else:
            x, y, h = geometry.GeometryCalculator.calculate_line(x0, y0, hdg, ds)

        # 应用横向偏移
        if t != 0:
            x += t * (-math.sin(h))
            y += t * math.cos(h)

        return x, y, h

    def _generate_road_centerline(self, road_id: str, step: float = None) -> np.ndarray:
        """生成道路参考线的采样点"""
        if step is None:
            step = config.CENTERLINE_SAMPLE_STEP

        road = self.roads.get(road_id)
        if not road:
            return np.array([])

        points = []
        s = 0
        while s <= road['length']:
            x, y, _ = self._calculate_position(road_id, s)
            points.append([x, y])
            s += step

        # 确保包含终点
        x, y, _ = self._calculate_position(road_id, road['length'])
        points.append([x, y])

        return np.array(points)

    def _get_lane_offset(self, lane_offsets: List[dict], s: float) -> float:
        """计算给定s位置的车道偏移"""
        if not lane_offsets:
            return 0

        # 找到对应的offset段
        target_offset = lane_offsets[0]
        for offset in lane_offsets:
            if s >= offset['s']:
                target_offset = offset
            else:
                break

        ds = s - target_offset['s']
        return target_offset['a'] + target_offset['b'] * ds + target_offset['c'] * ds ** 2 + target_offset[
            'd'] * ds ** 3

    def _get_lane_width(self, lane_widths: List[dict], s: float) -> float:
        """计算给定s位置的车道宽度"""
        if not lane_widths:
            return config.DEFAULT_LANE_WIDTH

        # 找到对应的width段
        target_width = lane_widths[0]
        for width in lane_widths:
            if s >= width['s']:
                target_width = width
            else:
                break

        ds = s - target_width['s']
        return target_width['a'] + target_width['b'] * ds + target_width['c'] * ds ** 2 + target_width['d'] * ds ** 3

    def _build_lane_info(self):
        """构建详细的车道信息"""
        for road_id, road in self.roads.items():
            for sec_idx, section in enumerate(road['lane_sections']):
                # 计算当前section的长度
                next_s = road['length']
                if sec_idx + 1 < len(road['lane_sections']):
                    next_s = road['lane_sections'][sec_idx + 1]['s']
                section_length = next_s - section['s']

                # 处理左侧车道
                for lane in section['left']:
                    self._process_lane(road_id, section, lane, section_length, 'left')

                # 处理右侧车道
                for lane in section['right']:
                    self._process_lane(road_id, section, lane, section_length, 'right')

    def _process_lane(self, road_id: str, section: dict, lane: dict,
                      section_length: float, side: str):
        """处理单个车道，生成完整信息"""
        lane_id = int(lane['id'])

        # 生成唯一的车道标识
        unique_lane_id = f"{road_id}.{int(section['s'])}.{lane_id}"

        # 计算车道中心线
        centerline = self._generate_lane_centerline(road_id, section, lane, section_length, side)
        centerline = self._normalize_lane_centerline_direction(centerline, lane_id)

        if len(centerline) == 0:
            return

        # 获取车道线类型
        left_mark, right_mark = self._get_lane_marks(section, lane, side)

        # 获取相邻车道
        left_neighbor, right_neighbor = self._get_neighbor_lanes(road_id, section, lane, side)

        # 获取前后继车道
        predecessor, successor = self._get_lane_connections(road_id, section, lane, side)

        # 计算车道长度
        if len(centerline) > 1:
            lane_length = sum(np.linalg.norm(centerline[i + 1] - centerline[i])
                              for i in range(len(centerline) - 1))
        else:
            lane_length = section_length

        self.lanes[unique_lane_id] = {
            'lane_id': unique_lane_id,
            'road_id': road_id,
            'section_s': section['s'],
            'lane_num': lane_id,
            'type': lane['type'],
            'length': lane_length,
            'center_vertices': centerline,
            'predecessor': predecessor,
            'successor': successor,
            'left_lane': left_neighbor,
            'right_lane': right_neighbor,
            'left_mark_type': left_mark['type'] if left_mark else 'none',
            'left_mark_color': left_mark['color'] if left_mark else 'none',
            'right_mark_type': right_mark['type'] if right_mark else 'none',
            'right_mark_color': right_mark['color'] if right_mark else 'none'
        }

        self.lane_centerlines[unique_lane_id] = centerline

    @staticmethod
    def _normalize_lane_centerline_direction(centerline: np.ndarray, lane_id: int) -> np.ndarray:
        """
        统一车道中心线点序为“车辆行驶方向”。

        OpenDRIVE 中左侧车道(lane_id > 0)的行驶方向与道路参考线 s 增大方向相反，
        因此这里直接对左侧车道做一次反转，保证上层模块拿到的中心线方向和
        predecessor/successor 的语义一致。
        """
        centerline = np.asarray(centerline, dtype=float)
        if lane_id > 0 and len(centerline) > 1:
            return centerline[::-1].copy()
        return centerline.copy()

    def _generate_lane_centerline(self, road_id: str, section: dict, lane: dict,
                                  section_length: float, side: str,
                                  step: float = None) -> np.ndarray:
        """生成车道中心线"""
        if step is None:
            step = config.CENTERLINE_SAMPLE_STEP

        road = self.roads[road_id]
        lane_id = lane['id']
        points = []

        s = section['s']
        end_s = section['s'] + section_length

        while s <= end_s:
            # 计算横向偏移
            t_offset = self._get_lane_offset(section['lane_offsets'], s - section['s'])

            # 累计到当前车道的偏移
            if side == 'right':
                # 右侧车道，ID为负
                all_lanes = section['right']
                cumulative_offset = 0
                for l in all_lanes:
                    if l['id'] > lane_id:  # 更靠近中心的车道
                        width = self._get_lane_width(l['widths'], s - section['s'])
                        cumulative_offset -= width
                    elif l['id'] == lane_id:
                        width = self._get_lane_width(l['widths'], s - section['s'])
                        cumulative_offset -= width / 2
                        break
                t = t_offset + cumulative_offset
            else:
                # 左侧车道，ID为正
                all_lanes = section['left']
                cumulative_offset = 0
                for l in all_lanes:
                    if l['id'] < lane_id:  # 更靠近中心的车道
                        width = self._get_lane_width(l['widths'], s - section['s'])
                        cumulative_offset += width
                    elif l['id'] == lane_id:
                        width = self._get_lane_width(l['widths'], s - section['s'])
                        cumulative_offset += width / 2
                        break
                t = t_offset + cumulative_offset

            x, y, _ = self._calculate_position(road_id, s, t)
            points.append([x, y])
            s += step

        # 确保包含终点
        t_offset = self._get_lane_offset(section['lane_offsets'], section_length)
        if side == 'right':
            all_lanes = section['right']
            cumulative_offset = 0
            for l in all_lanes:
                if l['id'] > lane_id:
                    width = self._get_lane_width(l['widths'], section_length)
                    cumulative_offset -= width
                elif l['id'] == lane_id:
                    width = self._get_lane_width(l['widths'], section_length)
                    cumulative_offset -= width / 2
                    break
            t = t_offset + cumulative_offset
        else:
            all_lanes = section['left']
            cumulative_offset = 0
            for l in all_lanes:
                if l['id'] < lane_id:
                    width = self._get_lane_width(l['widths'], section_length)
                    cumulative_offset += width
                elif l['id'] == lane_id:
                    width = self._get_lane_width(l['widths'], section_length)
                    cumulative_offset += width / 2
                    break
            t = t_offset + cumulative_offset

        x, y, _ = self._calculate_position(road_id, end_s, t)
        points.append([x, y])

        return np.array(points)

    def _get_lane_marks(self, section: dict, lane: dict, side: str) -> Tuple[dict, dict]:
        """获取按车辆行驶方向定义的左右边界标记类型。"""
        lane_id = lane['id']

        # 当前车道的roadMark是其外侧边界
        outer_mark = lane['road_marks'][0] if lane['road_marks'] else {'type': 'none', 'color': 'none'}

        # 内侧边界需要查找相邻车道
        inner_mark = {'type': 'none', 'color': 'none'}

        if side == 'right':
            # 右侧车道，内侧是ID更大的车道（或center）
            if lane_id == -1:
                # 最内侧车道，内边界是中心线
                center_lane = section['center'][0] if section['center'] else None
                if center_lane and center_lane['road_marks']:
                    inner_mark = center_lane['road_marks'][0]
            else:
                for l in section['right']:
                    if l['id'] == lane_id + 1:
                        if l['road_marks']:
                            inner_mark = l['road_marks'][0]
                        break
            # 右侧车道(lane_id < 0)行驶方向与 road reference line 同向：
            # 车辆左侧是内侧，车辆右侧是外侧。
            return inner_mark, outer_mark
        else:
            # 左侧车道，内侧是ID更小的车道（或center）
            if lane_id == 1:
                center_lane = section['center'][0] if section['center'] else None
                if center_lane and center_lane['road_marks']:
                    inner_mark = center_lane['road_marks'][0]
            else:
                for l in section['left']:
                    if l['id'] == lane_id - 1:
                        if l['road_marks']:
                            inner_mark = l['road_marks'][0]
                        break
            # 左侧车道(lane_id > 0)行驶方向与 road reference line 反向：
            # 车辆左侧是内侧，车辆右侧是外侧。
            return inner_mark, outer_mark

    def _get_neighbor_lanes(self, road_id: str, section: dict, lane: dict,
                            side: str) -> Tuple[str, str]:
        """获取按车辆行驶方向定义的左右相邻车道 ID。"""
        lane_id = lane['id']
        left_neighbor = None
        right_neighbor = None

        if side == 'right':
            # 右侧车道(lane_id < 0)行驶方向与 road reference line 同向：
            # ID+1 靠内侧，是车辆左侧；ID-1 靠外侧，是车辆右侧。
            for l in section['right']:
                if l['id'] == lane_id + 1:
                    left_neighbor = f"{road_id}.{int(section['s'])}.{l['id']}"
                elif l['id'] == lane_id - 1:
                    right_neighbor = f"{road_id}.{int(section['s'])}.{l['id']}"
        else:
            # 左侧车道(lane_id > 0)行驶方向与 road reference line 反向：
            # ID-1 靠内侧，是车辆左侧；ID+1 靠外侧，是车辆右侧。
            for l in section['left']:
                if l['id'] == lane_id - 1:
                    left_neighbor = f"{road_id}.{int(section['s'])}.{l['id']}"
                elif l['id'] == lane_id + 1:
                    right_neighbor = f"{road_id}.{int(section['s'])}.{l['id']}"

        return left_neighbor, right_neighbor

    def _get_lane_connections(self, road_id: str, section: dict, lane: dict,
                              side: str) -> Tuple[List[str], List[str]]:
        """获取车道的前后继连接
        
        注意: OpenDRIVE规范中，左侧车道(lane_id > 0)的行驶方向与道路参考线方向相反，
        因此前后继需要反转：
        - 右侧车道(lane_id < 0): road.successor -> lane.successor, road.predecessor -> lane.predecessor
        - 左侧车道(lane_id > 0): road.successor -> lane.predecessor, road.predecessor -> lane.successor
        
        对于lane.link的定义：
        - lane.link.predecessor: 指向road.predecessor方向连接的车道ID
        - lane.link.successor: 指向road.successor方向连接的车道ID
        
        所以对于左侧车道的行驶方向：
        - 行驶predecessor = road.successor方向 = 使用lane.link.successor
        - 行驶successor = road.predecessor方向 = 使用lane.link.predecessor
        """
        road = self.roads[road_id]
        lane_id = lane['id']
        predecessors = []
        successors = []

        # 从lane的link中获取前后继lane ID
        pred_lane_id = lane['link']['predecessor']
        succ_lane_id = lane['link']['successor']

        # 获取道路级别的前后继
        road_pred = road['link']['predecessor']
        road_succ = road['link']['successor']

        # 判断是否为左侧车道（lane_id > 0）
        # 左侧车道行驶方向与道路参考线方向相反，需要交换前后继
        is_left_lane = lane_id > 0

        if is_left_lane:
            # 左侧车道：road的successor是车道的行驶predecessor，road的predecessor是车道的行驶successor

            # 处理车道的行驶predecessor（从road的successor中查找）
            # 使用lane.link.successor ID（因为它指向road.successor方向）
            if road_succ:
                if road_succ['type'] == 'road':
                    succ_road_id = road_succ['id']
                    if succ_lane_id is not None:  # 使用lane的successor ID
                        predecessors.append(f"{succ_road_id}.0.{succ_lane_id}")
                elif road_succ['type'] == 'junction':
                    # 路口连接
                    junc_id = road_succ['id']
                    if junc_id in self.junctions:
                        for conn in self.junctions[junc_id]['connections']:
                            # 查找connecting_road是当前road的连接（即当前road是出口）
                            if conn['connecting_road'] == road_id:
                                for ll in conn['lane_links']:
                                    if ll['to'] == lane_id:
                                        incoming_road_id = conn['incoming_road']
                                        predecessors.append(f"{incoming_road_id}.0.{ll['from']}")

            # 处理车道的行驶successor（从road的predecessor中查找）
            # 使用lane.link.predecessor ID（因为它指向road.predecessor方向）
            if road_pred:
                if road_pred['type'] == 'road':
                    pred_road_id = road_pred['id']
                    if pred_lane_id is not None:  # 使用lane的predecessor ID
                        successors.append(f"{pred_road_id}.0.{pred_lane_id}")
                elif road_pred['type'] == 'junction':
                    # 对于左侧车道，从road.predecessor的junction找successor
                    junc_id = road_pred['id']
                    if junc_id in self.junctions:
                        for conn in self.junctions[junc_id]['connections']:
                            # 查找incoming_road是当前road的连接（即当前road是入口）
                            if conn['incoming_road'] == road_id:
                                for ll in conn['lane_links']:
                                    if ll['from'] == lane_id:
                                        conn_road_id = conn['connecting_road']
                                        successors.append(f"{conn_road_id}.0.{ll['to']}")
        else:
            # 右侧车道：正常处理
            # 处理后继
            if road_succ:
                if road_succ['type'] == 'road':
                    succ_road_id = road_succ['id']
                    if succ_lane_id is not None:
                        # 查找后继道路的对应车道
                        successors.append(f"{succ_road_id}.0.{succ_lane_id}")
                elif road_succ['type'] == 'junction':
                    # 路口连接，需要查找junction中的连接关系
                    junc_id = road_succ['id']
                    if junc_id in self.junctions:
                        for conn in self.junctions[junc_id]['connections']:
                            if conn['incoming_road'] == road_id:
                                for ll in conn['lane_links']:
                                    if ll['from'] == lane_id:
                                        conn_road_id = conn['connecting_road']
                                        successors.append(f"{conn_road_id}.0.{ll['to']}")

            # 处理前驱
            if road_pred:
                if road_pred['type'] == 'road':
                    pred_road_id = road_pred['id']
                    if pred_lane_id is not None:
                        predecessors.append(f"{pred_road_id}.0.{pred_lane_id}")
                elif road_pred['type'] == 'junction':
                    # 需要反向查找
                    junc_id = road_pred['id']
                    if junc_id in self.junctions:
                        for conn in self.junctions[junc_id]['connections']:
                            if conn['connecting_road'] == road_id:
                                for ll in conn['lane_links']:
                                    if ll['to'] == lane_id:
                                        incoming_road_id = conn['incoming_road']
                                        predecessors.append(f"{incoming_road_id}.0.{ll['from']}")

        return predecessors, successors

    # ==================== 缓存功能 ====================
    def save_cache(self, cache_path: str):
        """
        保存解析结果到缓存文件
        
        参数:
            cache_path: 缓存文件路径 (.pkl)
        """
        cache_data = {
            'roads': self.roads,
            'junctions': self.junctions,
            'lanes': self.lanes,
            'road_centerlines': self.road_centerlines,
            'lane_centerlines': self.lane_centerlines,
            'geo_reference': self.geo_reference,
            'lane_centerline_direction_normalized': True,
        }

        with open(cache_path, 'wb') as f:
            pickle.dump(cache_data, f)

        print(f"✅ 缓存已保存至: {cache_path}")

    @classmethod
    def load_cache(cls, cache_path: str) -> 'XODRParser':
        """
        从缓存文件加载解析结果
        
        参数:
            cache_path: 缓存文件路径 (.pkl)
        
        返回:
            XODRParser实例
        """
        parser = cls()  # 创建空解析器

        with open(cache_path, 'rb') as f:
            cache_data = pickle.load(f)

        parser.roads = cache_data['roads']
        parser.junctions = cache_data['junctions']
        parser.lanes = cache_data['lanes']
        parser.road_centerlines = cache_data['road_centerlines']
        parser.lane_centerlines = cache_data['lane_centerlines']
        parser.geo_reference = cache_data['geo_reference']

        if not cache_data.get('lane_centerline_direction_normalized', False):
            for lane_id, lane_info in parser.lanes.items():
                lane_num = lane_info.get('lane_num')
                if lane_num is None:
                    try:
                        lane_num = int(str(lane_id).split('.')[-1])
                    except Exception:
                        lane_num = 0
                normalized = parser._normalize_lane_centerline_direction(
                    parser.lane_centerlines[lane_id], lane_num
                )
                parser.lane_centerlines[lane_id] = normalized
                parser.lanes[lane_id]['center_vertices'] = normalized
            print("⚠ 缓存中的 lane_centerlines 已按行驶方向自动补正")

        print(f"✅ 已从缓存加载: {cache_path}")
        print(f"   道路数: {len(parser.roads)}, 车道数: {len(parser.lanes)}")

        return parser

    def get_map_bounds(self) -> dict:
        """获取地图坐标范围"""
        all_x, all_y = [], []
        for centerline in self.lane_centerlines.values():
            if len(centerline) > 0:
                all_x.extend(centerline[:, 0])
                all_y.extend(centerline[:, 1])

        if all_x and all_y:
            return {
                'x_min': min(all_x),
                'x_max': max(all_x),
                'y_min': min(all_y),
                'y_max': max(all_y)
            }
        return None
