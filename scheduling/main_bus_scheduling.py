"""
Author: Zili Tian
Date: 2024/9/9
Description:
    This script is used to generate the bus schedule as key input data of test4mosa.py.

"""

import numpy as np
import pandas as pd
import geopandas as gpd
import networkx as nx
import multiprocessing
from functools import partial
from datetime import datetime, timedelta
import math
import ast
from haversine import haversine, Unit
from itertools import islice


class Line:
    def __init__(self,
                 lineId: int,
                 lineName: str,
                 lineCo: str,
                 lineDirection: str,
                 lineLength: float = None,  # km
                 lineStops: list = None,
                 lineCoords: list = None,
                 lineBusPairShortestPath: list = None,  # km
                 lineHeadway: list = None,  # min
                 lineVelocity: list = None,  # km/h
                 lineDuration: list = None  # min
                 ):
        if lineHeadway is None:
            # lineHeadway = [120] * 5 + [15] * 18 + [120]
            lineHeadway = [120] * 10 + [45] * 5 + [120]*9  # for small scale testing
        elif len(lineHeadway) != 24:
            lineHeadway = [120] * 5 + [15] * 18 + [120]
        else:
            lineHeadway = [math.ceil(num) for num in lineHeadway]
        if lineVelocity is None:
            lineVelocity = [15] * 24
        elif len(lineVelocity) != 24:
            lineVelocity = [15] * 24

        # compulsory input
        self.lineId = lineId
        self.lineName = lineName
        self.lineCo = lineCo
        self.lineDirection = lineDirection
        self.lineStops = lineStops
        self.lineCoords = lineCoords  # convert string to list

        # other inputs using for calculating the duration of every section between two adjoined stops
        self.lineLength = lineLength
        self.lineBusPairShortestPath = lineBusPairShortestPath
        self.lineHeadway = lineHeadway
        self.lineVelocity = lineVelocity
        if lineDuration is not None:
            if len(lineDuration) != 24:
                lineDuration = None
        if lineDuration is not None:
            self.lineDuration = [math.ceil(num) for num in lineDuration]
        elif lineLength is not None:
            length_with_conversion = lineLength * 60
            self.lineDuration = [math.ceil(length_with_conversion / value) for value in lineVelocity]
        else:
            self.lineDuration = [30] * 24
            print(f"【{self.lineCo} {self.lineName} to {self.lineDirection}】's duration has been assumed")

        # need to be generated
        self.schedule = pd.DataFrame()

    def timetable_scheduling(self, year, month, day):
        """for example: 2024 7 15"""
        c_start_time = datetime(year, month, day)  # current s_time
        c_hour = c_start_time.hour  # current hour
        former_hour = c_hour
        schedule = []  # store the start time, end time, velocity and duration
        while c_hour >= former_hour:  # break if circle to the next day
            former_hour = c_hour
            if self.lineHeadway[c_hour] <= 60:
                schedule.append([c_start_time, c_start_time + timedelta(minutes=self.lineDuration[c_hour]),
                                 self.lineVelocity[c_hour], self.lineDuration[c_hour]])
                c_start_time = c_start_time + timedelta(minutes=self.lineHeadway[c_hour])
            else:
                c_start_time = c_start_time + timedelta(minutes=60)
            c_hour = c_start_time.hour
        schedule = pd.DataFrame(schedule, columns=['s_time', 'e_time', 'avg_velocity', 'duration'])
        if self.lineLength is not None:
            schedule['distance'] = self.lineLength
        else:
            schedule['distance'] = ''
        schedule['lineName'] = self.lineName
        schedule['lineCo'] = self.lineCo
        schedule['lineDirection'] = self.lineDirection
        schedule['destination_coords'] = [self.lineCoords[-1]] * len(schedule)
        schedule['orientation_coords'] = [self.lineCoords[0]] * len(schedule)
        self.schedule = schedule


def find_duplicates(lineDict):
    duplicates_by_key = {}
    for key, line in lineDict.items():
        identifier = (line.lineCo, line.lineName)
        if identifier not in duplicates_by_key:
            duplicates_by_key[identifier] = [key]
        else:
            duplicates_by_key[identifier].append(key)
    return list(duplicates_by_key.values())


def swap_tuple(t):
    return t[1], t[0]


def find_and_remove_nearest(o, d):
    """找到o和d中最接近的一对点并移除，返回距离和这对点"""
    if not o or not d:
        return None, None, None  # 如果任一列表为空，则返回None
    # 计算所有可能的点对距离
    distances = [(haversine(swap_tuple(o_i), swap_tuple(d_j)), o_i, d_j) for o_i in o for d_j in d]

    # 找到最小距离的点对
    min_distance, o_nearest, d_nearest = min(distances)

    # 从列表中移除已匹配的点
    o.remove(o_nearest)
    d.remove(d_nearest)

    return min_distance, o_nearest, d_nearest


def vehicle_scheduling(lineDict, minInterval=5, hold_unknown=True, speed=25):
    """分配车辆，不考虑跨线调度，到达和下次出发时间间隔超过t但最短的即可匹配为一辆车
    可能存在单向、双向线路和三向（A to B to C to A, e.g.屿巴4路）
    一般能闭环，无法闭环就按速度V回到首站
    lineDict: dict(index is lineId); minInterval: minutes;
    hold_unknown: if holding unknown routes (by cooperation)
    speed: km/h speed with no passengers"""
    VSParkingDF = []
    route_list = find_duplicates(lineDict)  # group same route
    vehicle_no = 0
    for route_group in route_list:
        if not hold_unknown and 'unknown' in lineDict[route_group[0]].lineCo.lower():
            continue
        direction_num = len(route_group)
        route_schedule = []
        for routeID in route_group:
            if len(route_schedule):
                route_schedule = pd.concat([route_schedule, lineDict[routeID].schedule], axis=0)
            else:
                route_schedule = lineDict[routeID].schedule.copy(deep=False)
        route_schedule.sort_values(by='s_time', ascending=True, inplace=True)
        route_schedule.reset_index(drop=True, inplace=True)

        od_site = route_schedule.drop_duplicates(subset=['destination_coords', 'orientation_coords'])
        d_site = od_site['destination_coords'].tolist()
        o_site = od_site['orientation_coords'].tolist()

        d2o = {}
        while o_site and d_site:
            distance, o_point, d_point = find_and_remove_nearest(o_site, d_site)
            d2o[d_point] = [o_point, round(distance*1.4/speed*60)]  # 空载duration，后面用最短路替换

        # for d_coord in d_site:  # former code getting the d2o, but cannot cover all the sites
        #     closest_o = min(o_site, key=lambda o: haversine(swap_tuple(o), swap_tuple(d_coord)))
        #     if haversine(swap_tuple(closest_o),swap_tuple(d_coord))<0.3:
        #         d2o[d_coord] = closest_o

        route_schedule['vehicle_no'] = np.nan
        route_schedule['trip_no'] = np.nan
        for m in range(len(route_schedule)):
            if np.isnan(route_schedule['vehicle_no'].iloc[m]):
                vehicle_no += 1
                trip_no = 1
                route_schedule.loc[m, 'trip_no'] = trip_no
                route_schedule.loc[m, 'vehicle_no'] = vehicle_no
                c_e_time = route_schedule['e_time'].iloc[m]
                c_e_loc = route_schedule['destination_coords'].iloc[m]
                for n in range(m+1, len(route_schedule)):
                    if (route_schedule['orientation_coords'].iloc[n] == d2o[c_e_loc][0] and
                            route_schedule['s_time'].iloc[n] - c_e_time >= timedelta(minutes=minInterval+d2o[c_e_loc][1])):
                        trip_no += 1
                        route_schedule.loc[n, 'trip_no'] = trip_no
                        route_schedule.loc[n, 'vehicle_no'] = vehicle_no
                        c_e_time = route_schedule['e_time'].iloc[n]
                        c_e_loc = route_schedule['destination_coords'].iloc[n]
                        continue
        route_schedule['vehicle_no'] = route_schedule['vehicle_no'].astype(int)
        route_schedule['trip_no'] = route_schedule['trip_no'].astype(int)

        if len(VSParkingDF):
            VSParkingDF = pd.concat([VSParkingDF, route_schedule], axis=0)
        else:
            VSParkingDF = route_schedule.copy(deep=False)

    VSParkingDF['distance'] = VSParkingDF['distance']*1000
    VSParkingDF.sort_values(by='e_time', ascending=True, inplace=True)
    VSParkingDF.reset_index(drop=True, inplace=True)
    desired_order = ['vehicle_no', 'trip_no', 's_time', 'e_time', 'destination_coords', 'distance', 'avg_velocity',
                     'duration', 'lineName', 'lineCo', 'lineDirection', 'orientation_coords',]
    VSParkingDF = VSParkingDF[desired_order]

    return VSParkingDF


if __name__ == '__main__':
    line_info = pd.read_csv(r"../data/line_info.csv")
    lines = []
    for i in range(len(line_info)):
        _line = Line(lineId=i,  # No lineId in the raw data
                     lineName=line_info['lineName'].iloc[i],
                     lineCo=line_info['co'].iloc[i],
                     lineDirection=line_info['lineDirection'].iloc[i],
                     lineLength=line_info['lineLength'].iloc[i],
                     lineStops=line_info['lineStops'].iloc[i],
                     lineCoords=ast.literal_eval(line_info['lineCoords'].iloc[i]),
                     lineBusPairShortestPath=ast.literal_eval(line_info['lineBusPairShortestPath'].iloc[i]))
        lines.append(_line)

    for _line in lines:
        _line.timetable_scheduling(2024, 7, 15)

    all_schedule = []
    for _line in lines:
        if len(all_schedule):
            all_schedule = pd.concat([all_schedule, _line.schedule], axis=0)
        else:
            all_schedule = _line.schedule.copy(deep=False)

    line_dict = {_line.lineId: _line for _line in lines}  # convert line_info list into dict

    vs_parking_df = vehicle_scheduling(line_dict, minInterval=5, hold_unknown=False)
    vs_parking_df.to_csv(r"../data/HK_all_vs_parking.csv", index=False)
