import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from test4mosa import Location
import data_utils
from simulation import SimVehicle
import pickle
from dateutil import parser
import pandas as pd
import geopandas as gpd
from pyproj import CRS
from test4plot2 import clearing
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import numpy as np


def basic_stat(obj_df,var_df0,cs_num):
    """
    df: which processed after function 'clearing'
    """
    var_df = var_df0.copy()
    for col in range(1,cs_num+1):
        mask = var_df.iloc[:, col] == 0  # 找出第 col 列为 0 的行
        target1 = col + cs_num  # 后推 cs_num 的列索引
        target2 = col + 2 * cs_num  # 后推 2*cs_num 的列索引
        if target1 < var_df.shape[1]:  # 确保索引不越界
            var_df.iloc[mask, target1] = 0
        if target2 < var_df.shape[1]:  # 确保索引不越界
            var_df.iloc[mask, target2] = 0

    stat_df = obj_df.copy()
    stat_df['bulit_num'] = var_df.iloc[:, 1:cs_num + 1].sum(axis=1)
    # 计算非零值的平均值
    stat_df['fast_num_avg'] = var_df.iloc[:, 1 + cs_num:2 * cs_num + 1].replace(0, np.nan).mean(axis=1)
    stat_df['slow_num_avg'] = var_df.iloc[:, 1 + 2 * cs_num:3 * cs_num + 1].replace(0, np.nan).mean(axis=1)
    # stat_df['fast_num_avg'] = var_df.iloc[:, 1 + cs_num:2 * cs_num + 1].sum(axis=1) / cs_num
    # stat_df['slow_num_avg'] = var_df.iloc[:, 1 + 2 * cs_num:3 * cs_num + 1].sum(axis=1) / cs_num

    stat_df['bus_type'] = var_df.iloc[:, -2]
    stat_df['bus_num'] = var_df.iloc[:, -1]

    return stat_df


# def find_knee(objs):
#     df = objs.copy()
#     scaler = MinMaxScaler()
#     df[['obj1', 'obj2', 'obj3']] = scaler.fit_transform(df[['obj1', 'obj2', 'obj3']])
#
#     min_values = df[['obj1', 'obj2', 'obj3']].min()
#
#     df['distance'] = ((df[['obj1', 'obj2', 'obj3']] - min_values) ** 2).sum(axis=1) ** 0.5
#     closest_row = df.loc[df['distance'].idxmin()]
#
#     # 返回该行的 'solution_no'
#     solution_no = closest_row['solution_no']
#     return solution_no


def find_knee(objs, minmax=False, **args):
    """
    minmax=True: 当目标函数标准化需要基于情景计算时
    """
    df = objs.copy()
    if minmax:
        cs_GDF = gpd.read_file(r'../../../hkbus/data/input/cs_gdf/cs_gdf.shp', crs=CRS.from_epsg(4326))  # 充电站
        vs_parking_df = pd.read_csv(r'../../../hkbus/data/HKsimpreparation/HK_all_vs_parking_nodeid.csv')  # 车辆行程
        vs_parking_df.reset_index(drop=True, inplace=True)
        vs_parking_df['s_time'] = vs_parking_df['s_time'].apply(parser.parse)
        vs_parking_df['e_time'] = vs_parking_df['e_time'].apply(parser.parse)
        cs_dict = {v: k for k, v in zip(cs_GDF.index, cs_GDF['node_id'])}
        num_vars = data_utils.set_var_num(cs_GDF)

        grade = args.get('is_grade')
        if grade:
            vs_parking_df = vs_parking_df.drop(columns=['distance'])
            vs_parking_df = vs_parking_df.rename(columns={'evrange': 'distance'})
            model_name = 'Grade'
        else:
            model_name = 'Baseline'
        with open(rf"../../../hkbus/data/input/formosa/all_d2s_dict_{model_name}.pkl", 'rb') as f:
            all_d2s_dict = pickle.load(f)
        sim_v_info = pd.DataFrame.from_dict(vs_parking_df.groupby('v_name').  # 将车辆行程按车辆集计
                                            apply(lambda x: data_utils.derive_simV_info(x)).to_dict(), orient='index')



        least_solu_vars_slice = least_station(cs_GDF,max_dis=2500)
        MOST_LIST1 = []
        for i in range(3):
            for j in range(2):
                # least_solu_vars = least_solu_vars_slice + [3] * (len(cs_GDF) * 2) + [2,0] + [0,0]
                least_solu_vars = least_solu_vars_slice + [3] * (len(cs_GDF) * 2) + [i,j] + [0,0] # choose vehicle type with the biggest consumption
                least_solu_vars = np.array(least_solu_vars)
                sim_v_dict = {}
                for idx, row in sim_v_info.iterrows():
                    sim_v_dict[idx] = SimVehicle(idx, row.trip, row.s_time, row.e_time, row.destination, row.distance,
                                                 row.avg_velocity)
                problem = Location(num_vars=num_vars, sim_v_info=sim_v_info, cs_gdf=cs_GDF, vs_parking_df=vs_parking_df,
                                   all_d2s_dict=all_d2s_dict, cs_dict=cs_dict, is_grade=grade, sim_v_dict=sim_v_dict)
                OBJ1, CV1 = problem.eval_vars(least_solu_vars, is_test=False)
                OBJ1 = OBJ1 * (-1)
                OBJ1[0] = OBJ1[0]/1000000
                OBJ1[1] = OBJ1[1]/ 1000000
                OBJ1[2] = OBJ1[2]/365/len(vs_parking_df)*60
                MOST_LIST1.append(OBJ1)
        MOST_DF1 = pd.DataFrame(MOST_LIST1)
        min_1 = MOST_DF1[0].min()
        max_2 = MOST_DF1[1].max()
        max_3 = MOST_DF1[2].max()
        # most_solu_vars = [1]*len(cs_GDF)+[25]*len(cs_GDF)*2+[1,1]+[2000,2000]
        # most_solu_vars = np.array(most_solu_vars)
        MOST_LIST2 = []
        for i in range(3):
            for j in range(2):
                most_solu_vars = [1] * len(cs_GDF) + [25] * len(cs_GDF) * 2 + [i,j] + [2000, 2000]
                most_solu_vars = np.array(most_solu_vars)
                sim_v_dict = {}
                for idx, row in sim_v_info.iterrows():
                    sim_v_dict[idx] = SimVehicle(idx, row.trip, row.s_time, row.e_time, row.destination, row.distance,
                                                 row.avg_velocity)
                problem = Location(num_vars=num_vars, sim_v_info=sim_v_info, cs_gdf=cs_GDF, vs_parking_df=vs_parking_df,
                                   all_d2s_dict=all_d2s_dict, cs_dict=cs_dict, is_grade=grade, sim_v_dict=sim_v_dict)
                OBJ2, CV2 = problem.eval_vars(most_solu_vars, is_test=False)
                OBJ2 = OBJ2 * (-1)
                OBJ2[0] = OBJ2[0] / 1000000
                OBJ2[1] = OBJ2[1] / 1000000
                OBJ2[2] = OBJ2[2] / 365 / len(vs_parking_df) * 60
                MOST_LIST2.append(OBJ2)
        MOST_DF2 = pd.DataFrame(MOST_LIST2)
        max_1 = MOST_DF2[0].max()
        min_2 = MOST_DF2[1].min()
        min_3 = MOST_DF2[2].min()
        print(f"Max: {max_1} {max_2} {max_3}")
        print(f"Min: {min_1} {min_2} {min_3}")
        max_3 = 10

        df['obj1']=(df['obj1']-min_1)/(max_1-min_1)
        df['obj2'] = (df['obj2'] - min_2) / (max_2 - min_2)
        df['obj3'] = (df['obj3'] - min_3) / (max_3 - min_3)
        min_values = 0
    else:
        scaler = MinMaxScaler()
        df[['obj1', 'obj2', 'obj3']] = scaler.fit_transform(df[['obj1', 'obj2', 'obj3']])
        min_values = df[['obj1', 'obj2', 'obj3']].min()

    df['distance'] = ((df[['obj1', 'obj2', 'obj3']] - min_values) ** 2).sum(axis=1) ** 0.5
    closest_row = df.loc[df['distance'].idxmin()]

    # 返回该行的 'solution_no'
    solution_no = closest_row['solution_no']
    return solution_no


def least_station(gdf, max_dis=5000, plot=False):
    if gdf.crs != 'EPSG:2326':
        gdf = gdf.to_crs('EPSG:2326')
    sindex = gdf.sindex

    # 4. 计算每个站点的邻居（距离小于等于5公里的站点）
    neighbors = []
    for i in range(len(gdf)):
        point = gdf.geometry.iloc[i]
        # 使用空间索引查询距离小于等于5000米（5公里）的站点
        buffer = point.buffer(max_dis)  # 创建 5 公里缓冲区
        nearby_indices = list(sindex.query(buffer, predicate='intersects'))
        neighbors.append(nearby_indices)

    # 5. 贪心算法选择最少的充电站
    uncovered = set(range(len(gdf)))  # 未覆盖的站点集合
    selected = []  # 选中的充电站索引

    while uncovered:
        max_cover = 0
        best_station = None
        # 遍历所有未选中的站点，找到覆盖最多未覆盖站点的候选站点
        for i in range(len(gdf)):
            if i not in selected:
                cover = set(neighbors[i]) & uncovered  # 该站点能覆盖的未覆盖站点
                if len(cover) > max_cover:
                    max_cover = len(cover)
                    best_station = i
        if best_station is not None:
            selected.append(best_station)
            uncovered -= set(neighbors[best_station])  # 移除已覆盖的站点
        else:
            break  # 如果没有站点可选，退出循环（理论上不会发生，因为每个站点至少覆盖自己）
    charging_list = [1 if i in selected else 0 for i in range(len(gdf))]
    if plot:
        # 6. 生成充电站分布图
        gdf['is_charging'] = 0  # 添加列，默认值为0
        for idx in selected:
            gdf.loc[idx, 'is_charging'] = 1  # 选中的站点标记为1

        # 绘制分布图
        fig, ax = plt.subplots(figsize=(10, 10))
        gdf.plot(column='is_charging', cmap='cool', legend=True, ax=ax)
        ax.set_title('Charging Stations Distribution in Hong Kong')
        plt.show(block=True)
    return charging_list

if __name__ == '__main__':
    solutions_folder = rf"../../data/egMianyang/MOSA/TestingResults0304"
    cs_gdf = gpd.read_file(r'../../data/egMianyang/road/cs_gdf.shp', crs=CRS.from_epsg(4326))  # 充电站
    CS_NUM = len(cs_gdf)

    """module 1: convert solutions to shp"""
    # # # input
    # solution_name = 'SA10552M2'
    #
    # solutions_data = pd.read_csv(solutions_folder+r'/archive_vars.csv')
    # first_column = solutions_data.iloc[:, 0]
    # row_indices = first_column[first_column == solution_name].index.tolist()[0]
    #
    # cs_gdf['if_built'] = solutions_data.iloc[row_indices, 1:cs_num+1].tolist()
    # cs_gdf['fast_num'] = solutions_data.iloc[row_indices, 1+cs_num:2*cs_num+1].tolist()
    # cs_gdf['slow_num'] = solutions_data.iloc[row_indices, 1+2*cs_num:3*cs_num+1].tolist()
    #
    # # # output
    # cs_gdf.to_file(f"{solutions_folder}/cs_{solution_name}.shp", driver='ESRI Shapefile')


    """module 2: summary report"""
    objs = pd.read_csv(solutions_folder+r'/archive_objs.csv')
    vars = pd.read_csv(solutions_folder+r'/archive_vars.csv')
    objs = clearing(objs)

    stat = basic_stat(objs, vars, CS_NUM)

    # knee_point = find_knee(objs, minmax=True, is_grade=True)



    # objs.to_csv(f"{solutions_folder}/summary_report.csv")



