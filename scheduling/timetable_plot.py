import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from datetime import datetime


def timetable_plot(trajectory, line_stop, line_name=None, saved_path=None):
    """
    assume no boarding time
    :param trajectory: first column is plate(班次) id, second column is stop name (matched with line_stop),
    third column is time (datetime format)
    :param line_stop: first column is stop name, second column is milestone (km)
    """
    pd.set_option('display.float_format', lambda x: '%.2f' % x)  # 不显示科学计数法

    trajectory.columns = ['plate', 'stop_name', 'time']
    time_list = trajectory['time'].to_list()
    datetime_list, second_list = [], []
    for i in range(len(time_list)):
        datetime_list.append(datetime.fromisoformat(time_list[i]))
    earliest_date = min(datetime_list)
    earliest_date = earliest_date.replace(hour=0, minute=0, second=0, microsecond=0)
    for i in range(len(time_list)):
        second_list.append((datetime_list[i] - earliest_date).total_seconds())
    second_list = list(map(int, second_list))
    trajectory['second'] = second_list
    trajectory = trajectory.sort_values(by=['plate', 'second'], ascending=[True, True])
    trajectory = trajectory.reset_index(drop=True)
    trajectory_name = trajectory[trajectory['plate'] == trajectory['plate'].iloc[0]]
    trajectory_name['seq'] = trajectory_name.index
    trajectory_name = trajectory_name[['stop_name','seq']]
    trajectory = pd.merge(trajectory, trajectory_name, on='stop_name', how='left')
    trajectory.drop(['stop_name'], axis=1, inplace=True)

    line_stop.columns = ['stop_name', 'milestone']
    line_stop = line_stop.sort_values(by='milestone').reset_index(drop=True)
    line_stop['seq'] = line_stop.index
    statname = line_stop['stop_name'].tolist()
    staty = line_stop['milestone'].tolist()

    trajectory = pd.merge(trajectory, line_stop, on='seq', how='left')

    st = 0
    et = 86400
    xsite = np.arange(st, et+600, 600).tolist()
    xlabel = [x // 3600 if x % 3600 == 0 else '' for x in xsite]

    plate = trajectory.groupby(['plate'])['time'].count().reset_index()

    fs = (11, 5)
    fig = plt.figure(figsize=fs)
    # plt.style.use('seaborn-whitegrid')
    plt.rcParams['font.sans-serif'] = ['Times New Roman']
    matplotlib.rcParams['font.sans-serif'] = ['SimSun']  # 宋体
    plt.yticks(staty, statname)
    plt.xticks(xsite, xlabel)
    plt.xlim(min(second_list), max(second_list))
    plt.ylim(0, max(staty))
    plt.grid()
    if line_name:
        plt.title(f'Timetable of Route {line_name}')

    for i in range(len(plate)):
        PGdata = trajectory[trajectory['plate'] == plate['plate'].loc[i]]
        plt.plot(PGdata['second'], PGdata['milestone'], "-", alpha=0.8, color='royalblue', label="test_zhexian")

    fig.tight_layout()
    if saved_path:
        plt.savefig(saved_path, dpi=300, bbox_inches='tight')
    plt.show(block=True)

if __name__ == '__main__':
    Gdata0 = pd.read_csv(r"E:\Manufacture\Python\hk-bus\data\interim\scheduling\test_complete_routes.csv")
    Gdata0 = Gdata0[['route_id', 'seq', 'eta']]
    station = pd.read_csv(r"E:\Manufacture\Python\hk-bus\data\interim\scheduling\test_line_stop_info.csv")
    path = r"C:\Users\TZL\Desktop\Figure_1.png"
    timetable_plot(Gdata0, station, '278K',path)