import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import datetime
from utills import Platoon
from pattern_miner import Miner
INPUT_FOLDER = 'examples/geolife-trajectories/processed_data/'
FILE_NAME = INPUT_FOLDER + 'BeijingWalkingAreas.csv'
if not os.path.exists(FILE_NAME):
    list_df = []
    for file in os.listdir(INPUT_FOLDER):
        df = pd.read_csv(INPUT_FOLDER + file)
        list_df.append(df)
    df = pd.concat(list_df, ignore_index=True)
    df.drop('Unnamed: 0', inplace=True, axis=1)
    df['datetime'] = pd.to_datetime(df['datetime'], errors='ignore')
    # Beijing
    latMinB = 39.64
    latMaxB = 40.51
    longMinB = 115.76
    longMaxB = 116.88
    df = df[(df.lat.between(latMinB, latMaxB)) & (df.long.between(longMinB, longMaxB))]
    df = df[df['labels'] == 'walk']
    df = df[['lat', 'long', 'datetime', 'trajectory_id']]
    df.reset_index(drop=True, inplace=True)
    df.to_csv(FILE_NAME, index=False)
df = pd.read_csv(FILE_NAME)
if 'datetime' in df.columns:
    df['datetime'] = pd.to_datetime(df['datetime'])
# drop short trajectories
traj_len = 10
long_traj = df['trajectory_id'].value_counts()
long_traj = long_traj.where(long_traj.values > traj_len).dropna().keys()
df = df.loc[df['trajectory_id'].isin(long_traj)]
# compress data to have unit time intervals
# for now I've decided to use one minute interval
one_minute_interval = datetime.timedelta(minutes=1)
def compress(df, verbose=False):
    list_df = []
    iter_count = 0
    if verbose:
        print('Amount of iterations needed: ' + str(len(df['trajectory_id'].unique())))
    for t_id in df['trajectory_id'].unique():
        iter_count += 1
        single_trajectory = df.loc[df['trajectory_id']==t_id]
        single_trajectory['datetime'] = pd.to_datetime(single_trajectory['datetime'].values.astype('<M8[m]'))
        list_single_df = []
        for st_dt in single_trajectory['datetime'].unique():
            avg_values = single_trajectory[['lat', 'long']].loc[single_trajectory['datetime'].values==st_dt].mean().values.tolist() + [st_dt, t_id]
            list_single_df.append(pd.DataFrame(np.reshape(avg_values, (1, len(avg_values))), columns=df.columns.values))
        list_df.append(pd.concat(list_single_df, ignore_index=True))
        if verbose:
            print('Iteration ' + str(iter_count) + ': Added ' + str(list_df[-1].count()[0]) + ' rows\n')
    return pd.concat(list_df, ignore_index=True)
df_compressed = compress(df)
df_compressed['datetime'] = pd.DatetimeIndex(df_compressed['datetime'].values)
# let's treat datetime column as only the time, i.e. as if all trajectories are recorded in the same day
df_compressed['datetime'] = pd.Series([datetime.datetime.combine(datetime.date.min, val.time()) for val in df_compressed['datetime']])
time_interval = df_compressed['datetime'].max() - df_compressed['datetime'].min()
pl = Platoon(2, 3, 2, time_interval)
miner = Miner(df_compressed, pl, one_minute_interval)
G = miner.graph()
nx.draw_networkx(G, with_labels=False, node_size=1)
plt.savefig("connections_graph.png", dpi=1000)
nx.write_graphml(G,'connections_graph.xml')