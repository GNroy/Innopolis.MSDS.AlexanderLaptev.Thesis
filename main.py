import os
import re
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from utills import Candidate, Platoon
from pattern_miner import Miner

def get_trajectory_id(text):
    m = re.search('client_(.+).csv', text)
    if m:
        found = m.group(1)
        return found
    else:
        raise ValueError()

TRAJ_FOLDER = 'paths'
columns = ['lat', 'long', 'datetime', 'trajectory_id']
FILE_NAME = os.path.join(TRAJ_FOLDER, 'processed.csv')

if not os.path.exists(FILE_NAME):
    if not os.path.exists(TRAJ_FOLDER):
        raise ValueError(TRAJ_FOLDER + ' does not exist')
    folder_files = os.listdir(TRAJ_FOLDER)
    list_df = []
    for filename in folder_files:
        df = pd.read_csv(os.path.join(TRAJ_FOLDER, filename), names=columns)
        df['datetime'] = pd.to_datetime(df['datetime'], unit='s')
        df['trajectory_id'] = get_trajectory_id(filename)
        list_df += [df]
    df = pd.concat(list_df, ignore_index=True)
    df.to_csv(FILE_NAME, index=False)

df = pd.read_csv(FILE_NAME, parse_dates=[columns.index('datetime')], dtype={'lat': np.float32, 'long': np.float32, 'trajectory_id': np.str_})

def unify_datetime(df, delta, split_delta=None):
    def split_if_needed(s, t, delta):
        res = pd.Series.copy(s)
        part_num = 0
        res.iloc[0] += '_' + str(part_num)
        for i in range(1, len(s)):
            if t.iloc[i]-t.iloc[i-1]>=delta:
                part_num += 1
            res.iloc[i] += '_' + str(part_num)
        return(res)
    def interpolate(df, delta):
        datetimes = np.array([df['datetime'].iloc[0]+delta*j for j in range(1, round((df['datetime'].iloc[1]-df['datetime'].iloc[0])/delta))])
        lats = np.linspace(df['lat'].iloc[0], df['lat'].iloc[1], len(datetimes)+1)[1:]
        longs = np.linspace(df['long'].iloc[0], df['long'].iloc[1], len(datetimes)+1)[1:]
        trajectory_ids = np.full(len(datetimes), df['trajectory_id'].iloc[0])
        return pd.DataFrame({'lat':lats, 'long':longs, 'datetime':datetimes, 'trajectory_id':trajectory_ids})
    def upsample(df, delta):
        list_df = []
        last_i = 0
        for i in range(1, len(df)):
            t_div = df['datetime'].iloc[i]-df['datetime'].iloc[i-1]
            if t_div!=delta and df['trajectory_id'].iloc[i]==df['trajectory_id'].iloc[i-1]:
                list_df += [df[last_i:i], interpolate(df[i-1:i+1], delta)]
                last_i = i
        return pd.concat(list_df+[df[last_i:len(df)]], ignore_index=True)
    
    new_df = pd.DataFrame.copy(df)
    new_df['datetime'] -= pd.to_timedelta(new_df['datetime'].view('int64') % delta.view('int64') / 10**9, unit='s')
    new_df = new_df.groupby(['trajectory_id','datetime']).mean().reset_index()[new_df.columns.tolist()]
    if split_delta:
        new_df['trajectory_id'] = new_df.groupby('trajectory_id')['trajectory_id'].apply(split_if_needed, t = new_df['datetime'], delta=split_delta)
    return(upsample(new_df, delta))

sampling_interval = pd.Timedelta(minutes=5)
split_border = pd.Timedelta(days=1)
df = unify_datetime(df, sampling_interval, split_border)
max_time_interval = df['datetime'].max()-df['datetime'].min()
pl = Platoon(2, 3, 2, max_time_interval // sampling_interval)
miner = Miner(df, pl, sampling_interval)
miner.compute_timestamps(eps=1)

G = miner.graph()
nx.draw_networkx(G, with_labels=False, node_size=1)
plt.savefig("connections_graph_main.png", dpi=1000)
nx.write_graphml(G,'connections_graph_main.xml')