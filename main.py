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

sampling_interval = pd.Timedelta(minutes=5)
split_border = pd.Timedelta(days=1)
max_time_interval = df['datetime'].max()-df['datetime'].min()
pl = Platoon(2, 3, 2, max_time_interval // sampling_interval)
miner = Miner(df, pl, sampling_interval)
miner.unify_datetime(split_border)
miner.compute_timestamps(eps=1)

G = miner.graph()
nx.draw_networkx(G, with_labels=False, node_size=1)
plt.savefig("connections_graph_main.png", dpi=1000)
nx.write_graphml(G,'connections_graph_main.xml')