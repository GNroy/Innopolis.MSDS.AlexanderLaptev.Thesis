import pandas as pd
import datetime as dt
import pyproj
import numpy as np
import os

geod = pyproj.Geod(ellps='WGS84')

def to_datetime(string):
    return dt.datetime.strptime(string, '%Y-%m-%d %H:%M:%S')

def calculate_distance(long1, lat1, long2, lat2):
    if lat1 == lat2 and long1 == long2:
        return 0
    if False in np.isfinite([long1, long2, lat1, lat2]):
        return np.nan
    if lat1 < -90 or lat1 > 90 or lat2 < -90 or lat2 > 90:
        #raise ValueError('The range of latitudes seems to be invalid.')
        return np.nan
    if long1 < -180 or long1 > 180 or long2 < -180 or long2 > 180:
        return np.nan
        #raise ValueError('The range of longitudes seems to be invalid.')
    angle1,angle2,distance = geod.inv(long1, lat1, long2, lat2)
    return distance

def calculate_velocity(distance, timedelta):
    if timedelta.total_seconds() == 0: return np.nan
    return distance / timedelta.total_seconds()

def calculate_acceleration(velocity, velocity_next_position, timedelta):
    delta_v = velocity_next_position - velocity
    if timedelta.total_seconds() == 0: return np.nan
    return delta_v / timedelta.total_seconds()

headers_trajectory = ['lat', 'long', 'null', 'altitude','timestamp_float', 'date', 'time']

def load_trajectory_df(full_filename):
    subfolder = full_filename.split('/')[-3]
    trajectory_id = full_filename.split('/')[-1].split('.')[0]
    
    df = pd.read_csv(full_filename, skiprows = 6, header = None, names = headers_trajectory)
   
    df['datetime'] = df.apply(lambda z: to_datetime(z.date + ' ' + z.time), axis=1)
    df['datetime_next_position'] = df['datetime'].shift(-1)
    df['timedelta'] = df.apply(lambda z: z.datetime_next_position - z.datetime, axis=1)
    df = df.drop(['datetime_next_position'], axis=1)
    df = df.drop(['null', 'timestamp_float', 'date', 'time'], axis=1)
    
    
    df['long_next_position'] = df['long'].shift(-1)
    df['lat_next_position'] = df['lat'].shift(-1)
    df['distance'] = df.apply(lambda z: calculate_distance(z.long, z.lat, z.long_next_position, z.lat_next_position), axis=1)
    df = df.drop(['long_next_position', 'lat_next_position'], axis=1)
    
    df['velocity'] = df.apply(lambda z: calculate_velocity(z.distance, z.timedelta), axis=1)
    df['velocity_next_position'] = df['velocity'].shift(-1)
    df['acceleration'] = df.apply(lambda z: calculate_acceleration(z.velocity, z.velocity_next_position, z.timedelta), axis=1)
    df = df.drop(['velocity_next_position'], axis=1)
    
    df['trajectory_id'] = trajectory_id
    df['subfolder'] = subfolder
    df['labels'] = ''
    return df

def load_labels_df(filename):
    df = pd.read_csv(filename, sep='\t')
    df['start_time'] = df['Start Time'].apply(lambda x: dt.datetime.strptime(x, '%Y/%m/%d %H:%M:%S'))
    df['end_time'] = df['End Time'].apply(lambda x: dt.datetime.strptime(x, '%Y/%m/%d %H:%M:%S'))
    df['labels'] = df['Transportation Mode']
    df = df.drop(['End Time', 'Start Time', 'Transportation Mode'], axis=1)
    return df

LABELS_FILE = 'labels.txt'
MAIN_FOLDER = 'Data/'
TRAJ_FOLDER = 'Trajectory/'
OUTPUT_FOLDER = 'processed_data/'

if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)
directories = os.listdir(MAIN_FOLDER)

for subfolder in directories:
    list_df_traj = []
    subfolder_ = MAIN_FOLDER + subfolder + '/'
    traj_folder = MAIN_FOLDER + subfolder + '/' + TRAJ_FOLDER
    traj_files = os.listdir(traj_folder)
    
    traj_files_full_path = [traj_folder + traj_file for traj_file in traj_files]
    print(subfolder, len(traj_files_full_path))
    
    for file in traj_files_full_path:
        list_df_traj.append(load_trajectory_df(file))
    
    df_traj_all = pd.concat(list_df_traj)
    list_df_traj = []
    
    if LABELS_FILE in os.listdir(subfolder_):
        filename = subfolder_ + LABELS_FILE
        df_labels = load_labels_df(filename)
        for idx in df_labels.index.values:
            st = df_labels.ix[idx]['start_time']
            et = df_labels.ix[idx]['end_time']
            labels = df_labels.ix[idx]['labels']
            if labels:
                df_traj_all.loc[(df_traj_all['datetime'] >= st) & 
                                (df_traj_all['datetime'] <= et), 'labels'] = labels

    output_filename = OUTPUT_FOLDER + subfolder + '.csv'
    df_traj_all.to_csv(output_filename)
    del df_traj_all
