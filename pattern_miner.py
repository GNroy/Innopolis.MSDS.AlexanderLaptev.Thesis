import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from itertools import combinations
from scipy import sparse
from sklearn.cluster import DBSCAN, Birch
import networkx as nx
import datetime
import json
import re

from utills import Candidate, Pattern

class Miner:
    def __init__(self, df, pattern, delta):
        self._df = df.copy()
        self._pattern = pattern
        self._delta = delta
        self.is_unified = False
        self._sp_map = None
        self._candidate_stars = None
        self._pattern_set = None
        self._graph = None
        self._connection_rate = None
    
    def df(self):
        return self._df
    
    def pattern(self):
        return self._pattern
    
    def delta(self):
        return self._delta
    
    def staypoints_heatmap(self):
        if self._sp_map is None and not self.is_unified:
            self.extract_staypoints_heatmap()
        return self._sp_map
    
    def candidate_stars(self):
        if self._candidate_stars is None:
            self.compute_candidate_stars()
        return self._candidate_stars
    
    def pattern_set(self):
        if self._pattern_set is None:
            self.compute_pattern_set()
        return self._pattern_set
    
    def graph(self):
        if self._graph is None:
            self.compute_graph()
        return self._graph
    
    def connection_rate(self):
        if self._connection_rate is None:
            self.compute_connection_rate()
        return self._connection_rate
    
    # a shortcut for histogram2d call
    def _compute_heatmap(self, df, bins=50):
        return np.histogram2d(df['lat'], df['long'], bins=bins, range=[[self._df['lat'].min(), self._df['lat'].max()], [self._df['long'].min(), self._df['long'].max()]])[0]
    
    # get staypoints heatmap from raw trajectories (before upsampling)
    # usage after upsampling may cause an unpredictable result
    def extract_staypoints_heatmap(self, dist_thres=20, time_thres=pd.Timedelta(minutes=30), norm_ord=2, norm_coeff=1):
        def extract_staypoints_from_trajectory(traj, dist_thres, time_thres):
            t_mean = lambda x, y: x+(y-x)/2
            staypoint_list = []
            i = 0
            N = traj.shape[0]
            for j in range(i+1, N):
                dist = np.linalg.norm(traj.iloc[i][['lat', 'long']] - traj.iloc[j][['lat', 'long']])
                if dist>dist_thres or j==N-1:
                    if traj.iloc[j]['datetime']-traj.iloc[i]['datetime']>time_thres:
                        spatial_mean = traj.iloc[i:j][['lat', 'long']].mean(axis=0)
                        time = [traj.iloc[i]['datetime'] if i==0 else t_mean(traj.iloc[i-1]['datetime'], traj.iloc[i]['datetime']),
                               traj.iloc[j]['datetime'] if j==N-1 else t_mean(traj.iloc[j-1]['datetime'], traj.iloc[j]['datetime'])]
                        staypoint_list += [[spatial_mean[0], spatial_mean[1], time[0], time[1]]]
                    i = j
            return pd.DataFrame(staypoint_list, columns=['lat', 'long', 't_start', 't_end'])
        self._sp_map = self._compute_heatmap(self._df.groupby('trajectory_id').apply(extract_staypoints_from_trajectory, dist_thres=dist_thres, time_thres=time_thres))
        self._sp_map /= norm_coeff*np.linalg.norm(self._sp_map, ord=norm_ord) # think about proper norming
    
    # save staypoints heatmap from .npy file
    def save_staypoints_heatmap(self, filename):
        np.save(filename, self._sp_map)
    
    # load staypoints heatmap from .npy file
    def load_staypoints_heatmap(self, filename):
        self._sp_map = np.load(filename)
    
    # perform data unification procedure: splitting, unit time lenth casting and upsampling
    def unify_datetime(self, split_delta=None):
        def split_if_needed(s, delta):
            part_num = 0
            d_mask = np.array(s['datetime'].diff()>=delta)
            for i in range(len(s)):
                if d_mask[i]:
                    part_num += 1
                s['trajectory_id'].iloc[i] += '_' + str(part_num)
            return s
        def interpolate(df):
            datetimes = np.array([df['datetime'].iloc[0]+self._delta*j for j in range(1, round((df['datetime'].iloc[1]-df['datetime'].iloc[0])/self._delta))])
            lats = np.linspace(df['lat'].iloc[0], df['lat'].iloc[1], len(datetimes)+1)[1:]
            longs = np.linspace(df['long'].iloc[0], df['long'].iloc[1], len(datetimes)+1)[1:]
            trajectory_ids = np.full(len(datetimes), df['trajectory_id'].iloc[0])
            return pd.DataFrame({'lat':lats, 'long':longs, 'datetime':datetimes, 'trajectory_id':trajectory_ids})
        def upsample(df):
            list_df = []
            last_i = 0
            for i in range(1, len(df)):
                t_div = df['datetime'].iloc[i]-df['datetime'].iloc[i-1]
                if t_div!=self._delta and df['trajectory_id'].iloc[i]==df['trajectory_id'].iloc[i-1]:
                    list_df += [df[last_i:i], interpolate(df[i-1:i+1])]
                    last_i = i
            return pd.concat(list_df+[df[last_i:len(df)]], ignore_index=True)
        if self.is_unified:
            print('Already unified')
            return
        if self._sp_map is None:
            self.extract_staypoints_heatmap()
        self._df['datetime'] -= pd.to_timedelta(self._df['datetime'].view('int64') % self._delta.view('int64') / 10**9, unit='s')
        self._df = self._df.groupby(['trajectory_id','datetime']).mean().reset_index()[self._df.columns.tolist()]
        if split_delta:
            self._df = upsample(self._df.groupby('trajectory_id').apply(split_if_needed, delta=split_delta).reset_index(drop=True))
            self._df['trajectory_id'] = self._df['trajectory_id'].apply(lambda x: re.sub('_\d+$', '', x))
        else:
            self._df = upsample(self._df)
        self.is_unified = True
    
    # perform clusterization for every unique timestamp
    def compute_candidate_stars(self, eps=0.001, verbose=False):
        d = dict()
        def fill_dict(x):
            for c in combinations(x['trajectory_id'], 2):
                if c in d:
                    d[c] += [x['datetime'].iloc[0]]
                else:
                    d[c] = [x['datetime'].iloc[0]]
        self._candidate_stars = []
        def fill_stars(x, st):
            st += [x.apply(lambda xx: Candidate([x.name, xx['second_key']], xx['timestamps'], self._pattern, self._delta), axis=1).values]
        pattern_method = self._pattern.accepted_methods().get(self._pattern.method())
        if pattern_method == DBSCAN:
            cls = DBSCAN(eps, min_samples=2)
        elif pattern_method == Birch:
            cls = Birch(eps, n_clusters=None)
        else:
            raise NotImplementedError()
        for s_dt in sorted(self._df['datetime'].unique()):
            time_set = self._df.loc[self._df['datetime']==s_dt].copy()
            cls.fit(time_set[['lat', 'long']])
            u, c = np.unique(cls.labels_, return_counts=True)
            u = u[(c>1)&(u>=0)]
            if len(u) > 0:
                time_set['cluster'] = cls.labels_
                time_set = time_set[time_set['cluster'].isin(u)][['trajectory_id', 'cluster', 'datetime']]
                time_set.groupby('cluster').apply(fill_dict)
            if verbose:
                print('Time: ' + str(s_dt) + '\nEstimated number of clusters: %d\n' % len(u))
        key_parts = np.array(list(d.keys())).T
        pd.DataFrame({'timestamps': list(d.values()), 'first_key': key_parts[0], 'second_key': key_parts[1]}).groupby('first_key').apply(fill_stars, self._candidate_stars)
    
    # load candidate stars from .json file
    def load_candidate_stars(self, filename):
        class MyDecoder(json.JSONDecoder):
            def __init__(self, *args, **kwargs):
                json.JSONDecoder.__init__(self, object_hook=self.object_hook, *args, **kwargs)
            def object_hook(self, obj):
                if '__classname__' in obj:
                    if obj['__classname__']==Pattern.__name__:
                        #return Pattern(obj['_m'], obj['_k'], obj['_l'], pd.to_timedelta(obj['_g'], unit='ns'), obj['_method'])
                        return Pattern(obj['_m'], obj['_k'], obj['_l'], obj['_g'], obj['_method']) # FIX!!!
                    elif obj['__classname__']==Candidate.__name__:
                        return Candidate(obj['_objects'], obj['_timestamps'], obj['_pattern'], pd.to_timedelta(obj['_delta'], unit='ns'))
                    else:
                        raise ValueError('Unknown classname: %s' % obj['__classname__'])
                return obj
        with open(filename, 'r') as infile:
            self._candidate_stars = json.load(infile, cls=MyDecoder)
    
    # save candidate stars to .json file
    def save_candidate_stars(self, filename):
        class MyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, (datetime.datetime, datetime.date)):
                    return int(obj.astype(np.int64))
                elif isinstance(obj, datetime.timedelta):
                    return int(obj.to_timedelta64())
                elif isinstance(obj, Candidate):
                    o = obj.__dict__
                    o['__classname__'] = Candidate.__name__
                    return o
                elif isinstance(obj, Pattern):
                    o = obj.__dict__
                    o['__classname__'] = Pattern.__name__
                    key = '_accepted_methods' #kludge
                    if key in o:
                        del o[key]
                    return o
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                else:
                    return super(MyEncoder, self).default(obj)
        with open(filename, 'w') as outfile:
            json.dump(self._candidate_stars, outfile, cls=MyEncoder)
    
    # call apriori enumerator to obtain patterns from the stars
    # and transform patterns to more convenient form (group by pattern cardinality)
    def compute_pattern_set(self, max_card=np.inf, verbose=False, card_repr=True):
        def apriori_enumerator(star, max_level=np.inf):
            C = []
            for c in star:
                if c.sim():
                    C += [c]
            level = 3
            CS = []
            CR = list(C)
            while C:
                if level>max_level:
                    return CR
                for i in range(len(C)):
                    for j in range(i+1, len(C)):
                        cs = Candidate.merge(C[i], C[j])
                        if cs.obj_length() == level \
                                and cs.sim() and cs not in CS:
                            CS += [cs]
                C = CS
                CR += C
                CS = []
                level += 1
            return CR + C
        if self._candidate_stars is None:
            self.compute_candidate_stars()
        i = 0
        temp_patterns = []
        for star in self._candidate_stars:
            temp_patterns += [apriori_enumerator(star, max_card)]
            if not temp_patterns[-1]:
                i += 1
            elif verbose:
                print([[s.objects(), s.timestamps()] for s in temp_patterns[-1]])
                print()
        if verbose:
            print(str(i) + ' stars omitted as empty')
        if card_repr:
            self._pattern_set = []
            card = 2
            while card<=max_card:
                items_to_add = []
                for tp in temp_patterns:
                    for s in tp:
                        if s.obj_length() == card:
                            items_to_add += [s]
                if items_to_add:
                    self._pattern_set += [items_to_add]
                else:
                    break
                card += 1
        else:
            self._pattern_set = temp_patterns
    
    # get graph of connections between trajectory_ids or groups
    def compute_graph(self, cardinality=2):
        if self._pattern_set is None:
            self.compute_pattern_set()
        self._graph = nx.Graph()
        if cardinality == 2:
            for c in self._pattern_set[0]:
                self._graph.add_edge(c.objects()[0], c.objects()[-1], weight=len(c.timestamps()))
        else:
            raise NotImplementedError()
        return self._graph
    
    # get a connection rate matrix with labels legend
    def compute_connection_rate(self):
        if self._sp_map is None:
            self.extract_staypoints_heatmap()
        if self._pattern_set is None:
            self.compute_pattern_set()
        factor = pd.factorize(self._df['trajectory_id'], sort=True)
        labels = pd.Series(np.unique(factor[0]), index=factor[1])
        connection_rate_matrix = sparse.lil_matrix((labels.shape[0], labels.shape[0]), dtype=np.float)
        
        for p in self._pattern_set[0]:
            heatmap = self._compute_heatmap(self._df[(self._df['trajectory_id']==p.objects()[0]) & (self._df['datetime'].isin(p.timestamps()))])
            connection_rate_matrix[labels[p.objects()[0]], labels[p.objects()[1]]] = len(p.timestamps()) - np.sum(self._sp_map*heatmap)
        connection_rate_matrix = connection_rate_matrix.tocsr()
        self._connection_rate = {'matrix': connection_rate_matrix, 'labels': np.array(labels[1])}
    
    # save connection rate to .npz file
    def save_connection_rate(self, filename):
        np.savez(filename, labels=self._connection_rate['labels'], data=self._connection_rate['matrix'].data, indices=self._connection_rate['matrix'].indices, indptr=self._connection_rate['matrix'].indptr, shape=self._connection_rate['matrix'].shape)
    
    # load connection rate from .npz file
    def load_connection_rate(self, filename):
        loader = np.load(filename)
        self._connection_rate = {'matrix': sparse.csr_matrix((loader['data'], loader['indices'], loader['indptr']), shape=loader['shape']), 'labels': loader['labels']}