import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN
from HACluster import Clusterer
import networkx as nx
import datetime
from utills import Candidate, Pattern
class Miner:
    def __init__(self, df, pattern, delta):
        self._df = df
        self._pattern = pattern
        self._delta = delta
        self._timestamps = None
        self._candidate_stars = None
        self._pattern_set = None
        self._graph = None
    
    def df(self):
        return self._df
    
    def pattern(self):
        return self._pattern
    
    def delta(self):
        return self._delta
    
    def timestamps(self):
        if self._timestamps is None:
            self.compute_timestamps()
        return self._timestamps
    
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
    
    # perform clusterization for every unique timestamp
    def compute_timestamps(self, eps=0.001, verbose=False):
        if self._pattern.accepted_methods().get(self._pattern.method()) == DBSCAN:
            list_df = []
            db = DBSCAN(eps=eps, min_samples=2)
            for s_dt in sorted(self._df['datetime'].unique()):
                time_set = self._df.loc[self._df['datetime']==s_dt].copy()
                db.fit(time_set[['lat', 'long']])
                n_clusters_ = len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)
                if n_clusters_ > 0:
                    time_set['cluster'] = pd.Series(db.labels_, index=time_set.index)
                    time_set = time_set.loc[time_set['cluster']>=0]
                    list_df += [time_set[['trajectory_id', 'cluster', 'datetime']]]
                if verbose:
                    print('Time: ' + str(s_dt) + '\nEstimated number of clusters: %d\n' % n_clusters_)
            self._timestamps = pd.concat(list_df, ignore_index=True)
        else:
            raise NotImplementedError()
    
    def load_timestamps(self, filename):
        self._timestamps = pd.read_csv(filename, parse_dates=[2], dtype={'cluster': np.int, 'trajectory_id': np.str_})
    
    # get the initial candidate stars
    def compute_candidate_stars(self, verbose=False):
        if self._timestamps is None:
            self.compute_timestamps()
        counter = 0
        self._candidate_stars = []
        for label in np.sort(self._timestamps['trajectory_id'].unique()):
            label_df = self._timestamps[self._timestamps['trajectory_id']==label]
            tuple_list = list(zip(label_df['datetime'], label_df['cluster']))
            candidates_df = self._timestamps[(self._timestamps['trajectory_id']>label) & (self._timestamps['datetime'].isin(label_df['datetime'])) & (self._timestamps['cluster'].isin(label_df['cluster']))].groupby(['datetime', 'cluster']).filter(lambda x: x.name in tuple_list)[['trajectory_id', 'datetime']]
            star = []
            for other_label in candidates_df['trajectory_id'].unique():
                star += [Candidate([label, other_label], candidates_df['datetime'][candidates_df['trajectory_id']==other_label].values, self._pattern, self._delta)]
            self._candidate_stars += [star]
            counter += 1
            if verbose:
                print(counter, label)
    
    # call apriori enumerator to obtain patterns from the stars
    # and transform patterns to more convenient form (group by pattern cardinality)
    def compute_pattern_set(self, verbose=False):
        def apriori_enumerator(star):
            C = []
            for c in star:
                if c.sim():
                    C += [c]
            level = 3
            CS = []
            CR = list(C)
            while C:
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
            temp_patterns += [apriori_enumerator(star)]
            if not temp_patterns[-1]:
                i += 1
            elif verbose:
                print([[s.objects(), s.timestamps()] for s in temp_patterns[-1]])
                print()
        if verbose:
            print(str(i) + ' stars omitted as empty')
        self._pattern_set = []
        card = 2
        while True:
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