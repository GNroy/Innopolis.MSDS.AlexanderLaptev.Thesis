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
        if not self._timestamps:
            self.compute_timestamps()
        return self._timestamps
    
    def candidate_stars(self):
        if not self._candidate_stars:
            self.compute_candidate_stars()
        return self._candidate_stars
    
    def pattern_set(self):
        if not self._pattern_set:
            self.compute_pattern_set()
        return self._pattern_set
    
    def graph(self):
        if not self._graph:
            self.compute_graph()
        return self._graph
    
    # perform clusterization for every unique timestamp
    def compute_timestamps(self, eps=0.001, verbose=False):
        self._timestamps = {}
        if self._pattern.accepted_methods().get(self._pattern.method()) == DBSCAN:
            db = DBSCAN(eps=eps, min_samples=2)
            for s_dt in sorted(self._df['datetime'].unique()):
                time_set = self._df[['lat', 'long', 'trajectory_id']].loc[self._df['datetime']==s_dt]
                db.fit(time_set[['lat', 'long']])
                n_clusters_ = len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)
                if n_clusters_ > 0:
                    time_set['cluster'] = pd.Series(db.labels_, index=time_set.index)
                    time_set = time_set.loc[time_set['cluster']>=0]
                    time_set.reset_index(drop=True, inplace=True)
                    self._timestamps[s_dt] = time_set[['trajectory_id', 'cluster']]
                if verbose:
                    print('Time: ' + str(s_dt) + '\nEstimated number of clusters: %d\n' % n_clusters_)
        else:
            raise NotImplementedError()
    
    # get the initial candidate stars
    def compute_candidate_stars(self):
        if not self._timestamps:
            self.compute_timestamps()
        # get multigraph from timestamps
        def get_graphs(dfs):
            def add_edges_from_cluster(G, values, timestamp):
                for i in range(len(values)):
                    for j in range(i+1, len(values)):
                        G.add_edge(values[i], values[j], weight=timestamp)
            G = nx.MultiGraph()
            for df_key in dfs.keys():
                inner_df = dfs[df_key]
                for cl in inner_df['cluster'].unique():
                    add_edges_from_cluster(G, inner_df['trajectory_id'].loc[inner_df['cluster']==cl].values, df_key)
            return G
        # merge multigraph to graph
        def merge_multigraph_to_graph(M):
            G = nx.Graph()
            for u, v, data in M.edges_iter(data=True):
                if 'weight' in data:
                    w = data['weight']
                else:
                     raise ValueError('Invalid multigraph: one of the edges has no weight assigned.')
                if G.has_edge(u,v):
                    G[u][v]['weight'] += [w]
                else:
                    G.add_edge(u, v, weight=[w])
            return G
        G = merge_multigraph_to_graph(get_graphs(self._timestamps))
        self._candidate_stars = []
        nodes = G.nodes()
        for n in nodes:
            star = []
            for edge in G.edges(n):
                objs = list(edge)
                star += [Candidate(objs, G.get_edge_data(objs[0], objs[-1])['weight'], self._pattern, self._delta)]
            self._candidate_stars += [star]
            G.remove_node(n)
        return self._candidate_stars
    
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
        if not self._candidate_stars:
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
        if not self._pattern_set:
            self.compute_pattern_set()
        self._graph = nx.Graph()
        if cardinality == 2:
            for c in self._pattern_set[0]:
                self._graph.add_edge(c.objects()[0], c.objects()[-1], weight=len(c.timestamps()))
        else:
            raise NotImplementedError()
        return self._graph