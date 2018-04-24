from sklearn.cluster import DBSCAN, Birch
from pandas import Timedelta, to_timedelta
import numpy as np
class Pattern:
    _accepted_methods = {'density' : DBSCAN, 'disk' : Birch}
    
    def __param_check(self, *args):
        for arg in args:
            if isinstance(arg, int):
                if arg < 1:
                    raise ValueError('Integer coefficient < 1: %d' % arg)
            elif isinstance(arg, str):
                if not arg in self._accepted_methods:
                    raise ValueError('Not accepted method: %s' % str(arg))
            elif not isinstance(arg,Timedelta):
                raise ValueError('Wrong argiment type: %s' % str(type(arg)))
                
    
    def __init__(self, m, k, l, g, method):
        self.__param_check(m, k, l, g, method)
        self._m = m
        self._k = k
        self._l = l
        self._g = g
        self._method = method
    
    def m(self):
        return self._m
    
    def k(self):
        return self._k
    
    def l(self):
        return self._l
    
    def g(self):
        return self._g
    
    def method(self):
        return self._method
    
    def accepted_methods(self):
        return self._accepted_methods
class Group(Pattern):
    def __init__(self, g):
        super().__init__(2, 1, 2, g, 'disk')
        self._accepted_methods = {'disk' : self._accepted_methods['disk']}
class Flock(Pattern):
    def __init__(self, m, k):
        super().__init__(m, k, k, 1, 'disk')
        self._accepted_methods = {'disk' : self._accepted_methods['disk']}
class Convoy(Pattern):
    def __init__(self, m, k):
        super().__init__(m, k, k, 1, 'density')
        self._accepted_methods = {'density' : self._accepted_methods['density']}
class Swarm(Pattern):
    def __init__(self, m, k, g):
        super().__init__(m, k, 1, g, 'density')
        self._accepted_methods = {'density' : self._accepted_methods['density']}
class Platoon(Pattern):
    def __init__(self, m, k, l, g):
        super().__init__(m, k, l, g, 'density')
        self._accepted_methods = {'density' : self._accepted_methods['density']}
class Candidate:
    def __param_check(self, objects, timestamps, pattern, delta):
        if not isinstance(objects, list):
            raise ValueError(objects)
        if not isinstance(timestamps, (list, np.ndarray)):
            raise ValueError(timestamps)
        if not isinstance(pattern, Pattern):
            raise ValueError(pattern)
        if not isinstance(delta, Timedelta):
            raise ValueError(delta)
    
    def __init__(self, objects, timestamps, pattern, delta):
        self.__param_check(objects, timestamps, pattern, delta)
        self._objects = objects
#        if timestamps==[]:
#            self._timestamps = timestamps
#        elif isinstance(timestamps[0], int):
#            self._timestamps = timestamps
        if timestamps==[] or isinstance(timestamps[0], int):
            self._timestamps = timestamps
        else:
            self._timestamps = timestamps.astype(np.int64).tolist() if isinstance(timestamps, np.ndarray) else [int(a.to_datetime64()) for a in  timestamps]
        self._pattern = pattern
        self._delta = int(delta.to_timedelta64())
    
    def __str__(self):
        return 'objects: ' + str(self._objects) + '\ntimestamps: ' + str(self._timestamps)
    
    def __eq__(self, other):
        return set(self._objects) == set(other._objects) and set(self._timestamps) == set(other._timestamps)
    
    def __ne__(self, other):
        return set(self._objects) != set(other._objects) or set(self._timestamps) != set(other._timestamps)
    
    def objects(self):
        return self._objects
    
    def timestamps(self):
        return self._timestamps
    
    def obj_length(self):
        return len(self._objects)
    
    def tst_length(self):
        return len(self._timestamps)
    
    def pattern(self):
        return self._pattern
    
    def delta(self, raw=False):
        return self._delta if raw else to_timedelta(self._delta, unit='ns')
    
    @staticmethod
    def _intersection(*args):
        #f = Candidate.__flatten(list(args), 1)
        # efficiency boost but kludge
        # use carefully, or comment this and uncomment line above
        f = list(args) if len(args) == 2 else Candidate.__flatten(list(args), 1)
        s = set(f[0])
        for i in range(1,len(f)):
            s = s.intersection(f[i])
        return list(s)
    
    @staticmethod
    def _union(*args):
        #f = Candidate.__flatten(list(args))
        # efficiency boost but kludge
        # use carefully, or comment this and uncomment line above
        f = args[0] + args[1] if len(args) == 2 else Candidate.__flatten(list(args))
        return list(set().union(f))
    
    @staticmethod
    def __flatten(S, depth=0):
        if not S:
            return S
        if Candidate.__d_ch(S, depth):
            return Candidate.__flatten(S[0], depth) + Candidate.__flatten(S[1:], depth)
        return S[:1] + Candidate.__flatten(S[1:], depth)
    
    @staticmethod
    def __d_ch(S, depth):
        if len(S) == 0 or S == []:
            return False
        elif not (isinstance(S[0], list) or isinstance(S[0], np.ndarray)):
            return False
        if depth == 0:
            return True
        return Candidate.__d_ch(S[0], depth-1)
    
    @staticmethod
    def merge(a, *args):
        if isinstance(a, list):
            return Candidate(Candidate._union([o.objects() for o in a]), Candidate._intersection([t.timestamps() for t in a]), a[0].pattern(), a[0].delta())
        if len(args) == 1:
            b = list(args)[0]
            if isinstance(a, Candidate) and isinstance(b, Candidate):
                return Candidate(Candidate._union(a.objects(), b.objects()), Candidate._intersection(a.timestamps(), b.timestamps()), a.pattern(), a.delta())
        raise ValueError(a, args)
    
    def sim(self):
        if self.tst_length() < self._pattern.k():
            return False
        # remove the unqualified consecutive parts of timestamps
        con_start = 0
        ts = np.array(self._timestamps)
        for i in range(1, len(ts)):
            if ts[i] - ts[i-1] != self._delta:
                if i - con_start < self._pattern.l():
                    for j in range(con_start, i):
                        self._timestamps.remove(ts[j])
                con_start = i
        if len(ts) - con_start < self._pattern.l():
            for j in range(con_start, len(ts)):
                self._timestamps.remove(ts[j])
        # remove the L-G-L anomolies
        if self._timestamps == []:
            return False
        con_start = 0
        ts = np.array(self._timestamps)
        if ts[-1] - ts[0] > self._delta * self._pattern.g():
            current_sum = 1
            for i in range(1, len(ts)):
                if ts[i] - ts[i-1] > self._delta * self._pattern.g():
                    if current_sum < self._pattern.k():
                        for j in range(con_start, i):
                            self._timestamps.remove(ts[j])
                    con_start = i
                    current_sum = 1
                else:
                    current_sum += 1
        if len(ts) - con_start < self._pattern.k():
            for j in range(con_start, len(ts)):
                self._timestamps.remove(ts[j])
        return self.tst_length() >= self._pattern.k()
