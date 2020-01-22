''' author: samtenka
    change: 2020-01-18
    create: 2019-06-17
    descrp: interface for optimization results
'''

from collections import namedtuple
import numpy as np
from utils import CC
import torch



#=============================================================================#
#           0.                                                                #
#=============================================================================#

OptimKey = namedtuple('OptimKey',
    ('kind', 'metric', 'evalset', 'sampler', 'eta', 'T', 'N')
) 
# TODO: add batches and separate objective from interepoch shuffling

class OptimLog(object):
    '''
        Optimization log: record results of many optimization trials with some
        shared and some different optimization parameters. 
    '''
    def __init__(self, buffer_len=1000):
        '''
        '''
        self.buffer = {}
        self.summary = {}
        self.buffer_len = buffer_len

    def flush(self, okey):
        vals = np.array(self.buffer[okey])
        if not len(vals): return
        if okey not in self.summary:
            self.summary[okey] = {'mean':0.0, 'stdv':0.0, 'nb_samples':0} 
        d = self.summary[okey]
        madd, sadd, Nadd = np.mean(vals), np.std(vals), len(vals) 
        mold, sold, Nold = d['mean'], d['stdv'], d['nb_samples']

        Nnew = Nold + Nadd 
        pold = Nold / float(Nnew)
        padd = Nadd / float(Nnew)

        mnew = pold*mold + padd*madd
        snew = np.sqrt(
            pold*(sold**2 + mold**2) + padd*(sadd**2 + madd**2)
            - mnew**2
        ) 

        d['mean'] = mnew
        d['stdv'] = snew
        d['nb_samples'] = Nnew

        self.buffer[okey] = []

    def accum(self, okey, value):
        '''
        '''
        if okey not in self.buffer:
            self.buffer[okey] = []
        self.buffer[okey].append(value)
        #if len(self.buffer[okey]) >= self.buffer_len:
        #    self.flush(name)

    def absorb_buffer(self, rhs):
        '''
        '''
        for okey, values in rhs.buffer.items():
            if okey not in self.buffer:
                self.buffer[okey] = []
            self.buffer[okey] += values

    def compute_diffs(self):
        '''
        '''
        hamming = lambda x, y: sum((1 if xx!=yy else 0) for xx,yy in zip(x, y))
        shrink = lambda x, y: ((x,y) if x!=y else x) 

        diffs = {}
        for okey_base, value_base in self.buffer.items():
            if okey_base.kind=='diff': continue
            for okey_comp, value_comp in self.buffer.items():
                if okey_comp.kind=='diff': continue
                if hamming(okey_base, okey_comp) not in [1, 2]:  continue
                if okey_base.metric != okey_comp.metric: continue
                if len(value_base) != len(value_comp): continue

                value_diff = np.array(value_comp) - np.array(value_base)
                okey_diff = OptimKey(
                    kind    = 'diff',
                    sampler = shrink(okey_comp.sampler, okey_base.sampler),
                    eta     = shrink(okey_comp.eta,     okey_base.eta),
                    T       = shrink(okey_comp.T,       okey_base.T),
                    N       = shrink(okey_comp.N,       okey_base.N),
                    evalset = shrink(okey_comp.evalset, okey_base.evalset),
                    metric  = shrink(okey_comp.metric,  okey_base.metric),
                )  
                diffs[okey_diff] = value_diff
        for k in diffs:
            self.buffer[k] = diffs[k]

    def __str__(self):
        '''
        '''
        self.compute_diffs()
        for name in self.buffer:
            self.flush(name)
        lines = [
            '    {}: \t {{ "mean":{}, "stdv":{}, "nb_samples":{} }}'.format(
                okey, values['mean'], values['stdv'], values['nb_samples']
            )
            for okey, values in self.summary.items()
        ]
        return '{\n'+',\n'.join(sorted(lines, reverse=True))+'\n}'

    def load_from(self, file_nm):
        with open(file_nm) as f:
            self.summary = eval(f.read())

    def query_eta_curve(self, kind='main', metric='loss', evalset='test',
                        sampler='sgd', T=None, N=None):
        '''
        '''
        X, Y, S = [], [], []
        for okey in self.summary:
            if okey.kind != kind: continue
            if okey.metric != metric: continue
            if okey.evalset != evalset: continue
            if okey.sampler != sampler: continue
            if type(okey.eta) == type(()): continue
            if okey.T != T: continue
            if okey.N != N: continue
            X.append(okey.eta)
            Y.append(self.summary[okey]['mean'])
            S.append(self.summary[okey]['stdv']/self.summary[okey]['nb_samples']**0.5)
        X = np.array(X)
        Y = np.array(Y)
        S = np.array(S)

        return (X,Y,S)


    def query_multi_curve(self, metric='loss', evalset='test', sampler='sgd',
                          T_big=None, T_small=None):
        '''
        '''
        X, Y, S = [], [], []
        for okey in self.summary:
            if okey.kind != 'diff': continue
            if okey.metric != metric: continue
            if okey.evalset != evalset: continue
            if okey.sampler != sampler: continue
            if type(okey.eta) != type(()): continue
            if (okey.T) != (T_big, T_small): continue
            if okey.T[0] * okey.eta[0] != okey.T[1] * okey.eta[1]: continue
            X.append(okey.eta[1])
            Y.append(self.summary[okey]['mean'])
            S.append(self.summary[okey]['stdv']/self.summary[okey]['nb_samples']**0.5)
        X = np.array(X)
        Y = np.array(Y)
        S = np.array(S)

        return (X,Y,S)


