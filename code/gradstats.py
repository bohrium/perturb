''' author: samtenka
    change: 2019-06-17
    create: 2019-06-17
    descrp: interface for gradient statistics data structure
'''

import numpy as np
from utils import CC
import torch

grad_stat_names = {
    # degree 0
    '.':'()(0)', 
    # degree 1
    '-'   :'(01)(0-1)', 
    '-all':'(01)(01)',
    # degree 2
    'L'   :'(01-02)(0-1-2)',
    'Llvs':'(01-02)(0-12)',
    'Ltwg':'(01-02)(01-2)',
    'Lall':'(01-02)(012)',
    # degree 3: tree
    'Y'   :'(01-02-03)(0-1-2-3)',
    'Ylvs':'(01-02-03)(0-1-23)',
    'Yerg':'(01-02-03)(0-123)',     # an ergo sign
    'Ytwg':'(01-02-03)(01-2-3)',    
    'Y:&|':'(01-02-03)(01-23)',     # a colon and a bar
    'Yvee':'(01-02-03)(012-3)',
    'Yall':'(01-02-03)(0123)',
    # degree 3: vine
    'Z'   :'(01-02-13)(0-1-2-3)',
    'Zlvs':'(01-02-13)(0-1-23)',
    'Zalt':'(01-02-13)(0-12-3)',
    'Zexc':'(01-02-13)(0-123)',     # an exclamation point
    'Ztwg':'(01-02-13)(0-13-2)',
    'Zmid':'(01-02-13)(01-2-3)',
    'Z:&|':'(01-02-13)(01-23)',     # a colon and a bar
    'Zvee':'(01-02-13)(012-3)',
    'Zall':'(01-02-13)(0123)',
    'Z|&|':'(01-02-13)(02-13)',     # two bars
    'Z:&:':'(01-02-13)(03-12)',     # two colons
}

class GradStats(object):
    def __init__(self, buffer_len=10):
        self.buffer = {
            nm:[] for nm in grad_stat_names.values()
        }
        self.summary = {
            nm:{'mean':0.0, 'stdv':0.0, 'nb_samples':0}
            for nm in grad_stat_names.values()
        }
        self.recent_flushed = {
            nm:None for nm in grad_stat_names.values()
        }
        self.buffer_len = buffer_len

    def flush(self, name):
        vals = np.array(self.buffer[name])
        if not len(vals): return
        d = self.summary[name]
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

        self.recent_flushed[name] = self.buffer[name][-1] 
        self.buffer[name] = []

    def accum(self, name, value):
        self.buffer[name].append(value.detach().numpy())
        if len(self.buffer[name]) >= self.buffer_len:
            self.flush(name)

    def recent(self, name):
        return (
            self.buffer[name][-1] if self.buffer[name] else
            self.recent_flushed[name] 
        )

    def __str__(self):
        for name in self.buffer:
            self.flush(name)
        return '{\n'+',\n'.join(
            '    "{}": {{ "mean":{}, "stdv":{}, "nb_samples":{} }}'.format(
                name, d['mean'], d['stdv'], d['nb_samples']
            )
            for name, d in sorted(self.summary.items())
        )+'\n}'
