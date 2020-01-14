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
    def __init__(self):
        self.logs = {
            nm:[] for nm in grad_stat_names.values()
        }
    def accum(self, name, value):
        self.logs[name].append(value.detach().numpy())
    def recent(self, name):
        return self.logs[name][-1]
    def __str__(self):
        return '{\n'+',\n'.join(
            '    "{}": {{ "mean":{}, "stdv":{}, "nb_samples":{} }}'.format(
                name, np.mean(values), np.std(values), len(values)
            )
            for name, values in sorted(self.logs.items())
        )+'\n}'
