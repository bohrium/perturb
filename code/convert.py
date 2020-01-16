''' author: samtenka
    change: 2020-01-16
    create: 2020-01-16
    descrp: data wrangling
'''

from collections import namedtuple
import sys
from utils import pre, CC



#=============================================================================#
#           0. TRANSLATE OLD OPTIMLOGS TO NEW OPTIMLOGS                       #
#=============================================================================#


OptimKey_old = namedtuple('OptimKey_old',
    ('metric', 'sampler', 'beta', 'eta', 'T', 'N', 'evalset')
)

OptimKey = namedtuple('OptimKey',
    ('kind', 'metric', 'sampler', 'beta', 'eta', 'T', 'N', 'evalset')
)

hamming = lambda x, y: sum((1 if xx!=yy else 0) for xx,yy in zip(x, y))
shrink = lambda x, y: ((x,y) if x!=y else x) 

def translate_optimlog(file_nm):
    with open(file_nm) as f:
        text = f.read().replace('OptimKey', 'OptimKey_old')
        try:
            d = eval(text)
        except TypeError:
            print(CC+'@^ @M {} @D is probably @Y already @D in the new format!'.format(file_nm))
            return
    
    d_new = {}
    for k, v in d.items():   
        if type(k.sampler) == type(()):
            dd = k._asdict() 
            fst, snd = (
                tuple(dd[kk][i] for kk in sorted(dd.keys()))
                for i in range(2)
            )
            if hamming(fst, snd) != 1:
                continue
            k_new = OptimKey(
                kind    = 'diff',
                sampler = shrink(k.sampler[0], k.sampler[1]),
                beta    = shrink(k.beta   [0], k.beta   [1]),
                eta     = shrink(k.eta    [0], k.eta    [1]),
                T       = shrink(k.T      [0], k.T      [1]),
                N       = shrink(k.N      [0], k.N      [1]),
                evalset = shrink(k.evalset[0], k.evalset[1]),
                metric  = shrink(k.metric [0], k.metric [1]),
            )
            d_new[k_new] = v
        else:
            k_new = OptimKey(
                kind    = 'main',
                sampler = k.sampler,
                beta    = k.beta   ,
                eta     = k.eta    ,
                T       = k.T      ,
                N       = k.N      ,
                evalset = k.evalset,
                metric  = k.metric ,
            )
            d_new[k_new] = v
    
    lines = [
        '    {}: \t {{ "mean":{}, "stdv":{}, "nb_samples":{} }}'.format(
            okey, stats['mean'], stats['stdv'], stats['nb_samples']
        )
        for okey, stats in d_new.items()
    ]
    text_new = '{\n'+',\n'.join(sorted(lines, reverse=True))+'\n}'
    
    with open(file_nm, 'w') as f:
        f.write(text_new)

file_nms = sys.argv[1:]
for file_nm in file_nms: 
    print(CC+'processing @M {} @D ...'.format(file_nm))
    translate_optimlog(file_nm)
