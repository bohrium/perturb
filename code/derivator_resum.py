''' author: samtenka
    change: 2020-04-14
    change: 2020-04-14
    descrp: define gradient statistics (resummed)
'''

import numpy as np

from utils import CC, pre, reseed 
from landscape import FixedInitsLandscape
from gradstats import grad_stat_names, GradStats
import torch
import tqdm

def compute_grad_stats_resum(land, N, SS,D,TT, etaT, I2, idx, seed=0):
    '''
    '''
    nab = land.nabla
    dim = len(D)

    print(CC+'\n\n\n\n\npreprocessing H ... @^ @^ @^ @^ @^ @^ ')

    expD = np.exp( -np.maximum(D*etaT,-1.0) )
    intD = (1.0 - expD) / D
    intH = torch.Tensor( np.matmul(SS * intD, TT) )
    H    = torch.Tensor( np.matmul(SS * D   , TT) )
    SS   = torch.Tensor(SS)
    TT   = torch.Tensor(TT)

    intD2H = torch.Tensor( ((1.0 - expD*expD) / (2.0 * D)) * D )

    gg   = []
    ggh  = []
    ch   = []
    gggj = []
    gcj  = []
    tj   = []
    ggjg = []
    cjg  = []

    print(CC+'\n\n\n\n\nstarting inner loop @^ @^ @^ @^ @^ @^ ')
    for i in tqdm.tqdm(range(I2), mininterval=1.0):
        land.switch_to(idx)

        A, B, C, D = (
            land.get_loss_stalk(land.sample_data(N, seed=seed+4*i+_))
            for _ in range(4)
        )
        
        GA, GB, GC, GD = (
            nab(X) 
            for X in (A, B, C, D)
        )

        GAintH = GA.unsqueeze(0).mul(intH)

        gg.append(
            GAintH
            .mm(GB.unsqueeze(1))
            .detach().numpy()
        )

        ggh.append(
            GAintH
            .mm(H)
            .mm(intH)
            .mm(GB.unsqueeze(1))
            .detach().numpy()
        )

        ch.append(
            N * (
                (GA-GB).unsqueeze(0).mm(SS).squeeze(0)
            ).mul(
                GA.unsqueeze(0).mm(SS).squeeze(0)
            ).dot(
                intD2H
            ).detach().numpy()
        )

        #gggj.append(
        #    (1.0/6) *
        #    nab(nab(GA.dot(GB.detach())).dot(GC.detach())).dot(GD.detach())
        #)

        #gcj.append(
        #    N *
        #    nab(nab(GC.dot(GA.detach()).dot((GA-GB).detach())).dot(GD)
        #)

        #tj.append(
        #    nab(nab(GB.dot(GA.detach())).dot(GA.detach())).dot(GA.detach()) # TODO: center! 
        #)

        #ggjg.append(
        #)

        #cjg.append(
        #)


    print(CC+'\n\n\n\n\nfinished inner loop @^ @^ @^ @^ @^ @^ ')

    return gg, ggh, ch

if __name__ == '__main__':
    from cifar_landscapes import CifarLogistic, CifarLeNet
    from fashion_landscapes import FashionLogistic, FashionLeNet
    from nongauss_landscapes import FitGauss, CubicChi
    from thermo_landscapes import LinearScrew

    model_nm = 'fashion-lenet' 
    model, in_nm = {
        'fashion-lenet': (
            FashionLeNet,
            'fashion-lenet.npy',
        ),
        'fashion-logistic': (
            FashionLogistic,
            'fashion-logistic.npy',
        ),
    }[model_nm]

    LC = model()
    LC.load_from('saved-weights/{}'.format(in_nm))

    for idx in tqdm.tqdm([0, 1]):
        D = np.load('hess-vals-{}-{}.npy'.format(model_nm, idx))
        T = np.load('hess-vecs-{}-{}.npy'.format(model_nm, idx))

        stats = {}

        #for etaT in tqdm.tqdm([1000*s for s in np.arange(0.005, 0.051, 0.005)]): 
        for etaT in tqdm.tqdm([1000*s for s in np.arange(0.05, 0.051, 0.05)]): 
            print(CC+'\n\n\nstarting idx = @P {} @D ; etaT = @O {} @D @^ @^ @^ @^ '.format(idx, etaT))
            gg, ggh, ch = compute_grad_stats_resum(
                LC, N=500,
                SS=np.transpose(T),D=D,TT=T,
                etaT=etaT,
                idx=idx, I2=4, seed=0
            )

            stats[(idx, etaT)] = {
                nm:{
                    'mean':np.mean(arr),
                    'stdv':np.std(arr),
                    'maxm':np.amax(arr),
                    'medn':np.median(arr),
                    'minm':np.amin(arr),
                    'trls':len(arr),
                }
                for nm, arr in {
                    'gg' : gg, 
                    'ggh': ggh,
                    'ch' : ch, 
                }.items()
            }

        out_nm = 'gs-resum-{}-{}.data'.format(model_nm, idx) 
        with open(out_nm, 'w') as f:
            f.write('{{\n    {}\n}}'.format(
                ',\n    '.join(
                    '{} : {{\n        {}\n    }}'.format(
                        str(k),
                        ',\n        '.join(
                            '{}:{}'.format(vk,vv) for vk,vv in v.items()
                        )
                    )
                    for k,v in stats.items()
                )
            ))
        print(CC+'\n\n\n\nsaved to @P {} @D @^ @^ @^ @^ @^ '.format(out_nm))
