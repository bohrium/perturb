''' author: samtenka
    change: 2020-04-14
    create: 2020-04-14
    descrp: diagonalize hessian
'''

import numpy as np
import torch
import tqdm

from utils import CC, pre, reseed 
from utils import secs_endured

from landscape import FixedInitsLandscape
from gradstats import grad_stat_names, GradStats

ref_dim = 1000
ref_dur = None
def estimate_time():
    global ref_dur
    r = (lambda r: r+np.transpose(r))(np.random.randn(ref_dim, ref_dim))
    start = secs_endured()
    vals,vecs = np.linalg.eigh(r)
    ref_dur = secs_endured() - start

def diagonalize_symmetric(h): 
    if ref_dur is None:
        estimate_time()

    s, _ = h.shape
    print('diagonalizing (will take about {:.1f} seconds)...'.format(
        ref_dur * (s/float(ref_dim))**3
    ))
    vals, vecs = np.linalg.eigh(h) 
    return vals, vecs

def compute_hessian(land, N=1000, seed=0):
    '''
    '''
    nab = land.nabla

    gs = GradStats()
    land.switch_to(idx)

    ll = land.get_loss_stalk(land.sample_data(N, seed=seed))
    lg = nab(ll) 
    dim = int(lg.shape[0])
    
    lh = np.zeros(shape=(dim, dim)) 
    for d in tqdm.tqdm(range(dim)):
        lh[d] = nab(lg[d]).detach().numpy()
    print(lh)
    return lh

if __name__ == '__main__':
    from cifar_landscapes import CifarLogistic, CifarLeNet
    from fashion_landscapes import FashionLogistic, FashionLeNet
    from nongauss_landscapes import FitGauss, CubicChi
    from thermo_landscapes import LinearScrew

    model_nm = 'fashion-lenet'
    model, in_nm, out_nm = {
        'fashion-lenet': (
            FashionLeNet,
            'fashion-lenet.npy',
            'gs-fashion-lenet-{:02d}.data',
        ),
        'fashion-logistic': (
            FashionLogistic,
            'fashion-logistic.npy',
            'gs-fashion-logistic-{:02d}.data',
        ),
    }[model_nm]

    LC = model()
    LC.load_from('saved-weights/{}'.format(in_nm))
    for idx in tqdm.tqdm([4, 5]):#tqdm.tqdm(range(6)):
        lh = compute_hessian(LC)
        np.save('hess-{}-{}.npy'.format(
            model_nm, idx
        ), lh)

        vals, vecs = diagonalize_symmetric(lh)
        np.save('hess-vals-{}-{}.npy'.format(model_nm, idx), vals)
        np.save('hess-vecs-{}-{}.npy'.format(model_nm, idx), vecs)
        print(vals)
