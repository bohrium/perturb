''' author: samtenka
    change: 2019-08-17
    create: 2019-06-17
    descrp: do gradient descent on landscapes
'''

import numpy as np

from utils import CC, pre, reseed
from optimlogs import OptimKey, OptimLog
from landscape import PointedLandscape
from cifar_landscapes import CifarLogistic, CifarLeNet
from fashion_landscapes import FashionLogistic, FashionLeNet
from fitgauss_landscape import FitGauss
import torch
import tqdm



#=============================================================================#
#           0. DEFINE OPTIMIZATION LOOPS (SGD, GD, GDC)                       #
#=============================================================================#

opts = [
    ('SGD', None),
    ('GD', None),
    ('GDC', 1.0),
]

def compute_losses(land, eta, T, N, I=1, idx=None, opts=opts, test_extra=3,
                   seed=0):
    '''
        Simulate optimizers on  

        Argument details:
            land        --- pointed landscape to optimize on (see landscape.py) 
            T           --- number of updates to perform 
            N           --- size of training set
            I           --- number of trials (for each optimizer) to average
            idx         --- index of weight of fixedinits pointed landscape 
            opts        --- list of (sampler, beta) pairs
            test_extra  --- number of samples by which test exceeds train set
    '''
    pre(N%2==0,
        'GDC simulator needs N to be even for covariance estimation'
    )
    ol = OptimLog()

    nabla = land.nabla
    stalk = land.get_loss_stalk

    for opt, beta in opts: 
        #---------------------------------------------------------------------#
        #           0.0 define optimization updates                           #
        #---------------------------------------------------------------------#

        compute_gradients = {
            'SGD':  lambda D_train, t:  nabla(stalk(D_train[(t%N):(t%N)+1]))  ,
            'GD':   lambda D_train, t:  nabla(stalk(D_train               ))  ,
            'GDC':  lambda D_train, t: (nabla(stalk(D_train[:N//2]        )),   
                                        nabla(stalk(D_train[N//2:]        )) ),
        }[opt]
        compute_update = {
            'SGD':  lambda g: g,
            'GD':   lambda g: g,
            'GDC':  lambda a,b: (a+b)/2 + beta * nabla(a.dot(a-b))*(N//2),
        }[opt]

        for i in tqdm.tqdm(range(I)):
            #-----------------------------------------------------------------#
            #       0.1 sample data (shared for all (opt, beta) pairs)        #
            #-----------------------------------------------------------------#

            D = land.sample_data(N + (N + test_extra), seed=seed+i) 
            D_train, D_test = D[:N], D[N:]

            #-----------------------------------------------------------------#
            #       0.2 perform optimization loop                             #
            #-----------------------------------------------------------------#

            land.switch_to(idx)
            for t in range(T):
                land.update_weight(
                    -eta * compute_update(compute_gradients(D_train, t)).detach()
                )

            #-----------------------------------------------------------------#
            #       0.3 compute losses and accuracies                         #
            #-----------------------------------------------------------------#

            test_metrics = land.get_metrics(D_test)
            for metric_nm, val in test_metrics.items():
                ol.accum(
                    OptimKey(
                        kind='main', metric=metric_nm, evalset='test',
                        sampler=opt.lower(), beta=beta, eta=eta, N=N, T=T,
                    ),
                    val
                )

    return ol

#=============================================================================#
#           1. SET SIMULATION HYPERPARAMETER RANGES                           #
#=============================================================================#

#-----------------------------------------------------------------------------#
#                   1.0 sanity check on quadratic landscape                   #
#-----------------------------------------------------------------------------#

def test_on_quad_landscape(T=100):
    LC = Quadratic(dim=12)
    DIM = 8
    Q = Quadratic(dim=DIM)
    ol = OptimLog()
    for eta in tqdm.tqdm(np.arange(0.0005, 0.005, 0.001)):
        ol.absorb(compute_losses(Q, eta=eta, T=T, N=T, I=int(100000.0/(T+1))))
    print(ol)
    with open('ol.data', 'w') as f:
        f.write(str(ol))
    print(CC+'measured @G {:+.1f} @D - @G {:+.1f} @D \t '+
             'expected @Y {:.1f} @D '.format(
        mean - 1.96 * stdv/nb_samples**0.5,
        mean + 1.96 * stdv/nb_samples**0.5,
        DIM/2.0 + DIM/2.0 * (1-ETA)**(2*T)
    ))

#-----------------------------------------------------------------------------#
#                   1.1 lenet                                                 #
#-----------------------------------------------------------------------------#

def simulate_lenet(idxs, T, N, I, eta_d, eta_max, model, in_nm,
                   out_nm_by_idx):
    '''
    '''
    LC = model()
    LC.load_from(in_nm, nb_inits=6, seed=0)
    for idx in tqdm.tqdm(idxs):
        ol = OptimLog()
        for eta in tqdm.tqdm(np.arange(eta_d, eta_max+eta_d/2, eta_d)):
            for T in [T]:
                ol.absorb(compute_losses(
                    LC, eta=eta, T=T, N=N, I=I, idx=idx,
                    opts=[('SGD', None)]
                ))

        with open(out_nm_by_idx(idx), 'w') as f:
            f.write(str(ol))

if __name__=='__main__':
    import sys
    pre(sys.argv[1][:2]=='T=', 'first arg should have form T=...')
    pre(sys.argv[2][:2]=='N=', 'second arg should have form N=...')
    T = int(sys.argv[1][2:])
    N = int(sys.argv[2][2:])
    model_nm = sys.argv[3]
    eta_d = float(sys.argv[4])
    eta_max = float(sys.argv[5])
    idxs = list(int(i) for i in sys.argv[6].split(','))

    model, in_nm, out_nm, I = {
        'cifar-lenet': (
            CifarLeNet,
            'saved-weights/cifar-lenet.npy',
            'ol-cifar-lenet-T{}-{:02d}.data',
            int(100000/T),
        ),
        'fashion-lenet': (
            FashionLeNet,
            'saved-weights/fashion-lenet.npy',
            'ol-fashion-lenet-T{}-{:02d}.data',
            int(100000/T),
        ),
        'fit-gauss':   (
            FitGauss,
            'saved-weights/fitgauss.npy',
            'ol-fitgauss-T{}-{:02d}.data',
            int(3000000/T),
        ),
    }[model_nm]

    simulate_lenet(
        idxs=idxs, T=T, N=N, I=I,
        eta_d=eta_d, eta_max=eta_max,
        model=model, in_nm=in_nm,
        out_nm_by_idx=lambda idx: out_nm.format(T, idx)
    )
