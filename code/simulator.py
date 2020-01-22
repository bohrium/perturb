''' author: samtenka
    change: 2020-01-16 
    create: 2019-06-17
    descrp: do gradient descent on landscapes
'''

import numpy as np

from utils import CC, pre, reseed
from optimlogs import OptimKey, OptimLog
from landscape import PointedLandscape
import torch
import tqdm



#=============================================================================#
#           0. DEFINE OPTIMIZATION LOOPS (SGD, GD, GDC)                       #
#=============================================================================#

opts = [
    'SGD'
    'SDE'
    'GD'
    'GDC'
]

def compute_losses(land, eta, T, N, I=1, idx=None, opts=opts, test_extra=30,
                   seed=0, record_train=True, SDE_alpha=16):
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
    pre('GDC' not in opts or N%2==0,
        'GDC simulator needs N to be even for covariance estimation'
    )
    ol = OptimLog()

    nabla = land.nabla
    stalk = land.get_loss_stalk

    for opt in tqdm.tqdm(opts): 
        #---------------------------------------------------------------------#
        #           0.0 define optimization updates                           #
        #---------------------------------------------------------------------#

        # TODO: explain better:
        beta = float(eta) * float(N-1)/(4*N)

        actual_N   =  N    *      (1 if opt!='SDE' else SDE_alpha**2 * 3)
        actual_T   =  T    *      (1 if opt!='SDE' else SDE_alpha   ) 
        actual_eta =  eta  / float(1 if opt!='SDE' else SDE_alpha   ) 

        compute_gradients = {
            'SGD':  lambda D_train, t, i:  nabla(stalk(D_train[(t%N):(t%N)+1]))  ,
            'GD':   lambda D_train, t, i:  nabla(stalk(D_train               ))  ,
            'GDC':  lambda D_train, t, i: (
                                           nabla(stalk(D_train[:N//2]        )),   
                                           nabla(stalk(D_train[N//2:]        ))
                                          ),
            'SDE':  lambda D_train, t, i: (
                                           nabla(stalk(D_train[((3*t+0)*N*SDE_alpha) % actual_N : ((3*t+1)*N*SDE_alpha-1) % actual_N + 1])),  
                                           nabla(stalk(D_train[((3*t+1)*N*SDE_alpha) % actual_N : ((3*t+2)*N*SDE_alpha-1) % actual_N + 1])),  
                                           nabla(stalk(D_train[((3*t+2)*N*SDE_alpha) % actual_N : ((3*t+3)*N*SDE_alpha-1) % actual_N + 1])),  
                                           #(lambda: (reseed(i*actual_T + t), np.random.randn()))()[1], 
                                          ),
        }[opt]
        compute_update = {
            'SGD':  lambda g: g,
            'GD':   lambda g: g,
            'GDC':  lambda a: (
                (a[0]+a[1])/2 +
                2 * beta * nabla(
                    a[0].dot((a[0]-a[1]).detach())
                )*(N//2)
            ),
            'SDE':  lambda a: (
                #a[0] + a[2] * (a[1]-a[0])
                a[0] + (a[2]-a[1])*(1.0/2)**0.5 * (SDE_alpha * N*SDE_alpha - 1.0)**0.5
            ),
        }[opt]

        for i in tqdm.tqdm(range(I)):
            #-----------------------------------------------------------------#
            #       0.1 sample data (shared for all (opt, beta) pairs)        #
            #-----------------------------------------------------------------#

            D = land.sample_data(actual_N + (N + test_extra), seed=seed+i) 
            D_train, D_test = D[:actual_N], D[actual_N:]

            #-----------------------------------------------------------------#
            #       0.2 perform optimization loop                             #
            #-----------------------------------------------------------------#

            land.switch_to(idx)
            for t in range(actual_T):
                land.update_weight(
                    -actual_eta * compute_update(
                        compute_gradients(D_train, t, i)
                    ).detach()
                )

            #-----------------------------------------------------------------#
            #       0.3 compute losses and accuracies                         #
            #-----------------------------------------------------------------#
            data_evalsets = [(D_test, 'test')] 
            if record_train:
                data_evalsets.append((D_train, 'train'))

            for data, evalset_nm in data_evalsets:
                metrics = land.get_metrics(data)
                for metric_nm, val in metrics.items():
                    ol.accum(
                        OptimKey(
                            kind='main', metric=metric_nm, evalset=evalset_nm,
                            sampler=opt.lower(), eta=eta, N=N, T=T,
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
        ol.absorb_buffer(compute_losses(Q, eta=eta, T=T, N=T, I=int(100000.0/(T+1))))
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

def simulate_lenet(idxs, T, N, I, eta_d, eta_max, model, opts,
                   in_nm, out_nm_by_idx):
    '''
    '''
    LC = model()
    LC.load_from(in_nm, nb_inits=6, seed=0)
    for idx in tqdm.tqdm(idxs):
        ol = OptimLog()
        for eta in tqdm.tqdm(np.arange(eta_d, eta_max+eta_d/2, eta_d)):
            for T in [T]:
                ol.absorb_buffer(compute_losses(
                    LC, eta=eta, T=T, N=N, I=I, idx=idx,
                    opts=opts
                ))

        with open(out_nm_by_idx(idx), 'w') as f:
            f.write(str(ol))

def simulate_multi(idxs, N, Es, I, eta_d, eta_max, model,
                   in_nm, out_nm_by_idx):
    '''
    '''
    LC = model()
    LC.load_from(in_nm, nb_inits=6, seed=0)
    for idx in tqdm.tqdm(idxs):
        ol = OptimLog()
        for eta in tqdm.tqdm(np.arange(eta_d, eta_max+eta_d/2, eta_d)):
            for E in tqdm.tqdm(Es):
                ol.absorb_buffer(compute_losses(
                    LC, eta=eta/E, T=N*E,
                    # for OL diff comparison purposes, have I fixed between E values.
                    N=N, I=I, idx=idx, opts=['SGD']
                    #N=N, I=int(I/E), idx=idx, opts=['SGD']
                ))
        with open(out_nm_by_idx(idx), 'w') as f:
            f.write(str(ol))


if __name__=='__main__':
    from cifar_landscapes import CifarLogistic, CifarLeNet
    from fashion_landscapes import FashionLogistic, FashionLeNet
    from nongauss_landscapes import FitGauss, CubicChi
    from thermo_landscapes import LinearScrew

    import sys

    pre(sys.argv[1][:2]=='T=', 'first arg should have form T=...')
    pre(sys.argv[2][:2]=='N=', 'second arg should have form N=...')
    T = int(sys.argv[1][2:])
    N = int(sys.argv[2][2:])
    model_nm = sys.argv[3]
    eta_d = float(sys.argv[4])
    eta_max = float(sys.argv[5])
    idxs = list(int(i) for i in sys.argv[6].split(','))
    opts = sys.argv[7].split(',')
    opts = [
        {
            'sgd': 'SGD',
            'sde': 'SDE',
            'gd' : 'GD' ,
            'gdc': 'GDC',
        }[o]
        for o in opts
    ]

    model, in_nm, out_nm, I = {
        'cifar-logistic': (
            CifarLogistic,
            'saved-weights/cifar-logistic.npy',
            'ol-cifar-logistic-T{}-{:02d}-sde.data',
            int(100/T),
        ),
        'cifar-lenet': (
            CifarLeNet,
            'saved-weights/cifar-lenet.npy',
            'ol-cifar-lenet-T{}-{:02d}-sde.data',
            int(20000/T),
        ),
        'fashion-logistic': (
            FashionLogistic,
            'saved-weights/fashion-logistic.npy',
            'ol-fashion-logistic-T{}-{:02d}-sde.data',
            int(25000/T),
        ),
        'fashion-lenet': (
            FashionLeNet,
            'saved-weights/fashion-lenet.npy',
            'ol-fashion-lenet-T{}-{:02d}-sde.data',
            int(20000/T),
        ),
        'fit-gauss-sde':   (
            FitGauss,
            'saved-weights/fitgauss.npy',
            'ol-fitgauss-T{}-{:02d}-sde-smalleta-new-superfine.data',
            int(2000/T),
        ),
        'fit-gauss-sgd':   (
            FitGauss,
            'saved-weights/fitgauss.npy',
            'ol-fitgauss-T{}-{:02d}-sgd-smalleta.data',
            int(800000/T),
        ),
        'cubic-chi':   (
            CubicChi,
            'saved-weights/cubicchi.npy',
            'ol-cubicchi-T{}-{:02d}-real-loss.data',
            int(100000/T),
        ),
        'linear-screw-smalleta':   (
            LinearScrew,
            'saved-weights/linearscrew.npy',
            'ol-linear-screw-T{}-{:02d}-smalleta.data',
            int( 50000/T),
        ),
        'linear-screw-bigeta':   (
            LinearScrew,
            'saved-weights/linearscrew.npy',
            'ol-linear-screw-T{}-{:02d}-bigeta.data',
            int( 50000/T),
        ),
    }[model_nm]

    #simulate_multi(
    #    idxs=idxs, N=N, Es=[5, 8, 3, 2, 1], I=I,
    #    model=model,
    #    eta_d=eta_d, eta_max=eta_max,
    #    in_nm=in_nm, out_nm_by_idx=lambda idx: out_nm.format('', idx)
    #)
    simulate_lenet(
        idxs=idxs, T=T, N=N, I=I,
        eta_d=eta_d, eta_max=eta_max,
        model=model, opts=opts,
        in_nm=in_nm, out_nm_by_idx=lambda idx: out_nm.format(T, idx)
    )
