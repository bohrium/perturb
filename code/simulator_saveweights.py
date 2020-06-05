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
    'GDS'
    'GDT'
    'GDC'
]

def compute_losses(land, eta, T, N, I=1, idx=None, opts=opts, test_extra=30,
                   seed=0, record_train=True, SDE_alpha=16, load_or_set='load',
                   save_dt=None):
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

        actual_N   =  N    #*      (1 if opt!='SDE' else SDE_alpha**2 * 3)
        actual_T   =  T    #*      (1 if opt!='SDE' else SDE_alpha   ) 
        actual_eta =  eta  #/ float(1 if opt!='SDE' else SDE_alpha   ) 

        compute_gradients = {
            'SGD':  lambda D_train, t, i:  nabla(stalk(D_train[(t%N):(t%N)+1]))  ,
            'GD':   lambda D_train, t, i:  nabla(stalk(D_train               ))  ,
            'GDC':  lambda D_train, t, i: (
                                           nabla(stalk(D_train[:N//2]        )),   
                                           nabla(stalk(D_train[N//2:]        ))
                                          ),
            'GDS':  lambda D_train, t, i: (
                                           nabla(stalk(            D_train  )),   
                                           nabla(land.get_stic_reg(D_train  ))
                                          ),
            'GDT':  lambda D_train, t, i: (
                                           nabla(stalk(            D_train  )),   
                                           nabla(land.get_tic_reg(D_train  ))
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
            'GDS':  lambda a: a[0]+a[1],
            'GDT':  lambda a: a[0]+a[1],
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

        saved_weights = np.zeros((T//save_dt, land.get_weight().shape[-1]), dtype=np.float32)
        saved_weight2s= np.zeros((T//save_dt, land.get_weight().shape[-1]), dtype=np.float32)
        for i in tqdm.tqdm(range(I)):
            #-----------------------------------------------------------------#
            #       0.1 sample data (shared for all (opt, beta) pairs)        #
            #-----------------------------------------------------------------#

            D = land.sample_data(actual_N + (N + test_extra), seed=seed+i) 
            D_train, D_test = D[:actual_N], D[actual_N:]

            data_evalsets = [(D_test, 'test')] 
            if record_train:
                data_evalsets.append((D_train, 'train'))

            #-----------------------------------------------------------------#
            #       0.2 perform optimization loop                             #
            #-----------------------------------------------------------------#

            if load_or_set=='load':
                land.switch_to(idx)
            else:
                land.set_weight(land.inits[idx])
            for t in range((actual_T)):
                land.update_weight(
                    -actual_eta * compute_update(
                        compute_gradients(D_train, t, i)
                    ).detach()
                )

                #-------------------------------------------------------------#
                #       0.3 compute losses and accuracies                     #
                #-------------------------------------------------------------#
                if t % save_dt: continue
                saved_weights[t//save_dt] += land.get_weight()   
                saved_weight2s[t//save_dt] += np.square(land.get_weight())
        saved_weights /= float(T//save_dt)
        saved_weight2s/= float(T//save_dt)

    return saved_weights, saved_weight2s

#=============================================================================#
#           1. SET SIMULATION HYPERPARAMETER RANGES                           #
#=============================================================================#

#-----------------------------------------------------------------------------#
#                   1.1 lenet                                                 #
#-----------------------------------------------------------------------------#

def simulate_lenet(idxs, T, N, I, eta, model, opts,
                   in_nm, out_nm_by_idx, load_or_set='load', save_dt=None):
    '''
    '''
    LC = model()
    if load_or_set=='load':
        LC.load_from(in_nm, nb_inits=1, seed=0, idx=0)
    else: 
        LC.load_from(in_nm, nb_inits=1, seed=0, idx=None)

    for idx in tqdm.tqdm(idxs):
        sw, sw2 = compute_losses(
            LC, eta=eta, T=T, N=N, I=I, idx=idx,
            opts=opts, load_or_set=load_or_set,
            save_dt=save_dt
        )
        np.save(out_nm_by_idx(idx), (sw, sw2))

if __name__=='__main__':
    from cifar_landscapes import CifarLogistic, CifarLeNet
    from fashion_landscapes import FashionLogistic, FashionLeNet
    from nongauss_landscapes import FitGauss, CubicChi
    from thermo_landscapes import LinearScrew, Quad1D, Quad1DReg

    import sys

    pre(sys.argv[1][:2]=='T=', 'first arg should have form T=...')
    pre(sys.argv[2][:2]=='N=', 'second arg should have form N=...')
    T = int(sys.argv[1][2:])
    N = int(sys.argv[2][2:])
    model_nm = sys.argv[3]
    eta = float(sys.argv[4])
    save_dt = int(sys.argv[5])
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

    for TT in tqdm.tqdm(range(19999,100000,20000)[::-1]):
        model, in_nm, out_nm, I = {
            ##################################################################
            'cifar-lenet-valley': (
                CifarLeNet,
                'saved-weights/valley-cifar-lenet-0-{}.npy'.format(TT),
                'ol-valley-{}-cifar-lenet-T{}-0.weights.npy',
                int(4500000/T),
            ),
        }[model_nm]

        simulate_lenet(
            idxs=idxs, T=T, N=N, I=I, eta=eta,
            model=model, opts=opts,
            in_nm=in_nm,
            out_nm_by_idx=lambda idx: out_nm.format(TT, T, idx),
            load_or_set='set',
            save_dt=save_dt
        )

