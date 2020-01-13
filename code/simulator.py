''' author: samtenka
    change: 2019-08-17
    create: 2019-06-17
    descrp: do gradient descent on landscapes
'''

import numpy as np

from utils import CC, pre, reseed
from optimlogs import OptimKey, OptimLog
from landscape import PointedLandscape
#from mnist_landscapes import MnistLogistic, MnistLeNet, MnistMLP
from cifar_landscapes import CifarLogistic, CifarLeNet
#from quad_landscapes import Quadratic
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

def compute_losses(land, eta, T, N, I=1, idx=None, opts=opts, test_extra=300,
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

    for i in tqdm.tqdm(range(I)):

    #-------------------------------------------------------------------------#
    #               0.0 sample data (shared for all (opt, beta) pairs)        #
    #-------------------------------------------------------------------------#

        D = land.sample_data(N + (N + test_extra), seed=seed+i) 
        D_train, D_test = D[:N], D[N:]

        for opt, beta in opts: 
            nabla = land.nabla
            stalk = land.get_loss_stalk

    #-------------------------------------------------------------------------#
    #               0.1 define optimization updates                           #
    #-------------------------------------------------------------------------#

            compute_gradients = {
                'SGD':  lambda t:   nabla(stalk(D_train[(t%N):(t%N)+1]))  ,
                'GD':   lambda t:   nabla(stalk(D_train               ))  ,
                'GDC':  lambda t: ( nabla(stalk(D_train[:N//2]        ))  ,
                                    nabla(stalk(D_train[N//2:]        )) ),
            }[opt]
            compute_update = {
                'SGD':  lambda g: g,
                'GD':   lambda g: g,
                'GDC':  lambda g: (
                    (g[0] + g[1])/2 +
                    nabla(g[0].dot(g[0]-g[1]))*(N//2)
                ),
            }[opt]

    #-------------------------------------------------------------------------#
    #               0.2 perform optimization loop                             #
    #-------------------------------------------------------------------------#

            land.switch_to(idx)
            for t in range(T):
                land.update_weight(
                    -eta * compute_update(compute_gradients(t)).detach()
                )

    #-------------------------------------------------------------------------#
    #               0.3 compute losses and accuracies                         #
    #-------------------------------------------------------------------------#

            test_loss = land.get_loss_stalk(D_test).detach().numpy()
            test_acc = land.get_accuracy(D_test)
            for metric, tensor in {'loss': test_loss, 'acc':test_acc}.items():
                ol.accum(
                    OptimKey(
                        sampler=opt.lower(),
                        beta=beta,
                        eta=eta,
                        N=N,
                        T=T,
                        evalset='test',
                        metric=metric
                    ),
                    tensor
                )

    return ol

#=============================================================================#
#           1. SET SIMULATION HYPERPARAMETER RANGES                           #
#=============================================================================#

    #-------------------------------------------------------------------------#
    #               1.0 sanity check on quadratic landscape                   #
    #-------------------------------------------------------------------------#

def test_on_quad_landscape():
    LC = Quadratic(dim=12)
    DIM = 8
    Q = Quadratic(dim=DIM)
    ol = OptimLog()
    for eta in tqdm.tqdm(np.arange(0.0005, 0.005, 0.001)):
        for T in [100]:
            ol.absorb(compute_losses(Q, eta=eta, T=T, N=T, I=int(100000.0/(T+1))))
    print(ol)
    with open('ol.data', 'w') as f:
        f.write(str(ol))
    print(CC+'measured @G {:+.1f} @W - @G {:+.1f} @W \t expected @Y {:.1f} @W '.format(
        mean - 1.96 * stdv/nb_samples**0.5,
        mean + 1.96 * stdv/nb_samples**0.5,
        DIM/2.0 + DIM/2.0 * (1-ETA)**(2*T)
    ))

    #-------------------------------------------------------------------------#
    #               1.1 lenet                                                 #
    #-------------------------------------------------------------------------#

def simulate_lenet(idxs, T, N, I=100, eta_step=0.025, eta_max=0.25,
                   model=CifarLeNet, in_nm='saved-weights/cifar-lenet.npy',
                   out_nm=lambda idx:'ol-cifar-lenet-{:02d}.data'.format(idx)):
    '''
    '''
    LC = model()
    LC.load_from(in_nm)
    for idx in tqdm.tqdm(idxs):
        ol = OptimLog()
        for eta in tqdm.tqdm(np.arange(eta_step, eta_max+eta_step/2, eta_step)):
            for T in [T]:
                ol.absorb(compute_losses(LC, eta=eta, T=T, N=T, I=I, idx=idx, opts=[('SGD', None)]))

        with open(out_nm(idx), 'w') as f:
            f.write(str(ol))

if __name__=='__main__':
    simulate_lenet([0], T=100, N=100, I=10, eta_step=0.025, eta_max=0.25)
    #simulate_lenet([0], T=1000, N=1000, I=1000, eta_step=0.025, eta_max=0.25)

