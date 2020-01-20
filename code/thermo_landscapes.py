''' author: samtenka
    change: 2020-01-13
    create: 2020-01-12
    descrp: instantiate abstract class `Landscape` for a normal-fitting model
'''


from utils import device, prod, secs_endured, megs_alloced, CC, pre, reseed

from landscape import PointedLandscape, FixedInitsLandscape

import tqdm
import numpy as np
import torch
from torch import conv2d, matmul, tanh
from torch.nn.functional import log_softmax, nll_loss 
from torchvision import datasets, transforms



#=============================================================================#
#           0. Linear Screw                                                   #
#=============================================================================#

class LinearScrew(FixedInitsLandscape):
    ''' 
    '''

    #-------------------------------------------------------------------------#
    #               0.0. getters and setters of weight (and data)             #
    #-------------------------------------------------------------------------#

    def __init__(self, seed=0):
        self.set_weight(self.sample_weight(seed))

    def sample_weight(self, seed):
        return np.array([0.0, 0.0, 0.0])

    def sample_data(self, N, seed): 
        '''
            since datapoints are just floats, we use them directly instead of
            using more indirect indices.
        '''
        reseed(seed)
        return np.random.randn(N) 

    def get_weight(self):
        return self.weight.detach().numpy()

    def set_weight(self, weight):
        self.weight = torch.autograd.Variable(
            torch.Tensor(weight),
            requires_grad=True
        )

    def update_weight(self, displacement):
        '''
            Add the given numpy displacement to the current weight.
        '''
        self.weight.data += displacement.detach().data

    #-------------------------------------------------------------------------#
    #               0.1. the subroutines and diagnostics of descent           #
    #-------------------------------------------------------------------------#

    def get_loss_stalk(self, data):
        '''
            Negative log prob of data under N(0, sigma^2=exp(self.weight))    
            normal distribution (actually, the log prob is offset by an
            additive constant).
        '''
        x, y, z = self.weight[0], self.weight[1], self.weight[2]
        c_cov, s_cov = torch.cos(z        ), torch.sin(z        )
        c_hes, s_hes = torch.cos(z+np.pi/4), torch.sin(z+np.pi/4)
        return (
            torch.pow(x*c_hes + y*s_hes, 2) / 2 +
            torch.pow(x                , 2) / 2 +
            torch.pow(          y      , 2) / 2 +
            (x*c_cov + y*s_cov).mul(np.mean(data))
        )

    def nabla(self, scalar_stalk, create_graph=True):
        '''
            Differentiate a stalk, assumed to be at the current weight, with
            respect to this weight.
        '''
        return torch.autograd.grad(
            scalar_stalk,
            self.weight,
            create_graph=create_graph,
        )[0] 

    def get_metrics(self, data):
        return {
            'loss': self.get_loss_stalk(data).detach().numpy(),
            'xy-rad2': np.sum(np.square(self.weight.detach().numpy()[:2])),
            'z': self.weight.detach().numpy()[2]
        }

#=============================================================================#
#           2. DEMONSTRATE INTERFACE by REPORTING GRAD STATS during DESCENT   #
#=============================================================================#

if __name__=='__main__':

    #-------------------------------------------------------------------------#
    #               2.0. descent hyperparameters                              #
    #-------------------------------------------------------------------------#

    N = 100000
    BATCH = 1
    TIME = 100000
    LRATE = 0.1
    pre(N%BATCH==0,
        'batch size must divide train size!'
    )

    #-------------------------------------------------------------------------#
    #               2.1 specify and load model                                #
    #-------------------------------------------------------------------------#

    ML = LinearScrew(seed=0)
    ML.load_from('saved-weights/linearscrew.npy', nb_inits=1, seed=0)

    D = ML.sample_data(N=N, seed=22) 
    for t in range(TIME):
        #---------------------------------------------------------------------#
        #           2.2 perform one descent step                              #
        #---------------------------------------------------------------------#

        L = ML.get_loss_stalk(D[(BATCH*t)%N:(BATCH*(t+1)-1)%N+1])
        G = ML.nabla(L)
        ML.update_weight(-LRATE * G)

        #---------------------------------------------------------------------#
        #           2.3 compute and display gradient statistics               #
        #---------------------------------------------------------------------#

        if (t+1)%1000: continue

        L_train= ML.get_metrics(D)
        data = ML.sample_data(N=30000, seed=1)
        L_test = ML.get_metrics(data)

        print(CC+' @D \t'.join([
            'after @M {:4d} @D steps'.format(t+1),
            'train loss @Y {:.4f}'.format(L_train['loss']),
            'test loss @L {:.4f}'.format(L_test['loss']),
            'xy-rad2 @B {:.4f}'.format(L_train['xy-rad2']),
            'z @B {:f}'.format(L_train['z']),
        '']))

