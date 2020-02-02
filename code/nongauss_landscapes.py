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
#           0. Fitting a Normal to Normal data                                #
#=============================================================================#

class FitGauss(FixedInitsLandscape):
    ''' 
    '''

    #-------------------------------------------------------------------------#
    #               0.0. getters and setters of weight (and data)             #
    #-------------------------------------------------------------------------#

    def __init__(self, seed=0):
        self.set_weight(self.sample_weight(seed))

    def sample_weight(self, seed):
        return np.array([0.0])

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
        return (
            self.weight +
            torch.exp(-self.weight).mul(np.mean(np.square(data))) 
            -1.0
        )/2.0

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
        w = self.weight.detach().numpy()[0]
        return {
            'loss': self.get_loss_stalk(data).detach().numpy()[0],
            'weight': w,
            'real-loss': (w + np.exp(-w) - 1.0)/2.0,
        }

#=============================================================================#
#           1. An artificial cubic landscape                                  #
#=============================================================================#

class CubicChi(FixedInitsLandscape):
    ''' 
    '''

    #-------------------------------------------------------------------------#
    #               1.0. getters and setters of weight (and data)             #
    #-------------------------------------------------------------------------#

    def __init__(self, seed=0):
        self.set_weight(self.sample_weight(seed))

    def sample_weight(self, seed):
        return np.array([0.0])

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
    #               1.1. the subroutines and diagnostics of descent           #
    #-------------------------------------------------------------------------#

    def get_loss_stalk(self, data):
        '''
            Negative log prob of data under N(0, sigma^2=exp(self.weight))    
            normal distribution (actually, the log prob is offset by an
            additive constant).
        '''
        return (
            self.weight.pow(3)
            - self.weight.mul(np.mean(np.square(data)) - 1.0) 
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
            'loss': self.get_loss_stalk(data).detach().numpy()[0],
            'weight': self.weight.detach().numpy()[0],
            'real-loss': (self.weight.detach().numpy()[0])**3
        }



#=============================================================================#
#           2. DEMONSTRATE INTERFACE by REPORTING GRAD STATS during DESCENT   #
#=============================================================================#

if __name__=='__main__':

    #-------------------------------------------------------------------------#
    #               2.0. descent hyperparameters                              #
    #-------------------------------------------------------------------------#

    N = 10
    BATCH = 1
    TIME = 10
    LRATE = 0.0001
    pre(N%BATCH==0,
        'batch size must divide train size!'
    )

    #-------------------------------------------------------------------------#
    #               2.1 specify and load model                                #
    #-------------------------------------------------------------------------#

    ML = CubicChi(seed=0)
    ML.load_from('saved-weights/cubicchi.npy', nb_inits=1, seed=0)

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

        if (t+1)%1: continue

        L_train= ML.get_metrics(D)
        data = ML.sample_data(N=30000, seed=1)
        L_test = ML.get_metrics(data)

        print(CC+' @D \t'.join([
            'after @M {:4d} @D steps'.format(t+1),
            'train loss @Y {:.4f}'.format(L_train['loss']),
            'test loss @L {:.4e}'.format(L_test['loss']),
            'real loss @B {:.4e}'.format(L_train['real-loss']),
            'weight @B {:.4f}'.format(L_train['weight']),
        '']))
