''' author: samtenka
    change: 2020-04-16
    create: 2020-04-16
    descrp:  
'''

import tqdm
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
import torch

from utils import CC, pre

#=============================================================================#
#       0. OBTAIN AXES TO WHICH TO RESTRICT                                   #
#=============================================================================#

def normalize(v):
    return v / np.linalg.norm(v)
def graham(normedx, y):
    return normalize(y - np.dot(normedx, y)*normedx)

#=============================================================================#
#       1. SCAN THROUGH GRID                                                  #
#=============================================================================#

def get_loss(land, data, new_weight): 
    land.set_weight(new_weight)
    return float(land.get_loss_stalk(data).detach().numpy())

#=============================================================================#
#       2. SLICE VISUALIZATION                                                #
#=============================================================================#

def generate_grid(x0, y0, h, w, dd):
    ''' return triangular grid '''
    X = []
    Y = []

    dvert = dd 
    dhori = dd*(3.0**0.5)/2
    for i, x in enumerate(np.arange(-w, w, dhori)): 
        XX = []
        YY = []
        for y in (np.arange(-h, h+dvert/2, dvert) + (i%2)*dvert/2): 
            XX.append(x0+x)
            YY.append(y0+y)
        X.append(XX)
        Y.append(YY)

    return tuple(np.array(A) for A in (X, Y))

def plot_losses(X, Y, L):
    cc = plt.gca().contourf(X, Y, L, levels=25, cmap=plt.cm.bone, alpha=0.5)
    plt.gcf().colorbar(cc)

def finalize_plot(file_nm):
    plt.gca().set_aspect('equal', 'box')
    plt.savefig(file_nm)

#=============================================================================#
#       3. RENDER PICTURE                                                     #
#=============================================================================#

if __name__=='__main__':
    from fashion_two import FashionShallowOr

    #-------------------------------------------------------------------------#
    #           3.0 specify and load model                                    #
    #-------------------------------------------------------------------------#

    nb_inits = 1
    model_idx = 0 

    model_nm = 'SHALLOWOR'
    file_nm = 'saved-weights/fashion-{}.npy'.format(model_nm.lower())
    model = {'SHALLOWOR':FashionShallowOr}[model_nm]
    ML = model(verbose=True, seed=0)
    ML.load_from(file_nm, nb_inits=nb_inits, seed=0)
    ML.switch_to(model_idx)

    #-------------------------------------------------------------------------#
    #           3.1 obtain losses                                             #
    #-------------------------------------------------------------------------#

    data = ML.sample_data(N=4096, seed=0) 

    scale = 10.0
    xcoefs, ycoefs = generate_grid(2.5, 7.5, scale, scale, scale/8.0)
    L = [] 
    C = []
    for i in tqdm.tqdm(range(xcoefs.shape[0])): 
        LL = []
        CC = []
        for j in tqdm.tqdm(range(xcoefs.shape[1])): 
            xc = xcoefs[i][j]
            yc = ycoefs[i][j]
            new_weight = torch.Tensor([xc,yc])
            LL.append(get_loss(ML, data, new_weight))
        L.append(LL)
    L = np.array(L)
    plot_losses(xcoefs, ycoefs, L)
    finalize_plot('heyo.png')
