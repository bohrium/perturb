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

def streaming_axes(land, data, epochs=50):
    ll = land.get_loss_stalk(data)
    gl = land.nabla(ll).detach().numpy()

    dim = len(land.get_weight()) 

    la = land.get_loss_stalk(data[:len(data)//2])
    ga = land.nabla(la).detach().numpy()
   
    lb = land.get_loss_stalk(data[len(data)//2:])
    gb = land.nabla(la).detach().numpy()
 
    xvec = normalize(ga-gl)
    yvec = graham(xvec, gb-gl)

    eta = 1.0 
    for _ in tqdm.tqdm(range(epochs)): 
        eta = 0.8*eta
        avg_resid=0
        for p in data:
            la = land.get_loss_stalk([p])
            ga = land.nabla(la).detach().numpy()
            gd = ga-gl

            xcoef = np.dot(xvec, gd)
            ycoef = np.dot(yvec, gd)
            resid = gd - xcoef*xvec - ycoef*yvec  
            avg_resid += np.linalg.norm(resid) 

            xvec = normalize(xvec + eta*resid*xcoef)
            yvec = graham(xvec, yvec + eta*resid*ycoef)
    #    print(avg_resid/len(data))

    return xvec, yvec

#=============================================================================#
#       1. SCAN THROUGH GRID                                                  #
#=============================================================================#

def get_loss(land, data, displace): 
    w = land.get_weight() 
    land.update_weight(displace)
    print(land.get_weight())
    ll = float(land.get_loss_stalk(data).detach().numpy())
    land.set_weight(w)
    return ll

def get_cov(land, data, displace, proj): 
    w = land.get_weight() 
    land.update_weight(displace)

    ll = land.get_loss_stalk(data)
    gl = land.nabla(ll).detach().numpy()
    restl = np.matmul(proj, gl) 
    gg = np.outer(restl, restl)

    small = proj.shape[0] 
    sum_gradsqs = np.zeros((small, small), dtype=np.float32)
    for p in data:
        la = land.get_loss_stalk([p])
        ga = land.nabla(la).detach().numpy()
        resta = np.matmul(proj, gl) 
        sum_gradsqs += np.outer(resta, resta)

    land.set_weight(w)

    cov = (sum_gradsqs/len(data) - gg) * (len(data) / (len(data) - 1.0))
    return cov 

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

def plot_covars(X, Y, C): 
    for i in range(X.shape[0]): 
        for j in range(X.shape[1]): 
            dx = X[i][j]
            dy = Y[i][j]

            cov = C[i][j]#0.05 * np.array([[2+np.sin(5*dy), 1], [1, 2+np.sin(3*dx)]])  

            vals, vecs = np.linalg.eigh(cov) 
            w, h = max(0.0, vals[0])**0.5, max(0.0, vals[1])**0.5
            w, h = h+0.05*max(h,w), w+0.05*max(h,w)
            ellipse = Ellipse((0, 0), height=h, width=w, facecolor='red', edgecolor='red', alpha=0.2)
            angle = np.angle(vecs[0][0] + 1j*vecs[1][0])
            transf = transforms.Affine2D().rotate(angle).translate(dx, dy)
            ellipse.set_transform(transf + plt.gca().transData)
            plt.gca().add_patch(ellipse)

def finalize_plot(file_nm):
    plt.gca().set_aspect('equal', 'box')
    plt.savefig(file_nm)

#=============================================================================#
#       3. RENDER PICTURE                                                     #
#=============================================================================#

if __name__=='__main__':
    from cifar_landscapes import CifarLogistic, CifarLeNet
    from fashion_two import FashionShallowOr

    #-------------------------------------------------------------------------#
    #           3.0 specify and load model                                    #
    #-------------------------------------------------------------------------#

    #model_nm = 'LENET'
    #file_nm = 'saved-weights/valley-cifar-lenet-0-99999.npy'
    #ML = CifarLeNet(verbose=True, seed=0)
    #model_nm = 'LOGISTIC'
    #file_nm = 'saved-weights/valley-cifar-logistic-0-999.npy'
    #ML = CifarLogistic(verbose=True, seed=0)
    #ML.set_weight(np.load(file_nm))#[0])

    nb_inits = 1
    model_idx = 0 

    model_nm = 'SHALLOWOR'
    file_nm = 'saved-weights/fashion-{}.npy'.format(model_nm.lower())
    model = {'SHALLOWOR':FashionShallowOr}[model_nm]
    ML = model(verbose=True, seed=0)
    ML.load_from(file_nm, nb_inits=nb_inits, seed=0)
    ML.switch_to(model_idx)

    #-------------------------------------------------------------------------#
    #           3.1 obtain losses and covariances                             #
    #-------------------------------------------------------------------------#

    data = ML.sample_data(N=4096, seed=4) 

    #xvec, yvec = streaming_axes(ML, data)
    #proj = np.array([xvec, yvec])

    #scale = 0.00012
    scale = 10.0
    xcoefs, ycoefs = generate_grid(0.0, 5.0, scale, scale, scale/8.0)
    L = [] 
    C = []
    for i in tqdm.tqdm(range(xcoefs.shape[0])): 
        LL = []
        CC = []
        for j in tqdm.tqdm(range(xcoefs.shape[1])): 
            xc = xcoefs[i][j]
            yc = ycoefs[i][j]
            displacement = torch.Tensor([xc,yc])#torch.Tensor(xc*xvec + yc*yvec)

            LL.append(get_loss(ML, data, displacement))
            #CC.append(get_cov(ML, data, displacement, proj))

        L.append(LL)
        #C.append(CC)
    L = np.array(L)
    #C = np.array(C)

    #print(C)
    #ss = np.sqrt(np.mean(np.abs(C)))
    #print((scale/10.0) / ss)

    plot_losses(xcoefs, ycoefs, L)
    #plot_covars(xcoefs, ycoefs, C * ((scale/10.0) / ss)**2 )
    finalize_plot('heyo_.png')
