''' author: samtenka
    change: 2020-01-15
    create: 2019-02-14
    descrp: compare and plot descent losses as dependent on learning rate.
            Valid plotting modes are
                test-gd,  test-sgd,  test-gdc,  test-diff,  test-all
                train-gd, train-sgd, train-gdc, train-diff, train-all
                gen-gd,   gen-sgd,   gen-gdc,   gen-diff,   gen-all
            To run, type:
                python visualize.py new-data/ol-lenet-00.data new-data/gs-lenet-00.data test-sgd test-sgd-lenet-00.png
            The   optimlogs.data   gives   a filename storing descent trajectory summaries;
            the   gradstats.data   gives   a filename storing gradient statistic estimates;
            the   test-DIFF        gives   a plotting mode
            the   out-diff.png     gives   a filename to write to 
'''

from utils import pre
from matplotlib import pyplot as plt
import numpy as np
from predictor import Predictor
import coefficients
from optimlogs import OptimKey
import sys 

pre(len(sys.argv)==1+4,
    '`visualize.py` needs 4 command line arguments: ol, gs, mode, outnm'
)
OPTIMLOGS_FILENM, GRADSTATS_FILENM, MODE, IMG_FILENM = sys.argv[1:] 

OPTIMLOGS_FILENM='../logs/ol-fashion-lenet-T10-02.data'
GRADSTATS_FILENM='../logs/gs-fashion-lenet-02.data'

def get_optimlogs(optimlogs_filenm,
                  kind='main', metric='loss', evalset='test',
                  sampler='sgd', beta=None):
    with open(optimlogs_filenm) as f:
        ol = eval(f.read())

    X, Y, S = [], [], []
    last_okey = None
    for okey in ol:
        if okey.kind != kind: continue
        if okey.metric != metric: continue
        if okey.evalset != evalset: continue
        if okey.sampler != sampler : continue
        if okey.beta != beta: continue
        X.append(okey.eta)
        Y.append(ol[okey]['mean'])
        S.append(ol[okey]['stdv']/ol[okey]['nb_samples']**0.5)
        last_okey=okey
    X = np.array(X)
    Y = np.array(Y)
    S = np.array(S)

    return (X,Y,S), last_okey 

    #--------------------------------------------------------------------------#
    #               2.1 plotting primitives                                    #
    #--------------------------------------------------------------------------#

red     = '#cc4444'
yellow  = '#aaaa44'
green   = '#44cc44'
cyan    = '#44aaaa'
blue    = '#4444cc'
magenta = '#aa44aa'

def prime_plot():
    '''
    '''
    plt.clf()
    plt.tick_params(direction='in')
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)

def finish_plot(title, xlabel, ylabel, img_filenm):
    '''
    '''
    plt.title(title, x=0.5, y=0.9)
    plt.xlabel(xlabel)
    plt.gca().xaxis.set_label_coords(0.5, -0.015)
    plt.ylabel(ylabel)
    plt.gca().yaxis.set_label_coords(-0.015, 0.5)

    #ymin, ymax = plt.gca().get_ylim()
    #yminn = np.ceil(ymin/0.01) * 0.01
    #ymaxx = np.floor(ymax/0.01) * 0.01
    #plt.yticks(np.arange(yminn, ymaxx, (ymaxx-yminn)/5.0)) 
    #plt.yticks([2.5, 2.6])
    #plt.xticks([0.0, 0.25])

    plt.legend(loc='best')
    plt.savefig(img_filenm, pad_inches=0.05, bbox_inches='tight')

def plot_fill(x, y, s, color, label, z=1.96, alpha=0.5):
    '''
        plot variance (s^2) around mean (y) via 2D shading around a curve
    '''
    plt.plot(x, y, color=color, alpha=0.5)
    plt.fill(
        np.concatenate([x, x[::-1]]),
        np.concatenate([y-z*s, (y+z*s)[::-1]]),
        facecolor=color, alpha=alpha, label=label
    )

def plot_bars(x, y, s, color, label, z=1.96, bar_width=1.0/50): 
    '''
        plot variance (s^2) around mean (y) via S-bars around a scatter plot
    '''
    e = bar_width * (max(x)-min(x))
    for (xx, yy, ss) in zip(x, y, s):
        # middle, top, and bottom stroke of I, respectively:
        plt.plot([xx,   xx  ], [yy-z*ss, yy+z*ss], color=color)
        plt.plot([xx-e, xx+0], [yy-z*ss, yy-z*ss], color=color)
        plt.plot([xx-0, xx+e], [yy+z*ss, yy+z*ss], color=color)
    # connect to the figure legend:
    plt.plot([xx, xx], [yy-z*ss, yy+z*ss], color=color, label=label)

def interpolate(x, bins = 100):
    unif = np.arange(0.0, (bins+1.0)/bins, 1.0/bins)
    return unif * (max(x)-min(x)) + min(x)

def plot_SGD(ol_nm, gs_nm, img_nm, mode='ode'):
    prime_plot()

    (X, Y, S), okey = get_optimlogs(ol_nm)
    #X = X[4:]
    #Y = Y[4:]
    #S = S[4:]
    T = okey.T

    plot_bars(X, Y, S, color=blue, label='experiment')

    X = interpolate([0] + list(X))

    P = Predictor(gs_nm)
    for degree, color in {1:red, 2:yellow, 3:green}.items():
        losses = P.evaluate_expr(
            P.extrapolate_from_taylor(
                coeff_strs=coefficients.sgd_vanilla_test,
                degree=degree,
                mode=mode
            ),
            params = {'T':T, 'eta':X, 'e':np.exp(1)}
        )
        plot_fill(
            X, losses['mean'], losses['stdv'],
            color=color, label='theory (deg {} {})'.format(degree, mode)
        )

    finish_plot(
        title=(
            "Vanilla SGD's Test Loss\n"
            '(after {} steps on fashion lenet)'
        ).format(T),
        xlabel='learning rate', ylabel='test loss', img_filenm=img_nm
    )

for i in range(6):
    plot_SGD(
        ol_nm =   '../logs/ol-fashion-lenet-T10-{:02}.data'.format(i),
        gs_nm =   '../logs/gs-fashion-lenet-{:02}.data'.format(i),
        img_nm= 'test-vanilla-fashion-ode-{:02}.png'.format(i),
    )


#    #------------------------------------------------------------------------#
#    #               2.1 plot curves                                          #
#    #------------------------------------------------------------------------#
#
#metric, optimizer = MODE.split('-') 
#
#def plot_GEN():
#    prime_plot()
#
#    (X, Y, S), okey = get_optimlogs(OPTIMLOGS_FILENM, metric, optimizer, beta=0.0) 
#    X, Y, S = (np.array([0.0]+list(nparr)) for nparr in (X,Y,S))
#    plot_bars(X, Y, S, color=blue, label='experiment')
#
#    X = interpolate(X)
#
#    Y, S = sgd_gen(gradstats, eta=X, T=okey.T, degree=1) 
#    plot_fill(X, Y, S, color=red, label='theory (deg 1 poly)')
#
#    Y, S = sgd_gen(gradstats, eta=X, T=okey.T, degree=2) 
#    plot_fill(X, Y, S, color=yellow, label='theory (deg 2 poly)')
#
#    Y, S = sgd_gen(gradstats, eta=X, T=okey.T, degree=3) 
#    plot_fill(X, Y, S, color=green, label='theory (deg 3 poly)', alpha=0.25)
#
#    finish_plot(
#        title='Prediction of SGD \n(gen loss after {} steps on mnist-10 lenet)'.format(
#            okey.T
#        ), xlabel='learning rate', ylabel='gen loss', img_filenm=IMG_FILENM
#    )
#
#
#def plot_GAUSS():
#    prime_plot()
#
#    (X, Y, S), okey = get_optimlogs(OPTIMLOGS_FILENM, metric, optimizer, beta=None) 
#    plot_bars(X, Y, S, color=blue, label='experiment')
#    
#    X = interpolate(np.array([0] + list(X)))
#
#    Y, S = sgd_test_taylor(gradstats, eta=X, T=okey.T, degree=3) 
#    plot_fill(X, Y, S, color=green, label='theory (deg 3 poly)')
#
#    Y, S = sgd_test_taylor_gauss(gradstats, eta=X, T=okey.T, degree=3) 
#    plot_fill(X, Y, S, color=magenta, label='theory (deg 3 poly gauss)')
#
#    finish_plot(
#        title='Prediction of SGD \n(test loss after {} steps on cifar-10 lenet)'.format(
#            okey.T
#        ), xlabel='learning rate', ylabel='test loss', img_filenm=IMG_FILENM
#    )
#
#def plot_SGD():
#    prime_plot()
#
#    (X, Y, S), okey = get_optimlogs(OPTIMLOGS_FILENM, metric, optimizer, beta=None) 
#    plot_bars(X, Y, S, color=blue, label='experiment')
#    
#    X = interpolate(np.array([0] + list(X)))
#
#    Y, S = sgd_test_taylor(gradstats, eta=X, T=okey.T, degree=1) 
#    plot_fill(X, Y, S, color=red, label='theory (deg 1 poly)')
#    
#    Y, S = sgd_test_taylor(gradstats, eta=X, T=okey.T, degree=2) 
#    plot_fill(X, Y, S, color=yellow, label='theory (deg 2 poly)')
#    #
#    Y, S = sgd_test_taylor(gradstats, eta=X, T=okey.T, degree=3) 
#    plot_fill(X, Y, S, color=green, label='theory (deg 3 poly)')
#
#    #Y, S = sgd_test_taylor(gradstats, eta=X, T=okey.T, degree=1) 
#    #plot_fill(X, Y, S, color=red, label='theory (deg 1 ode)')
#    #
#    #Y, S = sgd_test_exponential(gradstats, eta=X, T=okey.T, degree=2)
#    #plot_fill(X, Y, S, color=yellow, label='theory (deg 2 ode)')
#    #
#    #Y, S = sgd_test_exponential(gradstats, eta=X, T=okey.T, degree=3)
#    #plot_fill(X, Y, S, color=green, label='theory (deg 3 ode)')
#
#    #plt.ylim((2.4, 3.1))
#    plt.ylim((2.5, 3.0))
#
#    finish_plot(
#        #title='Prediction of SGD \n(test loss after 100 steps on mnist-10 logistic)'.format(
#        title='Prediction of SGD \n(test loss after {} steps on cifar-10 lenet)'.format(
#            okey.T
#        ), xlabel='learning rate', ylabel='test loss', img_filenm=IMG_FILENM
#    )
#
#
#def plot_OPT(): 
#    prime_plot()
#
#    for opt, beta, color in [('diffc', 1.0, cyan), ('diff', 0.0, magenta)]:
#        (X, Y, S), okey = get_optimlogs(OPTIMLOGS_FILENM, metric, opt, beta) 
#        X, Y, S = (np.array([0]+list(A[:-1])) for A in (X, Y, S))
#        plot_bars(X, Y, S, color=color, label=opt)
#
#    X = interpolate(X)
#    plot_fill(X, 0.0*X, 0.0, color=cyan, label='prediction')
#    Y, S = sgd_gd_diff(gradstats, eta=X, T=okey.T, degree=2, N=okey.N) 
#    plot_fill(X, -Y, S, color=magenta, label='prediction')
#    #Y, S = sgd_gd_diff(gradstats, eta=X, T=okey.T, degree=3, N=okey.N) 
#    #plot_fill(X, -Y, S, color=red, label='prediction3')
#
#    plt.ylim([-0.03, +0.02])
#    finish_plot(
#        title='Comparison of Optimizers \n({} after {} steps on mnist-10 lenet)'.format(
#            metric,
#            okey.T
#        ), xlabel='learning rate', ylabel='test loss difference from vanilla SGD', img_filenm=IMG_FILENM
#    )
#
#def plot_BETA_SCAN(): 
#    prime_plot()
#
#    #for beta, color in [(10**-3.0, green), (10**-2.5, cyan), (10**-2.0, blue), (10**-1.5, magenta), (10**-1.0, red)]:
#    #for beta, color in [(0.25, green), (0.5, cyan), (1.0, blue), (2.0, magenta), (4.0, red)]:
#    #for beta, color in [(0.25, green), (0.5, cyan), (1.0, blue)]:#, (2.0, magenta), (4.0, red)]:
#    #for beta, color in [(0.0, green), (0.25, cyan)]:
#    for beta, color in [(0.0, green), (0.25, cyan), (0.5, blue), (1.0, magenta), (2.0, red), (4.0, yellow)]:
#        (X, Y, S), okey = get_optimlogs(OPTIMLOGS_FILENM, metric, 'sgdc', beta) 
#        plot_bars(X, Y, S, color=color, label='sgdc {:.2e}'.format(beta))
#
#    finish_plot(
#        title='Comparison of Optimizers \n({} after {} steps on mnist-10 lenet)'.format(
#           metric,
#            okey.T
#        ), xlabel='learning rate', ylabel=metric, img_filenm=IMG_FILENM
#    )
#
#def plot_EPOCH(): 
#    prime_plot()
#
#    #for opt, beta, color in [('sgd.e2', 0.0, cyan), ('sgd.h2', 0.0, magenta)]:
#    for opt, beta, color in [('diff.e2.h2', 0.0, yellow)]:
#        (X, Y, S), okey = get_optimlogs(OPTIMLOGS_FILENM, metric, opt, beta) 
#        plot_bars(X, Y, S, color=color, label=opt)
#
#    X = interpolate(X)
#
#    #Y, S = sgd_test_taylor(gradstats, eta=2*X, T=okey.T, degree=2) 
#    #Y, S = sgd_test_multiepoch(gradstats, eta=2*X, T=okey.T, degree=2, E=1) 
#    #Y_, S_ = sgd_test_multiepoch(gradstats, eta=X, T=okey.T, degree=2, E=2) 
#    #Y, S = Y_ - Y, S_ + S
#    Y, S = sgd_test_multiepoch_diff_e2h2(gradstats, eta=X, T=okey.T, degree=2, E=2) 
#    plot_fill(X, Y, S, color=green, label='theory (deg 2 poly)')
#
#    finish_plot(
#        title='Comparison of Optimizers \n({} after {} steps on mnist-10 lenet)'.format(
#            metric,
#            okey.T
#        ), xlabel='learning rate', ylabel=metric, img_filenm=IMG_FILENM
#    )
#
#
##plot_GEN()
##plot_EPOCH()
##plot_SGD()
#plot_GAUSS()
##plot_OPT()
#plot_BETA_SCAN()
#