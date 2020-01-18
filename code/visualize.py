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

from utils import pre, CC
from matplotlib import pyplot as plt
import numpy as np
from predictor import Predictor
import coefficients
from optimlogs import OptimLog
import sys 

#pre(len(sys.argv)==1+4,
#    '`visualize.py` needs 4 command line arguments: ol, gs, mode, outnm'
#)
#OPTIMLOGS_FILENM, GRADSTATS_FILENM, MODE, IMG_FILENM = sys.argv[1:] 

#=============================================================================#
#           0. FILE READING                                                   #
#=============================================================================#

#-----------------------------------------------------------------------------#
#                   0.0 plotting primitives                                   #
#-----------------------------------------------------------------------------#

    #--------------------------------------------------------------------------#
    #               2.1 plotting primitives                                    #
    #--------------------------------------------------------------------------#

red     = '#cc4444';  bright_red     = '#ff6666';  dark_red     = '#aa2222'
yellow  = '#aaaa44';  bright_yellow  = '#cccc66';  dark_yellow  = '#888822'
green   = '#44cc44';  bright_green   = '#66ff66';  dark_green   = '#22aa22'
cyan    = '#44aaaa';  bright_cyan    = '#66cccc';  dark_cyan    = '#228888'
blue    = '#4444cc';  bright_blue    = '#6666ff';  dark_blue    = '#2222aa'
magenta = '#aa44aa';  bright_magenta = '#cc66cc';  dark_magenta = '#882288'

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

def plot_experiment(ol_nm, 
                    T=10, kind='main', evalset='test', sampler='sgd', metric='loss',
                    color=blue, label='experiment'):
    '''
    '''
    print(CC+'querying losses from @R optimlog @M {} @D ...'.format(ol_nm))
    OL = OptimLog(ol_nm)
    OL.load_from(ol_nm)
    (X, Y, S) = OL.query_eta_curve(
        kind=kind, evalset=evalset, sampler=sampler, T=T, metric=metric
    )
    plot_bars(X, Y, S, color=color, label=label)

    return X

def plot_theory(gs_nm,
                eta_range, coeff_strs, deg=2, mode='poly', T=None, N=None,
                color=None, label='theory'):
    print(CC+'computing predictions from @Y gradstats @M {} @D ...'.format(gs_nm))
    P = Predictor(gs_nm)
    losses = P.evaluate_expr(
        P.extrapolate_from_taylor(
            coeff_strs=coeff_strs, degree=deg, mode=mode
        ),
        params = {'T':T, 'eta':eta_range, 'e':np.exp(1), 'N':N}
    )
    plot_fill(
        eta_range, losses['mean'], losses['stdv'],
        color=color, label=label
    )

#-----------------------------------------------------------------------------#
#                   2.2 plotting primitives                                   #
#-----------------------------------------------------------------------------#

def plot_eta_curve(ol_nm, gs_nm, img_nm, model_nm,
                   T=None, N=None, kind='main', metric='loss',
                   experiment_params_list=[], 
                   theory_params_list=[]):
                   #coeff_strs, model_nm):
                   #T=10, N=None, kind='main', evalset='test', sampler='sgd', 
                   #deg=2, mode='poly'):
    prime_plot()

    eta_range = [0]
    for evalset, sampler, color, label in experiment_params_list:   
        eta_range += list(plot_experiment(
            ol_nm, T=T, kind=kind, evalset=evalset, sampler=sampler,
            color=color, label=label
        ))
    eta_range = interpolate(eta_range)

    for coeff_strs, deg, mode, color, label in theory_params_list: 
        plot_theory(
            gs_nm, eta_range, coeff_strs, deg=deg, mode=mode, T=T, N=N,
            color=color, label=label
        )

    print(CC+'@R rendering plot @D ...')
    finish_plot(
        title=(
            "{}'s {} Loss\n"
            '(after {} steps on {} samples from {})'
        ).format(sampler, evalset, T, N, model_nm),
        xlabel='learning rate', ylabel='loss', img_filenm=img_nm
    )

idx = 0
model_nm = 'cifar-lenet'
plot_eta_curve(
    ol_nm  = 'ol-{}-T10-{:02}-opts-new-smalleta.data'.format(model_nm, idx),
    gs_nm  = '../logs/gs-{}-{:02}.data'.format(model_nm, idx),
    img_nm = '_test-{}-{}.png'.format(model_nm, idx),
    model_nm = model_nm,
    T=10, N=10, kind='diff',
    experiment_params_list = [
        ('test', ('sgd', 'gd'), dark_blue, 'sgd-gd'),
        ('test', ('sgd', 'gdc'), bright_blue, 'sgd-gdc')
    ], 
    theory_params_list = [
        (coefficients.gd_minus_sgd_vanilla_test, 1, 'poly', red   , 'theory (deg 1 poly)'),
        (coefficients.gd_minus_sgd_vanilla_test, 2, 'poly', yellow, 'theory (deg 2 poly)')
    ]
)

#for i in range(1):
#    plot_SGD(
#        ol_nm =   '../logs/ol-cubicchi-T10-{:02}.data'.format(i),
#        gs_nm =   '../logs/gs-cubicchi-{:02}.data'.format(i),
#        img_nm= 'test-vanilla-cubicchi-poly-nongauss.png'.format(i),
#    )


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
