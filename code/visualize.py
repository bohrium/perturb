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
import matplotlib.ticker as ticker
import numpy as np
from predictor import Predictor
import coefficients
from optimlogs import OptimLog
import sys 
from convex import get_convex_hull 

#pre(len(sys.argv)==1+4,
#    '`visualize.py` needs 4 command line arguments: ol, gs, mode, outnm'
#)
#OPTIMLOGS_FILENM, GRADSTATS_FILENM, MODE, IMG_FILENM = sys.argv[1:] 

#=============================================================================#
#           0. PLOTTING HELP                                                  #
#=============================================================================#

#-----------------------------------------------------------------------------#
#                   0.0 plotting primitives                                   #
#-----------------------------------------------------------------------------#

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

def smart_round(a, b): 
    '''
        return a `round` interval [aa, bb] inside the interval [a, b]
    '''
    high_high =(10*b + 0*a)/10.0
    high_mid  = (9*b + 1*a)/10.0
    high_low  = (8*b + 2*a)/10.0

    low_high  = (3*b + 7*a)/10.0
    low_mid   = (2*b + 8*a)/10.0
    low_low   = (1*b + 9*a)/10.0

    for i in range(-10, 10):
        aa = np.ceil( low_mid * 10**i) / 10**i 
        bb = np.floor(high_mid* 10**i) / 10**i 
        if ((low_low  <= aa <= low_high ) and
            (high_low <= bb <= high_high)): break
    return aa, bb

def finish_plot(title, xlabel, ylabel, img_filenm, ymax=1.0, ymin=0.0):
    '''
    '''
    plt.title(title, x=0.5, y=0.9)

    plt.ylim([ymin, ymax])

    xlow, xhigh = smart_round(*plt.gca().get_xlim())
    plt.xticks([xlow, xhigh])
    ylow, yhigh = smart_round(*plt.gca().get_ylim())
    plt.yticks([ylow, yhigh])

    plt.gca().xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2e'))
    plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2e'))
    plt.yticks(rotation=90)

    plt.xlabel(xlabel)
    plt.gca().xaxis.set_label_coords(0.5, -0.01)

    plt.ylabel(ylabel)
    plt.gca().yaxis.set_label_coords(-0.01, 0.5)

    plt.legend(loc='center left')
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

def plot_bars(x, y, s, color, label, z=1.96, bar_width=1.0/100): 
    '''
        plot variance (s^2) around mean (y) via I-bars around a scatter plot
    '''
    e = bar_width * (max(x)-min(x))
    for (xx, yy, ss) in zip(x, y, s):
        # middle, top, and bottom stroke of I, respectively:
        plt.plot([xx,   xx  ], [yy-z*ss, yy+z*ss], color=color)
        plt.plot([xx-e, xx+e], [yy-z*ss, yy-z*ss], color=color)
        plt.plot([xx-e, xx+e], [yy+z*ss, yy+z*ss], color=color)
    # connect to the figure legend:
    plt.plot([xx, xx], [yy-z*ss, yy+z*ss], color=color, label=label)

def plot_tube(x, xs, y, ys, color, label, z=1.96, angle_granularity=36, alpha=0.5): 
    '''
        plot 2-D variance (xs^2, ys^2) around mean (x, y) via tube
    '''
    clusters = [
        [   
            (
                xx + z * ( + np.cos(angle)*xss ),
                yy + z * ( + np.sin(angle)*yss ),
            )
            for angle in np.arange(0.0, 1.0, 1.0/angle_granularity) * 2*np.pi
        ]
        for (xx, xss, yy, yss) in zip(x, xs, y, ys)
    ]

    for ca, cb in zip(clusters, clusters[1:]):
        tube_pts = get_convex_hull(ca + cb)
        plt.fill(
            [xx for xx,yy in tube_pts],
            [yy for xx,yy in tube_pts],
            facecolor=color, alpha=alpha
        )

    tube_pts = get_convex_hull(cb)
    plt.fill(
        [xx for xx,yy in tube_pts],
        [yy for xx,yy in tube_pts],
        facecolor=color, alpha=alpha, label=label
    )


def plot_plus(x, xs, y, ys, color, label, z=1.96): 
    '''
        plot 2-D variance (xs^2, ys^2) around mean (x, y) via plus signs
    '''
    for (xx, xss, yy, yss) in zip(x, xs, y, ys):
        # vertical and horizontal strokes of '+' sign, respectively:
        plt.plot([xx,       xx      ], [yy-z*yss, yy+z*yss], color=color)
        plt.plot([xx-z*xss, xx+z*xss], [yy    ,   yy      ], color=color)
    # connect to the figure legend:
    plt.plot([xx, xx], [yy, yy], color=color, label=label)


#-----------------------------------------------------------------------------#
#                   0.1 plotting primitives                                   #
#-----------------------------------------------------------------------------#

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

    return X, [0-3*max(S), max(Y)+3*max(S)]

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
#                   0.2 plotting primitives                                   #
#-----------------------------------------------------------------------------#

def plot_loss_vs_eta(ol_nm, gs_nm, img_nm, title, ylabel,
                     T=None, N=None, kind='main',
                     experiment_params_list=[], 
                     theory_params_list=[]):
    prime_plot()

    eta_range = [0]
    metric_range = []
    for evalset, sampler, metric, color, label in experiment_params_list:   
        etas, metrics = plot_experiment(
            ol_nm, color=color, label=label,
            T=T, kind=kind, evalset=evalset, sampler=sampler, metric=metric
        )
        eta_range += list(etas)
        metric_range += list(metrics)
    eta_range = interpolate(eta_range)

    for coeff_strs, deg, mode, color, label in theory_params_list: 
        plot_theory(
            gs_nm, eta_range, coeff_strs, deg=deg, mode=mode, T=T, N=N,
            color=color, label=label
        )

    print(CC+'@R rendering plot @D ...')
    finish_plot(
        title=title, xlabel='learning rate', ylabel=ylabel, img_filenm=img_nm,
        ymax=max(metric_range), ymin=min(metric_range)
    )

def plot_test():
    prime_plot()

    plot_plus(
        x = [0, 1, 2, 3],
        xs= [0.1, 0.2, 0.3, 0.4],
        y = [3, 2, 4, 8],
        ys= [0.1, 0.2, 0.3, 0.4],
        color=blue,
        label='gooo'
    )

    plot_tube(
        x = [0, 1, 2, 3],
        xs= [0.1, 0.2, 0.3, 0.4],
        y = [3, 2, 4, 8],
        ys= [0.1, 0.2, 0.3, 0.4],
        color=yellow,
        label='mooo'
    )

    print(CC+'@R rendering plot @D ...')
    finish_plot(
        title='moo', xlabel='learning rate', ylabel='yo', img_filenm='hi.png'
    )



#def plot_loss_vs_loss(ol_nm, gs_nm, img_nm, title, ylabel,
#                      T=None, N=None, kind='main', metric='loss',
#                      experiment_params_list=[], 
#                      theory_params_list=[]):
#    prime_plot()
#
#    eta_range = [0]
#    for evalset, sampler, color, label in experiment_params_list:   
#        eta_range += list(plot_experiment(
#            ol_nm, T=T, kind=kind, evalset=evalset, sampler=sampler,
#            color=color, label=label
#        ))
#    eta_range = interpolate(eta_range)
#
#    for coeff_strs, deg, mode, color, label in theory_params_list: 
#        plot_theory(
#            gs_nm, eta_range, coeff_strs, deg=deg, mode=mode, T=T, N=N,
#            color=color, label=label
#        )
#
#    print(CC+'@R rendering plot @D ...')
#    finish_plot(
#        title=title, xlabel='learning rate', ylabel=ylabel, img_filenm=img_nm
#    )

#-----------------------------------------------------------------------------#
#                   0.3 plotting primitives                                   #
#-----------------------------------------------------------------------------#

def plot_batch_match_loss_vs_eta(model_nm, idx, T):
    # TODO: change default ol_nm to ../logs/....
    title = (
        'GDC MIMICS SMALL-BATCH BEHAVIOR \n'
        '(test loss after {} steps on {})'.format(T, model_nm)
    )
    plot_loss_vs_eta(
        ol_nm  = 'ol-{}-T{}-{:02}-opts-new-smalleta.data'.format(model_nm, T, idx),
        gs_nm  = '../logs/gs-{}-{:02}.data'.format(model_nm, idx),
        img_nm = '_test-{}-{}.png'.format(model_nm, idx),
        title=title, ylabel='loss difference',
        T=T, N=T, kind='diff',
        experiment_params_list = [
            ('test', ('sgd', 'gd'), 'loss', dark_blue, 'sgd - gd'),
            ('test', ('sgd', 'gdc'), 'loss', dark_yellow, 'sgd - gdc')
        ], 
        theory_params_list = [
            (coefficients.gd_minus_sgd_vanilla_test, 1, 'poly', bright_yellow, 'prediction of sgd vs gd'),
            (coefficients.gd_minus_sgd_vanilla_test, 2, 'poly', bright_blue, 'prediction of sgd vs gdc')
        ],
    )

def plot_gen_gap_loss_vs_eta(model_nm, idx, T):
    # TODO: change default ol_nm to ../logs/....
    title = (
        'GENERALIZATION GAP \n'
        '(gen gap after {} steps on {})'.format(T, model_nm)
    )
    plot_loss_vs_eta(
        ol_nm  = 'ol-{}-T{}-{:02}-opts-new.data'.format(model_nm, T, idx),
        gs_nm  = '../logs/gs-{}-{:02}.data'.format(model_nm, idx),
        img_nm = '_gen-{}-{}.png'.format(model_nm, idx),
        title=title, ylabel='loss difference',
        T=T, N=T, kind='diff',
        experiment_params_list = [
            (('test', 'train'), 'sgd', 'loss', dark_blue, 'sgd gen gap'),
        ], 
        theory_params_list = [
            (coefficients.sgd_vanilla_gen, 1, 'poly', bright_red, 'degree 1 prediction of gen gap'),
            (coefficients.sgd_vanilla_gen, 2, 'poly', bright_yellow, 'degree 2 prediction of gen gap'),
        ],
    )

#def plot_sgd_sde_diff_vs_eta(model_nm, idx, T):
#    # TODO: change default ol_nm to ../logs/....
#    title = (
#        'SGD vs SDE\n'
#        '(test loss after {} steps on {})'.format(T, model_nm)
#    )
#    plot_loss_vs_eta(
#        ol_nm  = 'ol-{}-T{}-{:02}-sde.data'.format(model_nm, T, idx),
#        gs_nm  = '../logs/gs-{}-{:02}.data'.format(model_nm, idx),
#        img_nm = '_sde-{}-{}.png'.format(model_nm, idx),
#        title=title, ylabel='loss difference',
#        T=T, N=T, kind='main',
#        experiment_params_list = [
#            ('test', 'sde', 'loss', dark_blue, 'sgd - sde'),
#        ], 
#        theory_params_list = [
#            #(coefficients.sgd_vanilla_gen, 1, 'poly', bright_red, 'degree 1 prediction of gen gap'),
#            #(coefficients.sgd_vanilla_gen, 2, 'poly', bright_yellow, 'degree 2 prediction of gen gap'),
#        ],
#    )

def plot_gauss_nongauss_vs_eta(model_nm, idx, T):
    # TODO: change default ol_nm to ../logs/....
    title = (
        'NONGAUSSIAN NOISE AFFECTS SGD \n'
        '(test loss after {} steps on {})'.format(T, model_nm)
    )
    plot_loss_vs_eta(
        ol_nm  = 'ol-{}-T{}-{:02}-real-loss.data'.format(model_nm, T, idx),
        gs_nm  = '../logs/gs-{}-{:02}.data'.format(model_nm, idx),
        img_nm = '_nongauss-{}-{}.png'.format(model_nm, idx),
        title=title, ylabel='loss increase',
        T=T, N=T, kind='main',
        experiment_params_list = [
            ('test', 'sgd', 'real-loss', dark_blue, 'sgd'),
        ], 
        theory_params_list = [
            (coefficients.sgd_vanilla_gauss_test, 3, 'poly', bright_red,   'deg 3 prediction with gaussian approximation'),
            (coefficients.sgd_vanilla_test,       3, 'poly', bright_green, 'deg 3 prediction'),
        ],
    )


def plot_thermo_vs_eta(model_nm, idx, T):
    # TODO: change default ol_nm to ../logs/....
    title = (
        #'SGD SEEKS MINIMA FLAT WRT THE CURRENT COVARIANCE \n'
        'A NON-CONSERVATIVE ENTROPIC FORCE PUSHES SGD\n'
        '(displacement after {} steps on {})'.format(T, model_nm)
    )
    plot_loss_vs_eta(
        ol_nm  = 'ol-{}-T{}-{:02}.data'.format(model_nm, T, idx),
        gs_nm  = 'gs-{}-with-unit-source-{:02}.data'.format(model_nm, idx),
        img_nm = '_thermo-{}-{}.png'.format(model_nm, idx),
        title=title, ylabel='net displacement',
        T=T, N=T, kind='main',
        experiment_params_list = [
            ('test', 'sgd', 'z', dark_blue, 'net z-displacement by sgd'),
        ], 
        theory_params_list = [
             (coefficients.sgd_linear_screw_z        , 1, 'poly', bright_red,    'neglecting stochasticity'),
             (coefficients.sgd_linear_screw_z        , 3, 'poly', bright_yellow, 'deg 3 prediction'),
             (coefficients.sgd_linear_screw_renorm_z , 3, 'poly', bright_green,  'deg 3 prediction, renormalized'),
        ],
    )





#for model_nm in ['cifar-lenet', 'fashion-lenet']:
#    for idx in range(3):
#        plot_gen_gap_loss_vs_eta(model_nm, idx=idx, T=10)
#        #plot_batch_match_loss_vs_eta(model_nm, idx=idx, T=10)

#plot_test()
#plot_sgd_sde_diff_vs_eta('fashion-lenet', idx=0, T=10)
#plot_gauss_nongauss_vs_eta('fitgauss', idx=0, T=4)

#plt.figure(figsize=(8,4))
#plot_gauss_nongauss_vs_eta('cubicchi', idx=0, T=4)
plot_thermo_vs_eta('linear-screw', idx=0, T=10000)
