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
    high_high = (9*b + 1*a)/10.0
    high_mid  = (8*b + 2*a)/10.0
    high_low  = (6*b + 4*a)/10.0

    low_high  = (4*b + 6*a)/10.0
    low_mid   = (2*b + 8*a)/10.0
    low_low   = (1*b + 9*a)/10.0

    for i in range(-10, 10):
        aa = np.ceil( low_mid * 10**i) / 10**i 
        bb = np.floor(high_mid* 10**i) / 10**i 
        if ((low_low  <= aa <= low_high ) and
            (high_low <= bb <= high_high)): break
    return aa, bb

def finish_plot(title, xlabel, ylabel, img_filenm):
    '''
    '''
    plt.title(title, x=0.5, y=0.9)

    xlow, xhigh = smart_round(*plt.gca().get_xlim())
    plt.xticks([xlow, xhigh])
    ylow, yhigh = smart_round(*plt.gca().get_ylim())
    plt.yticks([ylow, yhigh])

    plt.gca().xaxis.set_major_formatter(ticker.FormatStrFormatter('%.1e'))
    plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1e'))
    plt.yticks(rotation=90)

    plt.xlabel(xlabel)
    plt.gca().xaxis.set_label_coords(0.5, -0.015)

    plt.ylabel(ylabel)
    plt.gca().yaxis.set_label_coords(-0.015, 0.5)

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

def plot_bars(x, y, s, color, label, z=1.96, bar_width=1.0/100): 
    '''
        plot variance (s^2) around mean (y) via S-bars around a scatter plot
    '''
    e = bar_width * (max(x)-min(x))
    for (xx, yy, ss) in zip(x, y, s):
        # middle, top, and bottom stroke of I, respectively:
        plt.plot([xx,   xx  ], [yy-z*ss, yy+z*ss], color=color)
        plt.plot([xx-e, xx+e], [yy-z*ss, yy-z*ss], color=color)
        plt.plot([xx-e, xx+e], [yy+z*ss, yy+z*ss], color=color)
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

def plot_loss_vs_eta(ol_nm, gs_nm, img_nm, model_nm,
                     T=None, N=None, kind='main', metric='loss',
                     experiment_params_list=[], 
                     theory_params_list=[]):
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
            "{}'s {} loss\n"
            '(after {} steps on {} samples from {})'
        ).format(
            sampler if type(sampler)==type('') else
            '{} vs {}'.format(*sampler),
            evalset,
            T,
            N,
            model_nm
        ),
        xlabel='learning rate', ylabel='loss', img_filenm=img_nm
    )

idx = 0
model_nm = 'cifar-lenet'
plot_loss_vs_eta(
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

