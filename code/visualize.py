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
black   = '#000000'

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
    high_high =( 9.5*b + 0.5*a)/10.0
    high_mid  = (9.0*b + 1. *a)/10.0
    high_low  = (9.0*b + 1. *a)/10.0

    low_high  = (1.5*b + 8.5*a)/10.0
    low_mid   = (1.5*b + 8.5*a)/10.0
    low_low   = (0.5*b + 9.5*a)/10.0

    for i in range(-10, 10):
        aa = np.ceil( low_mid * 10**i) / 10**i 
        bb = np.floor(high_mid* 10**i) / 10**i 
        if ((low_low  <= aa <= low_high ) and
            (high_low <= bb <= high_high)): break
    return aa, bb

def finish_plot(title, xlabel, ylabel, img_filenm, ymax=1.0, ymin=0.0):
    '''
    '''
    #plt.title(title, x=0.5, y=0.9)
    plt.title(title, x=0.5, y=0.8)

    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    plt.ylim([ymin, ymax])
    #plt.gca().axis('equal')

    xlow, xhigh = smart_round(*plt.gca().get_xlim())
    plt.xticks([xlow, xhigh])
    ylow, yhigh = smart_round(*plt.gca().get_ylim())
    plt.yticks([ylow, yhigh])

    plt.gca().xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2e'))
    plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2e'))
    plt.yticks(rotation=90)

    plt.gca().tick_params(axis='both', which='major', labelsize=14)

    plt.xlabel(xlabel, fontsize=14)
    plt.gca().xaxis.set_label_coords(0.5, -0.01)

    plt.ylabel(ylabel, fontsize=14)
    plt.gca().yaxis.set_label_coords(-0.01, 0.5)

    plt.legend(loc='lower left')
    #plt.legend(bbox_to_anchor=(0.4, 0.9), loc=2)

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
                    T=10, N=10, kind='main', evalset='test', sampler='sgd', metric='loss',
                    color=blue, label='experiment'):
    '''
    '''
    print(CC+'querying losses from @R optimlog @M {} @D ...'.format(ol_nm))
    OL = OptimLog(ol_nm)
    OL.load_from(ol_nm)
    (X, Y, S) = OL.query_eta_curve(
        kind=kind, evalset=evalset, sampler=sampler, T=T, N=N, metric=metric
    )
    plot_bars(X, Y, S, color=color, label=label)

    return X, [min(Y)-3*max(S), max(Y)+3*max(S)]

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
            T=T, N=N, kind=kind, evalset=evalset, sampler=sampler, metric=metric
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
        ymin=min(metric_range)-0.1*(max(metric_range)-min(metric_range)),
        ymax=max(metric_range)+0.1*(max(metric_range)-min(metric_range)),
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
        'GDC Mimics Small-Batch behavior \n'
        '(test loss on {})'.format(T, model_nm)
    )
    plot_loss_vs_eta(
        #ol_nm  = 'ol-{}-T{}-{:02}-opts-new.data'.format(model_nm, T, idx),
        ol_nm  = 'ol-{}-T{}-{:02}-bm.data'.format(model_nm, T, idx),
        gs_nm  = '../logs/gs-{}-{:02}.data'.format(model_nm, idx),
        img_nm = '../plots/_bm-{}-{}.png'.format(model_nm, idx),
        title=title, ylabel='loss difference',
        T=T, N=T, kind='diff',
        experiment_params_list = [
            ('test', ('sgd', 'gd'), 'loss', dark_blue, 'sgd - gd'),
            ('test', ('sgd', 'gdc'), 'loss', dark_yellow, 'sgd - gdc')
        ], 
        theory_params_list = [
            (coefficients.gd_minus_sgd_vanilla_test, 2, 'poly', bright_blue, 'prediction of sgd vs gd'),
            (coefficients.gd_minus_sgd_vanilla_test, 1, 'poly', bright_yellow, 'prediction of sgd vs gdc'),
        ],
    )

def plot_gen_gap_loss_vs_eta(model_nm, idx, T):
    # TODO: change default ol_nm to ../logs/....
    title = (
        'GENERALIZATION GAP \n'
        '(gen gap after {} steps on {})'.format(T, model_nm)
    )
    plot_loss_vs_eta(
        ol_nm  = 'ol-{}-T{}-{:02}-gen.data'.format(model_nm, T, idx),
        gs_nm  = '../logs/gs-{}-{:02}.data'.format(model_nm, idx),
        img_nm = '../plots/_gen-{}-{}.png'.format(model_nm, idx),
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

def plot_test_loss_vs_eta(model_nm, idx, T):
    title = (
        'Vanilla SGD\'s Test Loss \n'
        '(T=10, Fashion-MNIST convnet)'.format(T, model_nm)
    )

    plt.rcParams.update({'font.size': 14})
    plot_loss_vs_eta(
        ol_nm  = '../logs/ol-{}-T{}-{:02}.data'.format(model_nm, T, idx),
        gs_nm  = '../logs/gs-{}-{:02}.data'.format(model_nm, idx),
        img_nm = '../plots/new-test-{}.png'.format(idx),
        title=title, ylabel='test loss',
        T=T, N=T, kind='main',
        experiment_params_list = [
            ('test', 'sgd', 'loss', dark_blue, 'experiment'),
        ], 
        theory_params_list = [
            (coefficients.sgd_vanilla_test, 1, 'poly', bright_red, 'deg 1 prediction'),
            (coefficients.sgd_vanilla_test, 2, 'poly', bright_yellow, 'deg 2 prediction'),
            (coefficients.sgd_vanilla_test, 3, 'poly', bright_green, 'deg 3 prediction'),
        ],
    )


def plot_gen_gap_loss_vs_loss(model_nm, idxs, T):
    # TODO: change default ol_nm to ../logs/....
    title = (
        'GENERALIZATION GAP \n'
        '(gen gap after {} steps on {})'.format(T, model_nm)
    )

    true_mean = [] 
    true_stdv = [] 
    pred1_mean = [] 
    pred1_stdv = [] 
    pred2_mean = [] 
    pred2_stdv = [] 
    for idx in idxs:
        ol_nm  = 'ol-{}-T{}-{:02}-gen.data'.format(model_nm, T, idx)

        OL = OptimLog(ol_nm)
        OL.load_from(ol_nm)
        (X, Y, S) = OL.query_eta_curve(
            kind='diff', evalset=('test', 'train'), sampler='sgd', T=T, N=T, metric='loss'
        )
        true_mean += list(Y)
        true_stdv += list(S)

        gs_nm  = '../logs/gs-{}-{:02}.data'.format(model_nm, idx)
        P = Predictor(gs_nm)
        losses = P.evaluate_expr(
            P.extrapolate_from_taylor(
                coeff_strs=coefficients.sgd_vanilla_gen, degree=1, mode='poly'
            ),
            params = {'T':T, 'eta':X, 'e':np.exp(1), 'N':T}
        )
        pred1_mean += list(losses['mean'])
        pred1_stdv += list(losses['stdv'])

        losses = P.evaluate_expr(
            P.extrapolate_from_taylor(
                coeff_strs=coefficients.sgd_vanilla_gen, degree=2, mode='poly'
            ),
            params = {'T':T, 'eta':X, 'e':np.exp(1), 'N':T}
        )
        pred2_mean += list(losses['mean'])
        pred2_stdv += list(losses['stdv'])


    true_mean = np.array(true_mean) 
    true_stdv = np.array(true_stdv) 
    pred1_mean = np.array(pred1_mean) 
    pred1_stdv = np.array(pred1_stdv) 
    pred2_mean = np.array(pred2_mean) 
    pred2_stdv = np.array(pred2_stdv) 

    plot_bars(pred2_mean, true_mean, true_stdv, color=blue, label=None)
    #plot_fill(pred2_mean, pred1_mean, pred1_stdv, color=red, label=None)
    #plot_fill(pred2_mean, pred2_mean, pred2_stdv, color=yellow, label=None)

    img_nm = '../plots/_gen-{}-{}.png'.format(model_nm, idx)
    print(CC+'@R rendering plot @D ...')
    finish_plot(
        title=title, xlabel='prediction',
        ylabel='actual', img_filenm='../plots/big-gen.png',
        ymin=min(true_mean-5*true_stdv), ymax=max(true_mean+5*true_stdv),
    )



def plot_sgd_sde_diff_vs_eta(model_nm, idx, T):
    title = (
        'SGD DIFFERS FROM SDE\n'
        '(test loss after {} steps on {})'.format(T, model_nm)
    )
 
    prime_plot()
 
    ol_nm_sgd  = 'ol-{}-T{}-{:02}-sgd-smalleta.data'.format(model_nm, T, idx)
    ol_nm_sde  = 'ol-{}-T{}-{:02}-sde-smalleta-new-fine.data'.format(model_nm, T, idx)

    OL = OptimLog(ol_nm_sgd)
    OL.load_from(ol_nm_sgd)
    (X, Y_sgd, S_sgd) = OL.query_eta_curve(
        kind='main', evalset='test', sampler='sgd', T=T, N=1,metric='real-loss'
    )

    OL = OptimLog(ol_nm_sde)
    OL.load_from(ol_nm_sde)
    (_, Y_sde, S_sde) = OL.query_eta_curve(
        kind='main', evalset='test', sampler='sde', T=T, N=1,metric='real-loss'
    )

    metric_range = [
        min(Y_sgd-Y_sde) - 5*max(S_sgd+S_sde),
        max(Y_sgd-Y_sde) + 5*max(S_sgd+S_sde)
    ]
    print(Y_sgd-Y_sde)
    plot_bars(X, Y_sgd-Y_sde, S_sgd+S_sde, color='blue', label='sgd - sde')

    eta_range = interpolate([0] + list(X))
 
    gs_nm = '../logs/gs-{}-{:02}.data'.format(model_nm, idx)
    plot_theory(
        gs_nm, eta_range,
        coeff_strs=coefficients.sgd_minus_sde_vanilla_test,
        deg=1, mode='poly', T=T, N=T,
        color=bright_red, label='no difference'
    )
    plot_theory(
        gs_nm, eta_range,
        coeff_strs=coefficients.sgd_gauss_minus_sde_vanilla_test,
        deg=3, mode='poly', T=T, N=T,
        color=bright_yellow, label='theory: discrete time and gaussian noise'
    )
    plot_theory(
        gs_nm, eta_range,
        coeff_strs=coefficients.sgd_minus_sde_vanilla_test,
        deg=3, mode='poly', T=T, N=T,
        color=bright_green, label='theory: discrete time and non-gaussian noise'
    )
 
    print(CC+'@R rendering plot @D ...')
    finish_plot(
        title=title, xlabel='learning rate',
        ylabel='loss difference', img_filenm='../plots/vs-sde.png',
        ymax=max(metric_range), ymin=min(metric_range)
    )


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
        'A NON-CONSERVATIVE ENTROPIC FORCE\n'
        '(T={} SGD on {})'.format(T, model_nm)
        #'(displacement after {} steps on {})'.format(T, model_nm)
    )
    plt.rcParams.update({'font.size': 14})
    plot_loss_vs_eta(
        ol_nm  = 'ol-{}-T{}-{:02}.data'.format(model_nm, T, idx),
        gs_nm  = 'gs-{}-with-unit-source-{:02}.data'.format(model_nm, idx),
        #img_nm = '../plots/new-thermo-{}-{}.png'.format(model_nm, idx),
        img_nm = '../plots/new-thermo-linear-screw.png'.format(model_nm, idx),
        title=title, ylabel='displacement',
        T=T, N=T, kind='main',
        experiment_params_list = [
            ('test', 'sgd', 'z', dark_blue, 'net z-displacement'),
        ], 
        theory_params_list = [
             (coefficients.sgd_linear_screw_z        , 3, 'poly', bright_yellow, 'deg 3 prediction'),
             (coefficients.sgd_linear_screw_renorm_z , 3, 'poly', bright_green,  'deg 3 prediction, renorm'),
             (coefficients.sgd_linear_screw_z        , 1, 'poly', bright_red,    'Chaudhari & Soatto 2018'),
        ],
    )

def plot_multi_vs_eta(model_nm, idx):
    # TODO: change default ol_nm to ../logs/....
    title = (
        'EPOCHS DULL SGD\'s CHLADNI EFFECT \n'
        '(test loss difference on {})'.format(model_nm)
    )

    ol_nm = 'ol-{}-T-{:02}-multi.data'.format(model_nm, idx)

    OL = OptimLog(ol_nm)
    OL.load_from(ol_nm)

    metrics = []
    for col, T_big in [(magenta,80), (blue,50), (green,30), (red,20)]:
        (X, Y, S) = OL.query_multi_curve(
            evalset='test', sampler='sgd', T_big=T_big, T_small=10, metric='loss'
        )
        plot_bars(X, Y, S, color=col, label='sgd {} vs sgd 10'.format(T_big))
        metrics += [max(Y+5*S), min(0-3*S)]

        eta_range = interpolate([0] + list(X))

        gs_nm = 'gs-{}-{:02}.data'.format(model_nm, idx)
        plot_theory(
            gs_nm, eta_range,
            coeff_strs=coefficients.sgd_multi_minus_vanilla_test,
            deg=2, mode='poly', T=T_big, N=10,
            color=col, label='deg 2 prediction'
        )
 

    print(CC+'@R rendering plot @D ...')
    finish_plot(
        title=title, xlabel='learning rate', ylabel='loss',
        img_filenm='../plots/_multi-cifar-logistic-0.png',
        ymax=max(metrics), ymin=min(metrics)
    )


def plot_gengap_vs_hess(model_nm, T, N=10):
    title = (
        'SHARP AND FLAT MINIMA BOTH OVERFIT LESS\n'
        '(test loss difference on {})'.format(model_nm)
    )
 
    prime_plot()
    plt.rcParams.update({'font.size': 14})

    hesses = (
        list(np.arange(0.00, 1.01, 0.1)) +
        list(np.arange(1.00, 5.01, 0.5))
    )
    interp_hesses = interpolate([1e-10]+hesses[1:])

    for T, col in [(20, red), (10, blue)]:
        etas = []
        gg_mean = [] 
        gg_stdv = [] 
        tl_mean = []
        tl_stdv = []
        for hess in hesses:
            ol_nm = '../plots/old-quads/ol-quad-1d-h{:0.2f}'.format(hess)

            OL = OptimLog(ol_nm)
            OL.load_from(ol_nm)
            (X, Y, S) = OL.query_eta_curve(
                kind='diff', evalset=('test', 'train'), sampler='gd', T=T, N=N, metric='loss'
            )
            gg_mean.append(Y[0])
            gg_stdv.append(S[0])

            (X, Y, S) = OL.query_eta_curve(
                kind='main', evalset='test', sampler='gd', T=T, N=N, metric='loss'
            )
            tl_mean.append(Y[0])
            tl_stdv.append(S[0])

        eta = X[0]

        gg_mean = np.array(gg_mean)
        gg_stdv = np.array(gg_stdv)
        tl_mean = np.array(tl_mean)
        tl_stdv = np.array(tl_stdv)

        plot_bars(hesses, tl_mean, tl_stdv, color=col, label='experiment T={}'.format(T))

        predictions = 0.5*(1.0 - np.exp(-T*eta*interp_hesses))**2/(N*interp_hesses)
        plot_fill(
            interp_hesses, predictions, 0.0*interp_hesses,
            color=col, label='deg 2 prediction, renorm'
        )

        #predictions = 0.5 * T*eta* (T*eta*interp_hesses)/N
        #plot_fill(
        #    interp_hesses, predictions, 0.0*interp_hesses,
        #    color=black, label=None
        #)

    ih = interp_hesses[int(0.23*len(interp_hesses)):]
    predictions = 0.5/(N*ih)
    plot_fill(
        ih, predictions, 0.0*ih,
        color=black, label='Takeuchi Information'
    )


    print(CC+'@R rendering plot @D ...')
    finish_plot(
        title=title, xlabel='hessian eigenvalue',
        ylabel='test loss - min',
        #img_filenm='../plots/tak.png',
        img_filenm='../plots/new-tak.png',
        ymin=(0.0), ymax=(0.05),
    )


def plot_test_vs_hess(model_nm, N, mu=10.0):
    title = (
        'STIC REGULARIZATION HELPS IN FLAT DIRECTIONS \n'
        '(test loss ratio on {})'.format(model_nm)
    )
 
    prime_plot()

    hesses = ([]
        + [10**(-4.0), 1.5*10**(-4), 2.5*10**(-4), 4.0*10**(-4), 6.5*10**(-4)]
        + [10**(-3.0), 1.5*10**(-3), 2.5*10**(-3), 4.0*10**(-3), 6.5*10**(-3)]
        + [10**(-2.0), 1.5*10**(-2), 2.5*10**(-2), 4.0*10**(-2), 6.5*10**(-2)]
        + [10**(-1.0), 1.5*10**(-1), 2.5*10**(-1), 4.0*10**(-1), 6.5*10**(-1)]
        + [10**( 0.0), 1.5*10**( 0), 2.5*10**( 0), 4.0*10**( 0), 6.5*10**( 0)]
        + [10**( 1.0)] 
    )

    interp_hesses = interpolate(hesses)

    for T, col in [(1000, blue)]:#, (20, red)]:
        etas = []
        gd_mean = []
        gd_stdv = []
        gds_vs_gd_mean = []
        gds_vs_gd_stdv = []
        gdt_vs_gd_mean = []
        gdt_vs_gd_stdv = []
        for hess in hesses:
            ol_nm = '../quad-logs/ol-quad-1d-reg-h{:0.4f}-m{:0.2f}'.format(hess, mu)

            OL = OptimLog(ol_nm)
            OL.load_from(ol_nm)

            (X, Y, S) = OL.query_eta_curve(
                kind='main', evalset='test', sampler='gd', T=T, N=N, metric='real-loss'
            )
            gd_mean.append(Y[0])
            gd_stdv.append(S[0])

            (X, Y, S) = OL.query_eta_curve(
                kind='diff', evalset='test', sampler=('gds', 'gd'), T=T, N=N, metric='real-loss'
                #kind='main', evalset='test', sampler='gds', T=T, N=N, metric='expl'
            )
            gds_vs_gd_mean.append(Y[0])
            gds_vs_gd_stdv.append(S[0])

            (X, Y, S) = OL.query_eta_curve(
                kind='diff', evalset='test', sampler=('gdt', 'gd'), T=T, N=N, metric='real-loss'
                #kind='main', evalset='test', sampler='gds', T=T, N=N, metric='expl'
            )
            gdt_vs_gd_mean.append(Y[0] if Y[0] < 1e9 else 1e9)
            gdt_vs_gd_stdv.append(S[0] if Y[0] < 1e9 else 0)

        eta = X[0]
        gd_mean = np.array(gd_mean)
        gd_stdv = np.array(gd_stdv)
        gds_vs_gd_mean = np.array(gds_vs_gd_mean)
        gds_vs_gd_stdv = np.array(gds_vs_gd_stdv)
        gdt_vs_gd_mean = np.array(gdt_vs_gd_mean)
        gdt_vs_gd_stdv = np.array(gdt_vs_gd_stdv)

        plot_bars(
            np.log(hesses), 1.0 + gds_vs_gd_mean/gd_mean, gds_vs_gd_stdv/gd_mean, color=blue,
            label='gds vs gd, eta*T={}'.format(eta*T)
        )
        #plot_bars(
        #    np.log(hesses), 1.0 + gdt_vs_gd_mean/gd_mean, gdt_vs_gd_stdv/gd_mean, color=red,
        #    label='gdt vs gd, T={}'.format(T)
        #)

    plot_fill(
        np.log(interp_hesses), 1.0 + 0.0*interp_hesses, 0.0*interp_hesses,
        color=black, label=None
    )

    print(CC+'@R rendering plot @D ...')
    finish_plot(
        title=title, xlabel='log hessian eigenvalue',
        ylabel='test loss ratio', img_filenm='../quad-logs/tak-reg.png',
        #ymin=(-100.0),ymax=(500.0)
        ymin=(0.0), ymax=(2.0),
        #ymin=(-1.0), ymax=(0.5),
    )


def plot_batch_match_loss_vs_loss(idxs_and_model_nms, T):
    title = (
        'GDC Mimics Small-Batch behavior \n'
        '(excess test loss over SGD\'s. \n'
        ' axes scaled equally)'
        .format(T)
    )

    plt.rcParams.update({'font.size': 14})
    for (idxs, model_nm), col in zip(idxs_and_model_nms, [green, red, blue, yellow]):
        sgd_true_mean = [] 
        sgd_true_stdv = [] 
        gd_sgd_true_mean = [] 
        gd_sgd_true_stdv = [] 
        gd_sgd_pred_mean = [] 
        gd_sgd_pred_stdv = [] 
        gdc_sgd_true_mean = [] 
        gdc_sgd_true_stdv = [] 
        gdc_sgd_pred_mean = [] 
        gdc_sgd_pred_stdv = [] 

        for idx in idxs:
            ol_nm  = 'ol-{}-T{}-{:02}-bm.data'.format(model_nm, T, idx)
            OL = OptimLog(ol_nm)
            OL.load_from(ol_nm)

            (X, Y, S) = OL.query_eta_curve(
                kind='main', evalset='test', sampler='sgd', T=T, N=T, metric='loss'
            )
            sgd_true_mean += list(Y)
            sgd_true_stdv += list(S)

            (X, Y, S) = OL.query_eta_curve(
                kind='diff', evalset='test', sampler=('gd', 'sgd'), T=T, N=T, metric='loss'
            )
            gd_sgd_true_mean += list(Y)
            gd_sgd_true_stdv += list(S)

            (X, Y, S) = OL.query_eta_curve(
                kind='diff', evalset='test', sampler=('gdc', 'sgd'), T=T, N=T, metric='loss'
            )
            gdc_sgd_true_mean += list(Y)
            gdc_sgd_true_stdv += list(S)

            gs_nm  = '../logs/gs-{}-{:02}.data'.format(model_nm, idx)
            P = Predictor(gs_nm)
            losses = P.evaluate_expr(
                P.extrapolate_from_taylor(
                    coeff_strs=coefficients.gd_minus_sgd_vanilla_test, degree=2, mode='poly'
                ),
                params = {'T':T, 'eta':X, 'e':np.exp(1), 'N':T}
            )
            gd_sgd_pred_mean += list(losses['mean'])
            gd_sgd_pred_stdv += list(losses['stdv'])

            gdc_sgd_pred_mean += list([0])
            gdc_sgd_pred_stdv += list([0])

        sgd_true_mean     = np.array(    sgd_true_mean) 
        sgd_true_stdv     = np.array(    sgd_true_stdv)
        gd_sgd_true_mean  = np.array( gd_sgd_true_mean) 
        gd_sgd_true_stdv  = np.array( gd_sgd_true_stdv)
        gd_sgd_pred_mean  = np.array( gd_sgd_pred_mean)
        gd_sgd_pred_stdv  = np.array( gd_sgd_pred_stdv)
        gdc_sgd_true_mean = np.array(gdc_sgd_true_mean)
        gdc_sgd_true_stdv = np.array(gdc_sgd_true_stdv)
        gdc_sgd_pred_mean = np.array(gdc_sgd_pred_mean)
        gdc_sgd_pred_stdv = np.array(gdc_sgd_pred_stdv)

        plot_plus(
            gd_sgd_true_mean, gd_sgd_true_stdv,
            gdc_sgd_true_mean, gdc_sgd_true_stdv,
            color=col,
            label={
                'fashion-lenet':'Fashion-MNIST Convnet',
                'fashion-logistic':'Fashion-MNIST Logistic',
                'cifar-lenet':'CIFAR-10 Convnet',
                'cifar-logistic':'CIFAR-10 Logistic',
                }[model_nm]
        )
        #plot_plus(
        #    sgd_true_mean, sgd_true_stdv,
        #    gdc_sgd_true_mean, gdc_sgd_true_stdv,
        #    color=green, label='gdc vs sgd'
        #)
        #plot_plus(
        #    sgd_true_mean, sgd_true_stdv,
        #    gd_sgd_true_mean, gd_sgd_true_stdv,
        #    color=red, label='gd vs sgd'
        #)
        plot_fill(
            gd_sgd_true_mean,
            0.0*gd_sgd_true_mean,
            0.0*gd_sgd_true_mean,
            color=black, label=None
        )

    img_nm = '../plots/_bm-{}-{}.png'.format(model_nm, idx)
    print(CC+'@R rendering plot @D ...')
    finish_plot(
        title=title, xlabel='GD without regularizer', ylabel='GDC',
        #img_filenm='../plots/big-bm-new.png',
        img_filenm='../plots/new-big-bm-new.png',
        ymin=-1.7*10**-4, ymax=+1.7*10**-4,
    )

#for idx in range(6):
#    plot_test_loss_vs_eta('fashion-lenet', idx, 10)
#plot_test_vs_hess('quad-1d-reg', N=10)
#plot_gengap_vs_hess('quad-1d', T=10)

#plot_batch_match_loss_vs_loss(['cifar-lenet', 'fashion-lenet'], idxs=range(0,6), T=10)
plot_batch_match_loss_vs_loss([
    (range(0, 6), 'fashion-lenet'),
    (range(0, 6), 'cifar-lenet'),
    (range(0, 1), 'fashion-logistic'),
    (range(0, 1), 'cifar-logistic'),
], T=10)

#plot_gen_gap_loss_vs_loss('cifar-lenet', [0, 1, 2, 3, 4, 5], 10)

#for model_nm in ['cifar-lenet', 'fashion-lenet']:
#    for idx in range(0,6):
#        plot_gen_gap_loss_vs_eta(model_nm, idx=idx, T=10)
#        #plot_batch_match_loss_vs_eta(model_nm, idx=idx, T=10)

#plot_test()
#plot_gauss_nongauss_vs_eta('fitgauss', idx=0, T=4)

#plt.figure(figsize=(8,4))
#plot_gauss_nongauss_vs_eta('cubicchi', idx=0, T=4)
#plot_thermo_vs_eta('linear-screw', idx=0, T=10000)


#plot_sgd_sde_diff_vs_eta('fitgauss', idx=0, T=1)
#plot_multi_vs_eta('cifar-logistic', idx=0)
