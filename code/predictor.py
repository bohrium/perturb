''' author: samtenka
    change: 2020-01-15
    create: 2019-06-18
    descrp: predict losses based on diagram values
'''

from matplotlib import pyplot as plt
import numpy as np
from optimlogs import OptimKey
#import solver
import sys 
from utils import prod
from gradstats import grad_stat_names 
from parser import MathParser

sgd_vanilla_test_coeffs = {
    0: '+     Dpln',
    1: '- T:1 Ipln',
    2: '+ T:2 Vpln' 
       '+ T:1 Vlvs / 2!',
    3: '- T:3 (4 Zpln + 2 Ypln)' 
       '- T:2 (1.5 Ylvs + Zlvs + Zalt)'
       '- T:1 Yerg / 3!) )',
}

sgd_vanilla_gauss_test_coeffs = {
    0: '+     Dpln',
    1: '- T:1 Ipln',
    2: '+ T:2 Vpln' 
       '+ T:1 Vlvs / 2!',
    3: '- T:3 (4 Zpln + 2 Ypln)' 
       '- T:2 (1.5 Tlvs + Zlvs + Zalt)'
       '- T:1 (3 Ylvs - 2 Ypln) / 3!',
}

sgd_vanilla_gen_coeffs = {
    0: '-            0.0 ',
    1: '+ (T:1 / N) (  Iall -   Ipln)',
    2: '- (T:2 / N) (3 Vtwg +   Vlvs - 4 Vpln)' 
       '- (T:1 / N) (  Vall -   Vlvs) / 2!',
    3: '+ (T:3 / N) (4 Ztwg + 5 Zalt + 2 Zmid + 1 Zlvs - 12 Zpln)'
       '+ (T:3 / N) (4 Ytwg + 2 Ylvs - 6 Ypln)' 
       '+ (T:2 / N) (1 Yvee + 1.5 Ysli + 0.5 Ylvs - 3 Ylvs)'
       '+ (T:2 / N) (1 Ysli + 1 Yvee - 2 Ylvs)'
       '+ (T:2 / N) (1 Yvee + 1 Ysli - 2 Ytwg)'
       '+ (T:1 / N) (Yall - Yerg) / 3!',
}

gd_vanilla_test_minus_sgd_vanilla_test_coeffs = {
    0: '+                0.0 ',
    1: '-                0.0 ',
    2: '+ ((N-1)/2) T:2 (Ytwg - Ypln)' 
}

class Predictor(object): 
    def __init__(self, gs_name):
        self.MP = MathParser()
        with open('gs-cifar-lenet-00.data') as f:
            self.gradstats = eval(f.read())

    def get_mean(self, english):
        return (
            self.gradstats[grad_stat_names[english]]['mean']
        )
    
    def get_stdv(self, english):
        return (
            self.gradstats[grad_stat_names[english]]['stdv']
            / self.gradstats[grad_stat_names[english]]['nb_samples']**0.5
        )

    def evaluate_expr(self, expr_str, params, I=1000, seed=0):
        results = []
        for i in range(I):
            vals_by_name = {k:params[k] for k in params}
            for english in grad_stat_names:
                vals_by_name[english] = (
                    self.get_mean(english) +
                    np.random.randn() * self.get_stdv(english)
                )
            results.append(MP.eval(coeff_str, vals_by_name))
        return {
            'mean': np.mean(results),
            'stdv': np.stdv(results) * np.sqrt(I/(I-1.0)) 
        }

    def extrapolate_from_taylor(coeff_strs, degree, mode='poly'):
        '''
            Given expressions for coeffs, returns an expression involving eta
            that evaluates to a function having the specified Taylor
            coefficients in eta.  The two available extrapolation modes are
            'poly' and 'ode'. 
        '''
        pre(mode in ('poly', 'ode'),
            'mode should be poly or ode!'
        )

        if mode=='poly' or degree < 2: 
            formula = ' + '.join(
                'eta^{} {}'.format(d, c)
                for d, c in enumerate(coeff_strs)
            ) 
        elif mode=='ode' and degree==2:
            formula = (
                'scale e^(- rate eta) + offset'
                    .replace('rate',   '(-2 cs2/cs1)')
                    .replace('scale',  '(cs1^2 / (2 cs2))')
                    .replace('offset', '(cs0 - cs1^2/(2 cs2))')
                    .replace('cs0', '({})'.format(coeff_strs[0])) 
                    .replace('cs1', '({})'.format(coeff_strs[1])) 
                    .replace('cs2', '({})'.format(coeff_strs[2])) 
                )
            )
        elif mode=='ode' and degree==3:
            formula = (
                'scale e^(- rate eta) + offset'
                    .replace('offset', '(cs0 - 1/shift)')
                    .replace('shift',  '(((scale rate/cs1)^2)^0.25)')
                    .replace('scale',  '((1/2 + cs2/(2 cs1 rate))^2 (rate / cs1))')
                    .replace('rate',   '(((3 (cs2/cs1)^2 - 2 (cs3/cs1))^2)^0.25)')
                    .replace('cs0', '({})'.format(coeff_strs[0])) 
                    .replace('cs1', '({})'.format(coeff_strs[1])) 
                    .replace('cs2', '({})'.format(coeff_strs[2])) 
                    .replace('cs3', '({})'.format(coeff_strs[3])) 
                )
            )

        return formula


