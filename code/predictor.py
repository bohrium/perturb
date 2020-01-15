''' author: samtenka
    change: 2020-01-15
    create: 2019-06-18
    descrp: Combine Taylor coefficients with measured gradient statistics to
            fit a 'loss vs eta' curve with appropriately nontrivial error bars. 
'''

import sys 
from utils import prod, pre, CC
from gradstats import grad_stat_names 
from parser import MathParser
import numpy as np

class Predictor(object): 
    def __init__(self, gs_name='gs-fashion-lenet-00.data'):
        self.MP = MathParser()
        with open(gs_name) as f:
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
            results.append(self.MP.eval(expr_str, vals_by_name))
        return {
            'mean': np.mean(results, axis=0),
            'stdv': np.std(results, axis=0) * np.sqrt(I/(I-1.0)) 
        }

    def extrapolate_from_taylor(self, coeff_strs, degree, mode='poly'):
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
                'eta^{} ({})'.format(d, coeff_strs[d])
                for d in range(degree+1)
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
        elif mode=='ode' and degree==3:
            formula = (
                '1 / ( scale (e^(- rate eta) - 1) + shift ) + offset'
                .replace('offset', '(cs0 - 1/shift)')
                .replace('shift',  '(((scale rate/cs1)^2)^0.25)')
                .replace('scale',  '((1/2 + cs2/(2 cs1 rate))^2 (rate / cs1))')
                .replace('rate',   '(((3 (cs2/cs1)^2 - 2 (cs3/cs1))^2)^0.25)')
                .replace('cs0', '({})'.format(coeff_strs[0])) 
                .replace('cs1', '({})'.format(coeff_strs[1])) 
                .replace('cs2', '({})'.format(coeff_strs[2])) 
                .replace('cs3', '({})'.format(coeff_strs[3])) 
            )
            print(coeff_strs[3])

        return formula

if __name__=='__main__':
    P = Predictor()
    for mode in ('poly', 'ode'):
        for degree in range(4):
            print(CC + 'mode @R {} @D and degree @Y {} @D '.format(
                mode, degree
            ))
            print(CC + '@W {} @D '.format(
                P.extrapolate_from_taylor('ABCD', degree, mode)
            ))
            print()
