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

def get_mean(gradstats, grad_stat_names):
    return (lambda english:
        gradstats[grad_stat_names[english]]['mean']
    )

def get_stdv(gradstats, grad_stat_names):
    return (lambda english:
        gradstats[grad_stat_names[english]]['stdv']
        / gradstats[grad_stat_names[english]]['nb_samples']**0.5
    )

def from_string(gradstats, formula, eta, T, N, M=1, B=1):
    Y = formula[:]
    S = formula.replace('- ', '+ ').replace(' -', ' +')
    for english_nm in grad_stat_names:
        Y = Y.replace(k, 'mean({})'.format(k))
        S = S.replace(k, 'stdv({})'.format(k))
    return tuple(eval(form, {
        'eta':eta, 'T':T, 'M':M, 'N':N,
        'mean': get_mean(gradstats, grad_stat_names),
        'stdv': get_stdv(gradstats, grad_stat_names),
        'np':np, 'choose':choose
    }) for form in (Y, S))

sgd_test_coeffs = {
    0: '+ (     Dpln )',
    1: '- ( T:1 Ipln )',
    2: '+ ( T:2 Vpln   +   T:1 Vlvs / 2! )',
    3: '- ( T:3 (4 Zpln + 2 Tpln)   +   T:2 (1.5 Tlvs + Zlvs + Zalt)   +   T:1 (Terg / 3!) )',
}

MP = MathParser()
vals_by_name = {
    'T':3,
    'Dpln':1.0,
    'Ipln':1.0,
    'Vpln':1.0,
    'Vlvs':1.0,
    'Zpln':1.0,
    'Tpln':1.0,
    'Tlvs':1.0,
    'Zlvs':1.0,
    'Zalt':1.0,
    'Terg':1.0,
}
print(MP.eval(sgd_test_coeffs[3], vals_by_name))

#sgd_test_coeffs_gauss = (
#    '(+ ( ()(0) ) )',
#    '(- (choose(T, 1) * (01)(0-1)) )',
#    '(+ (choose(T, 2) * 2*(01-02)(0-1-2) + choose(T, 1) * 0.5*(01-02)(0-12)) )',
#    '(- (choose(T, 3) * (4*(01-02-13)(0-1-2-3) + 2*(01-02-03)(0-1-2-3)) + choose(T, 2) * (1.5 * (01-02-03)(0-1-23) + (01-02-13)(0-1-23) + (01-02-13)(0-12-3)) + choose(T, 1) * ((1.0/6) * (3 * (01-02-03)(0-1-23) - 2 * (01-02-03)(0-1-2-3)))) )',
#)
#
#sgd_test_multiepoch_coeffs = (
#    '(+ ( ()(0) ) )',
#    '(- (choose(E, 1) * choose(T, 1) * (01)(0-1)) )',
#    '(+ ((2 * choose(E, 2) + choose(E, 1)) * choose(T, 2) * 2*(01-02)(0-1-2) + choose(E, 2)*choose(T, 1)*((01-02)(0-12) + (01-02)(01-2)) + choose(E, 1)*choose(T, 1)*0.5*(01-02)(0-12)) )',
#    '(- ( ({}) ) )'.format(') + ('.join((
#        'choose(E, 3)*6*choose(T, 3) * ( (01-02-13)(0-1-2-3)  +  (01-02-03)(0-1-2-3)  +  (01-02-13)(0-1-2-3)  +  (01-02-13)(0-1-2-3)  +  (01-02-13)(0-1-2-3)  +  (01-02-03)(0-1-2-3) )',
#        'choose(E, 3)*2*choose(T, 2) * (  (01-02-13)(0-13-2)  +   (01-02-03)(0-1-23)  +   (01-02-13)(0-1-23)  +   (01-02-13)(0-13-2)  +   (01-02-13)(0-1-23)  +   (01-02-03)(0-1-23) )',
#        'choose(E, 3)*2*choose(T, 2) * (  (01-02-13)(0-12-3)  +   (01-02-03)(01-2-3)  +   (01-02-13)(0-12-3)  +   (01-02-13)(0-1-23)  +   (01-02-13)(0-12-3)  +   (01-02-03)(0-1-23) )',
#        'choose(E, 3)*2*choose(T, 2) * (  (01-02-13)(01-2-3)  +   (01-02-03)(01-2-3)  +   (01-02-13)(0-1-23)  +   (01-02-13)(0-12-3)  +   (01-02-13)(0-12-3)  +   (01-02-03)(0-1-23) )',
#        'choose(E, 3) * choose(T, 1) * (   (01-02-13)(012-3)  +    (01-02-03)(012-3)  +    (01-02-13)(0-123)  +    (01-02-13)(0-123)  +    (01-02-13)(0-123)  +    (01-02-03)(0-123) )',
#        'choose(E, 2)*3*choose(T, 3) * ( (01-02-13)(0-1-2-3)  +  (01-02-03)(0-1-2-3)  +  (01-02-13)(0-1-2-3)  +  (01-02-13)(0-1-2-3)  +  (01-02-13)(0-1-2-3)  +  (01-02-03)(0-1-2-3) )',
#        'choose(E, 2) * choose(T, 2) * (  (01-02-13)(0-12-3)  +   (01-02-03)(01-2-3)  +   (01-02-13)(0-12-3)  +   (01-02-13)(0-1-23)  +   (01-02-13)(0-12-3)  +   (01-02-03)(0-1-23) )',
#        'choose(E, 2) * choose(T, 2) * (  (01-02-13)(01-2-3)  +   (01-02-03)(01-2-3)  +   (01-02-13)(0-1-23)  +   (01-02-13)(0-12-3)  +   (01-02-13)(0-12-3)  +   (01-02-03)(0-1-23) )',
#        'choose(E, 2)*3*choose(T, 3) * ( (01-02-13)(0-1-2-3)  +  (01-02-03)(0-1-2-3)  +  (01-02-13)(0-1-2-3)  +  (01-02-13)(0-1-2-3)  +  (01-02-13)(0-1-2-3)  +  (01-02-03)(0-1-2-3) )',
#        'choose(E, 2) * choose(T, 2) * (  (01-02-13)(0-13-2)  +   (01-02-03)(0-1-23)  +   (01-02-13)(0-1-23)  +   (01-02-13)(0-13-2)  +   (01-02-13)(0-1-23)  +   (01-02-03)(0-1-23) )',
#        'choose(E, 2) * choose(T, 2) * (  (01-02-13)(0-12-3)  +   (01-02-03)(01-2-3)  +   (01-02-13)(0-12-3)  +   (01-02-13)(0-1-23)  +   (01-02-13)(0-12-3)  +   (01-02-03)(0-1-23) )',
#        'choose(E, 1) * choose(T, 3) * ( (01-02-13)(0-1-2-3)  +  (01-02-03)(0-1-2-3)  +  (01-02-13)(0-1-2-3)  +  (01-02-13)(0-1-2-3)  +  (01-02-13)(0-1-2-3)  +  (01-02-03)(0-1-2-3) )',
#        'choose(E, 2)*2*choose(T, 2) * (                      0.5*(01-02-03)(0-12-3) +0.5*(01-02-13)(0-1-23)  +                       0.5*(01-02-13)(0-1-23) +0.5*(01-02-03)(0-1-23) )',
#        'choose(E, 2)*2*choose(T, 2) * (                       0.5*(01-02-03)(0-123)  +0.5*(01-02-13)(0-123)  +                       0.5*(01-02-13)(0-1-23) + 0.5*(01-02-03)(0-123) )',
#        'choose(E, 2)*2*choose(T, 2) * (                                                                      0.5*(01-02-13)(0-12-3) +0.5*(01-02-13)(0-12-3) +0.5*(01-02-03)(0-1-23) )',
#        'choose(E, 2)*2*choose(T, 2) * (                                                                       0.5*(01-02-13)(0-123) + 0.5*(01-02-13)(0-123)  +0.5*(01-02-03)(0-123) )',
#        'choose(E, 1) * choose(T, 2) * (                      0.5*(01-02-03)(0-12-3) +0.5*(01-02-13)(0-1-23)  +                       0.5*(01-02-13)(0-1-23) +0.5*(01-02-03)(0-1-23) )',
#        'choose(E, 1) * choose(T, 2) * (                                                                      0.5*(01-02-13)(0-12-3) +0.5*(01-02-13)(0-12-3) +0.5*(01-02-03)(0-1-23) )',
#        'choose(E, 1) * choose(T, 1) * (                                                                                                                     1.0/6*(01-02-03)(012-3) )',
#    )))
#)
#
#sgd_gen_coeffs = (
#    '(- (0.0 ) )',
#    '(+ (choose(T, 1) * ((01)(01) - (01)(0-1)) )/N )',
#    '(- (choose(T, 2) * (3*(01-02)(01-2) + (01-02)(0-12) - 4*(01-02)(0-1-2)) + choose(T, 1)*0.5*((01-02)(012) - (01-02)(0-12)) )/N )',
#    '(+ (choose(T, 3) * (4*(01-02-13)(0-13-2)+5*(01-02-13)(0-12-3)+2*(01-02-13)(01-2-3)+1*(01-02-13)(0-1-23)-12*(01-02-13)(0-1-2-3) +4*(01-02-03)(01-2-3)+2*(01-02-03)(0-1-23)-6*(01-02-03)(0-1-2-3)) + choose(T, 2) * (1*(01-02-03)(012-3) + 1.5*(01-02-03)(01-23) + 0.5*(01-02-03)(0-123) -3*(01-02-03)(0-1-23) + 1*(01-02-13)(01-23) + 1*(01-02-13)(012-3) -2*(01-02-13)(0-1-23) + 1*(01-02-13)(012-3) + 1*(01-02-13)(02-13) -2*(01-02-13)(0-12-3)) + choose(T, 1) * (1.0/6 * (01-02-03)(0123) - 1.0/6 * (01-02-03)(0-123)) )/N)'
#)
#
#sgd_gd_diff_coeffs = (
#    '(+ (0.0 ) )',
#    '(- (0.0) )',
#    '(+ (choose(T, 2) * ( (01-02)(01-2) - (01-02)(0-1-2) )) / N )',
##    '(- (choose(T, 3) * ( 3*(01-02-13)(0-1-23) + 5*(01-02-13)(0-12-3) + 3*(01-02-13)(0-13-2) + 1*(01-02-13)(01-2-3) - 12*(01-02-13)(0-1-2-3) + 4*(01-02-03)(0-1-23) + 2*(01-02-03)(01-2-3) - 6*(01-02-03)(0-1-2-3) )) / N )',
#)
#
#
#
#def sgd_test_batchmatch_diff(gradstats, eta, T, degree):
#    pass
#
#def sgd_gd_diff(gradstats, eta, T, N, degree):
#    assert 1 <= degree, 'need strictly positive degree of approximation!'
#    formula = ' + '.join('eta*'*d + sgd_gd_diff_coeffs[d] for d in range(degree+1))
#    print(formula)
#    Y, S = from_string(gradstats, formula, eta, T)
#    return Y, S
#
#def sgd_test_multiepoch_diff_e2h2(gradstats, eta, T, degree, E=None):
#    assert 2 == degree, 'i only know the answer for degree 2!'
#    formula = 'eta*eta * choose(E, 2) * choose(T, 1) * (01-02)(01-2)'
#    print(formula)
#    Y, S = from_string(gradstats, formula, eta, T, E)
#    return Y, S
#
#def sgd_gen(gradstats, eta, T, degree):
#    assert 1 <= degree, 'need strictly positive degree of approximation!'
#    formula = ' + '.join('eta*'*d + sgd_gen_coeffs[d] for d in range(degree+1))
#    print(formula)
#    Y, S = from_string(gradstats, formula, eta, T)
#    return Y, S
#
#def sgd_test_multiepoch(gradstats, eta, T, degree, E=None):
#    assert 1 <= degree, 'need strictly positive degree of approximation!'
#    formula = ' + '.join('eta*'*d + sgd_test_multiepoch_coeffs[d] for d in range(degree+1))
#    print(formula)
#    Y, S = from_string(gradstats, formula, eta, T, E)
#    return Y, S
#
#def sgd_test_taylor(gradstats, eta, T, degree):
#    assert 1 <= degree, 'need strictly positive degree of approximation!'
#    formula = ' + '.join('eta*'*d + sgd_test_coeffs[d] for d in range(degree+1))
#    Y, S = from_string(gradstats, formula, eta, T)
#    return Y, S
#
#def sgd_test_taylor_gauss(gradstats, eta, T, degree):
#    assert 1 <= degree, 'need strictly positive degree of approximation!'
#    formula = ' + '.join('eta*'*d + sgd_test_coeffs_gauss[d] for d in range(degree+1))
#    Y, S = from_string(gradstats, formula, eta, T)
#    return Y, S
#
#def sgd_test_exponential(gradstats, eta, T, degree):
#    if degree==2:
#        cs = [from_string(gradstats, sgd_test_coeffs[d], None, T) for d in range(3)]
#        def get_Y2(cs0, cs1, cs2):
#            rate = -2*cs2/cs1
#            scale = cs1**2 / (2*cs2)
#            offset = cs0 - cs1**2/(2*cs2)
#
#            formula = '{} * np.exp(- {} * eta) + {}'.format(scale, rate, offset)
#            Y, _ = from_string(gradstats, formula, eta, T)
#            return Y
#
#        Ys = []
#        for i in range(100):
#            z = [(np.random.random()*2-1) for j in (0,1,2)]
#            Ys.append(get_Y2(*(cs[j][0]+z[j]*cs[j][1] for j in (0,1,2))))
#        Ys = np.array(Ys)  
#        Y, S = Ys.mean(axis=0), Ys.ptp(axis=0)/2
#    elif degree==3:
#        cs = [from_string(gradstats, sgd_test_coeffs[d], None, T) for d in range(4)]
#
#        def get_Y3(cs0, cs1, cs2, cs3):
#            rate = abs(3 * (cs2/cs1)**2 - 2 * (cs3/cs1))**0.5
#            scale = (1.0/2 + (cs2/cs1)/(2*rate))**2 * rate / cs1  
#            shift = abs((scale*rate)/cs1)**0.5   
#            offset = cs0 - 1.0/shift
#
#            formula = '1.0 / ( {} * (np.exp(- {} * eta) - 1) + {} ) + {}'.format(scale, rate, shift, offset)
#            Y, _ = from_string(gradstats, formula, eta, T)
#            return Y
#
#        Ys = []
#        for i in range(100):
#            z = [(np.random.random()*2-1) for j in (0,1,2,3)]
#            Ys.append(get_Y3(*(cs[j][0]+z[j]*cs[j][1] for j in (0,1,2,3))))
#        Ys = np.array(Ys)  
#        Y, S = Ys.mean(axis=0), Ys.ptp(axis=0)/2 
#
#    return Y, S
#
#def sgd_test_multiepoch_exponential(gradstats, eta, T, degree, E=None):
#    if degree==2:
#        cs = [from_string(gradstats, sgd_test_multiepoch_coeffs[d], None, T) for d in range(3)]
#        def get_Y2(cs0, cs1, cs2):
#            rate = -2*cs2/cs1
#            scale = cs1**2 / (2*cs2)
#            offset = cs0 - cs1**2/(2*cs2)
#
#            formula = '{} * np.exp(- {} * eta) + {}'.format(scale, rate, offset)
#            Y, _ = from_string(gradstats, formula, eta, T, E=E)
#            return Y
#
#        Ys = []
#        for i in range(100):
#            z = [(np.random.random()*2-1) for j in (0,1,2)]
#            Ys.append(get_Y2(*(cs[j][0]+z[j]*cs[j][1] for j in (0,1,2))))
#        Ys = np.array(Ys)  
#        Y, S = Ys.mean(axis=0), Ys.ptp(axis=0)/2
#    elif degree==3:
#        cs = [from_string(gradstats, sgd_test_multiepoch_coeffs[d], None, T) for d in range(4)]
#
#        def get_Y3(cs0, cs1, cs2, cs3):
#            rate = abs(3 * (cs2/cs1)**2 - 2 * (cs3/cs1))**0.5
#            scale = (1.0/2 + (cs2/cs1)/(2*rate))**2 * rate / cs1  
#            shift = abs((scale*rate)/cs1)**0.5   
#            offset = cs0 - 1.0/shift
#
#            formula = '1.0 / ( {} * (np.exp(- {} * eta) - 1) + {} ) + {}'.format(scale, rate, shift, offset)
#            Y, _ = from_string(gradstats, formula, eta, T, E=E)
#            return Y
#
#        Ys = []
#        for i in range(100):
#            z = [(np.random.random()*2-1) for j in (0,1,2,3)]
#            Ys.append(get_Y3(*(cs[j][0]+z[j]*cs[j][1] for j in (0,1,2,3))))
#        Ys = np.array(Ys)  
#        Y, S = Ys.mean(axis=0), Ys.ptp(axis=0)/2 
#
#    return Y, S
#
