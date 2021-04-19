from random import random
import numpy as np
 

# outputs:
#
#       h= 0.01
#       ......................................................................................................................................................
#       h=0.010, t=   0:    +0.5000(0.0000) - +0.5000(0.0000) = +0.0000(0.0000)
#       h=0.010, t=  10:    +0.5001(0.0000) - +0.5001(0.0000) = -0.0000(0.0000)
#       h=0.010, t=  18:    +0.5005(0.0000) - +0.5004(0.0000) = +0.0001(0.0001)
#       h=0.010, t=  32:    +0.5021(0.0002) - +0.5009(0.0001) = +0.0011(0.0003)
#       h=0.010, t=  56:    +0.5071(0.0006) - +0.5018(0.0002) = +0.0052(0.0008)
#       h=0.010, t= 100:    +0.5211(0.0019) - +0.5026(0.0003) = +0.0185(0.0022)
#       h=0.010, t= 180:    +0.5511(0.0049) - +0.5027(0.0003) = +0.0485(0.0052)
#       h=0.010, t= 320:    +0.5965(0.0107) - +0.5034(0.0003) = +0.0931(0.0110)
#       h=0.010, t= 560:    +0.6473(0.0200) - +0.5035(0.0004) = +0.1439(0.0204)
#       h=0.010, t=1000:    +0.6783(0.0274) - +0.5037(0.0004) = +0.1746(0.0278)
#       
# c.f. https://www.desmos.com/calculator/kvkjbnffi6
#      https://www.desmos.com/calculator/k8p3kmegbs#
##
# here, the weightspace parameterizes gaussians: 
#
#       th = [mean, log precision]

def grad(th, x):
    #x = np.random.randn()
    m,p = th[0],th[1]
    return np.array([(m-x)*np.exp(p),-0.5*(1-(m-x)**2*np.exp(p))])
def loss(th):
    m,p = th[0],th[1]
    return 0.5 * ((m**2 + 1)*np.exp(p) - p)

def sde_grad(th, K): 
    x = np.random.randn()
    m,p = th[0],th[1]
    return (
        np.array([m*np.exp(p),-0.5*(1-(m**2+1)*np.exp(p))])
        +
        np.sqrt(K) * x * np.exp(p) * np.sqrt(np.array([1.0, 0.5 + m**2])) 
    )

hs = [0.1]#[0.001, 0.003, 0.010, 0.030] 
ts = [1,2,3,4,5]#[0,1,2,3,4,5,6,7,8,9,10]#[0, 10, 18, 32, 56, 100]#, 180, 320, 560, 1000]
tts = set(ts)

samples_gd = {
    (h,t):[]
    for h in hs for t in ts 
}
samples_de = {
    (h,t):[]
    for h in hs for t in ts 
}

its = 50000
K = 1#0
N = 10#10
for h in hs:
    print('\nh=',h)
    for it in range(its):
        if it%100==0:
            print('.', end='', flush=True)
        th_gd = np.array([0.0, 0.0]);  th_de = np.array([0.0, 0.0])
        ls_gd = loss(th_gd)         ;  ls_de = loss(th_de) 
        xs = [np.random.randn() for n in range(N)] 
        for t in range(max(ts)+1):
            if t in tts:
                samples_gd[(h,t)].append(ls_gd)
                samples_de[(h,t)].append(ls_de)
            #
            th_gd -= h * grad(th_gd, xs[(t%N)])
            ls_gd += 0.05 * (loss(th_gd)-ls_gd)
            #
            for k in range(K):
                th_de -= (h/K) * sde_grad(th_de,K) 
            ls_de += 0.05 * (loss(th_de)-ls_de)

for h in hs:
    print()
    for t in ts:
        ls_gd = np.mean(samples_gd[(h,t)]) 
        #
        ss_gd = np.std(samples_gd[(h,t)])/np.sqrt(its-1)
        ls_de = np.mean(samples_de[(h,t)]) 
        ss_de = np.std(samples_de[(h,t)])/np.sqrt(its-1) 
        print('h={:.3f}, t={:4d}:   '.format(h,t),
                '{:+.5f}({:.5f}) - {:+.5f}({:.5f}) = {:+.5f}({:.5f})'.format(
                    ls_gd, ss_gd,
                    #0,0,0,0))
                    ls_de, ss_de,
                    ls_gd-ls_de, ss_gd+ss_de))


#h= 0.01
#........................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................
#h=0.010, t=   0:    +0.50000(0.00000) - +0.00000(0.00000) = +0.00000(0.00000)
#h=0.010, t=  10:    +0.50014(0.00000) - +0.00000(0.00000) = +0.00000(0.00000)
#h=0.010, t=  18:    +0.50052(0.00002) - +0.00000(0.00000) = +0.00000(0.00000)
#h=0.010, t=  32:    +0.50204(0.00007) - +0.00000(0.00000) = +0.00000(0.00000)
