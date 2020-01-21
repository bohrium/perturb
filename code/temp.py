import numpy as np
from utils import pre

def cross_product(v, w):
    # positive when w sits to the left of v   
    return v[0]*w[1] - v[1]*w[0]  

pre(0.0 <  cross_product((1, 0), (0, 1)), 'cross product error: pos')
pre(0.0 == cross_product((1, 0), (2, 0)), 'cross product error:zero')
pre(0.0 >  cross_product((0, 1), (1, 0)), 'cross product error: neg')

# cy for cyclic
def get_inner_edge_idxs(pt, cy_hull):
    inner_edge_idxs = []
    x, y = pt
    for i in range(len(cy_hull)):
        (ax, ay), (bx, by) = cy_hull[i], cy_hull[(i+1)%len(cy_hull)]
        pt_is_to_left = (0.0 <= cross_product(
            (bx-ax, by-ay),
            ( x-ax,  y-ay),
        )) 
        if not pt_is_to_left:
            inner_edge_idxs.append(i)
    return inner_edge_idxs

pre([] == get_inner_edge_idxs(
        (0.0, 0.0),
        [
            (+1,  0),
            ( 0, +1),
            (-1,  0),
            ( 0, -1),
        ]
    ),
    'inner edge error: empty'
)
pre([0] == get_inner_edge_idxs(
        (0.8, 0.8),
        [
            (+1,  0),
            ( 0, +1),
            (-1,  0),
            ( 0, -1),
        ]
    ),
    'inner edge error: single'
)
pre([0, 3] == get_inner_edge_idxs(
        (2.0, 0.0),
        [
            (+1,  0),
            ( 0, +1),
            (-1,  0),
            ( 0, -1),
        ]
    ),
    'inner edge error: range'
)

def update_hull(pt, cy_hull, inner_edge_idxs):
    pre(inner_edge_idxs, 'vacuous update detected!')
    new_hull=[]
    for i in range(len(cy_hull)): 
        pre_inner = ((i-1)%len(cy_hull) in inner_edge_idxs)
        post_inner= (i in inner_edge_idxs)
        if (not post_inner) and pre_inner:
            new_hull.append(pt)
        if not (post_inner and pre_inner):
            new_hull.append(cy_hull[i])
    return new_hull

def normalize_hull(cy_hull):
    ''' ensure maximum (x,y) tup is initial entry '''
    i = cy_hull.index(max(cy_hull))
    return cy_hull[i:] + cy_hull[:i]
def get_convex_hull(pts):
    pre(2<= len(pts), 'want at least 2 points')
    cy_hull = [pts[0], pts[1]]
    while True:
        for pt in pts: 
            inner_edge_idxs = get_inner_edge_idxs(pt, cy_hull)
            if not inner_edge_idxs: continue
            #print(pt, inner_edge_idxs, cy_hull)
            cy_hull = update_hull(pt, cy_hull, inner_edge_idxs)
            break
        else:
            break
    return normalize_hull(cy_hull)

'''
    . . . a .
    . b . . f
    . . g . .
    c d . . .
    . . . e .
'''
a = ( 1,  2)
b = (-1,  1)
c = (-2, -1)
d = (-1, -1)
e = ( 1, -2)
f = ( 2,  1)
g = ( 0,  0)

pts = [a,b,c,d,e,f,g]
cy_hull = [f,a,b,c,e]
pre(cy_hull==get_convex_hull(pts),
    'convex hull error!')

#
#
#
#
#
#def smart_round(a, b): 
#    '''
#        return a `round` interval [aa, bb] inside the interval [a, b]
#    '''
#    mid_high = (2*b + a)/3.0
#    mid_low  = (2*a + b)/3.0
#    for i in range(10):
#        aa = np.ceil( a * 10**i) / 10**i 
#        bb = np.floor(b * 10**i) / 10**i 
#        if aa < mid_low and mid_high < bb: break
#    return aa, bb
#
#while True:
#    a, b = input().split()
#    print(smart_round(float(a), float(b)))
