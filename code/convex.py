''' author: samtenka
    change: 2020-01-19
    create: 2020-01-19
    descrp: compute convex hull in two dimensions (for nice plotting of
            two-dimensional uncertainties) 
'''

import numpy as np
from utils import pre, CC

def cross_product(v, w):
    # positive when w sits to the left of v   
    return v[0]*w[1] - v[1]*w[0]  

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

if __name__=='__main__':
    pre(0.0 <  cross_product((1, 0), (0, 1)), 'cross product error: pos')
    pre(0.0 == cross_product((1, 0), (2, 0)), 'cross product error:zero')
    pre(0.0 >  cross_product((0, 1), (1, 0)), 'cross product error: neg')
    
    diamond = [(+1,  0), ( 0, +1), (-1,  0), ( 0, -1)]
    
    pre([] == get_inner_edge_idxs((0.0, 0.0), diamond), 'inner edge error: empty')
    pre([0] == get_inner_edge_idxs((0.8, 0.8), diamond), 'inner edge error: single')
    pre([0, 3] == get_inner_edge_idxs((2.0, 0.0), diamond), 'inner edge error: range')
    
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
    pre(cy_hull==get_convex_hull(pts), 'convex hull error: pentagon')

    # NOTE: we do not handle degenerate edges at all

    '''
        a . . . .
        . . e . d
        . . . b .
        . . . . .
        . . . . c
    '''
    a = (-2,  2)
    b = ( 1,  0)
    c = ( 2, -1)
    d = ( 2,  1)
    e = ( 0,  1)
    
    pts = [a,b,c,d,e]
    cy_hull = [d,a,c]
    pre(cy_hull==get_convex_hull(pts), 'convex hull error: degenerate')
    print(CC + '@G all convex hull tests passed! @D ')

