''' author: samtenka
    change: 2020-01-19
    create: 2020-01-19
    descrp: compute convex hull in two dimensions (for nice plotting of
            two-dimensional uncertainties).
            CAUTION: we do not handle degenerate edges at all
'''

import numpy as np
from utils import pre, CC

def cross_product(v, w):
    # positive when w sits to the left of v   
    return v[0]*w[1] - v[1]*w[0]  
def dot(v, w):
    return v[0]*w[0] + v[1]*w[1]  
def mag(v):
    return np.sqrt(dot(v, v))
def alignment(v, w):
    return dot(v, w) / (mag(v) * mag(w)) 

def angle(v, w):
    cp = cross_product(v, w)
    pre(cp!=0, 'unable to compute angle in degenerate cases')

    al = alignment(v, w)

    angle = np.arcsin(cp / (mag(v) * mag(w)))
    if al < 0:   
        angle = np.sign(angle)*np.pi - angle   

    return angle

def winding_number(cycle, pt):
    x, y = pt
    return int(round(sum(
        angle((ax-x, ay-y), (bx-x, by-y))
        for i in range(len(cycle))
        for (ax, ay), (bx, by) in [(cycle[i], cycle[(i+1)%len(cycle)])]
    )/(2 * np.pi)))

def cycle_contains(cycle, pt):
    return 0 != winding_number(cycle, pt)

def edge_classifier(edge, pt):
    x, y = pt
    (ax, ay), (bx, by) = edge 
    cp = cross_product((bx-ax, by-ay), (x-ax, y-ay)) 
    return ('left' if 0 < cp else 'right' if cp < 0 else 'edge') 

def get_edge(cycle, idx):
    return (
        cycle[(idx  )%len(cycle)],
        cycle[(idx+1)%len(cycle)]
    )

def right_turn_possible(ab, cd):
    ''' check if segments intersect and oriented correctly '''
    c, d = cd
    return (
        edge_classifier(ab, c)=='left' and
        edge_classifier(ab, d)=='right'
    )


def normalize_cycle(cycle):
    ''' ensure maximum (x,y) tup is initial entry '''
    i = cycle.index(max(cycle))
    return cycle[i:] + cycle[:i]


def join_cycles(lhs, rhs):
    '''
        boundary of simply-connected-closure of union of interiors of lhs and rhs
        assumes that curves are fine-grained and smooth enough so that they do 
        not self-intersect and so that if each edge of one intersects at most
        one edge of the other.  as always, also assumes that both are oriented
        counterclockwise and non-degenerate
    '''
    for ri, rhs_pt in enumerate(rhs): 
        if cycle_contains(lhs, rhs_pt): continue
        break
    else: # lhs contains all of rhs' points
        return lhs 
    # now, rhs_pt is not in lhs

    new_cycle = [rhs_pt]
    cur_idx = ri  
    cur_cycle, oth_cycle = rhs, lhs
    while True:
        cur_edge = get_edge(cur_cycle, cur_idx)  
        right_turns = [
            (oth_idx, oth_edge)
            for oth_idx in range(len(oth_cycle))
            for oth_edge in [get_edge(oth_cycle, oth_idx)]
            if right_turn_possible(cur_edge, oth_edge) 
        ] 
        if len(right_turns)==0:
            new_pt = cur_edge[1]
        elif len(right_turns)==1:
            oth_idx, oth_edge = right_turns[0]
            new_pt = oth_edge[1]
            cur_idx = oth_idx
            cur_cycle, oth_cycle = oth_cycle, cur_cycle 
        else:
            pre(len(right_turns)<=1,
                'assumption of at most one interesection violated!'
            )
        if new_pt == rhs_pt: break
        new_cycle.append(new_pt)
        cur_idx += 1

    return normalize_cycle(new_cycle)



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

def normalize_cycle(cy_hull):
    ''' ensure maximum (x,y) tup is initial entry '''
    i = cy_hull.index(max(cy_hull))
    return cy_hull[i:] + cy_hull[:i]
def get_convex_hull(pts):
    pts = list(set(pts))
    pre(2<= len(pts), 'want at least 2 points')
    cy_hull = [pts[0], pts[1]]
    while True:
        for pt in pts: 
            inner_edge_idxs = get_inner_edge_idxs(pt, cy_hull)
            if not inner_edge_idxs: continue
            cy_hull = update_hull(pt, cy_hull, inner_edge_idxs)
            break
        else:
            break
    return normalize_cycle(cy_hull)

if __name__=='__main__':
    #-------------------------------------------------------------------------#
    #               1.0 test trigonometry                                     #
    #-------------------------------------------------------------------------#

    pre(0.0 <  cross_product((1, 0), (0, 1)), 'cross product error: pos')
    pre(0.0 == cross_product((1, 0), (2, 0)), 'cross product error:zero')
    pre(0.0 >  cross_product((0, 1), (1, 0)), 'cross product error: neg')

    pre(1e-3>abs(+0.50 - angle(( 1,  0), ( 0,  1))/np.pi), 'angle error: +2 pi/4')
    pre(1e-3>abs(-0.50 - angle(( 1,  0), ( 0, -1))/np.pi), 'angle error: -2 pi/4')
    pre(1e-3>abs(+0.25 - angle(( 1,  0), ( 3,  3))/np.pi), 'angle error: +1 pi/4')
    pre(1e-3>abs(-0.25 - angle(( 1,  0), ( 3, -3))/np.pi), 'angle error: -1 pi/4')
    pre(1e-3>abs(+0.75 - angle(( 1,  0), (-5,  5))/np.pi), 'angle error: +3 pi/4')
    pre(1e-3>abs(-0.75 - angle(( 1,  0), (-5, -5))/np.pi), 'angle error: -3 pi/4')

    print(CC + '@G all trigonometry tests passed! @D ')

    #-------------------------------------------------------------------------#
    #               1.1 test convex hull                                      #
    #-------------------------------------------------------------------------#

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

    #-------------------------------------------------------------------------#
    #               1.2 test cycle joining                                    #
    #-------------------------------------------------------------------------#

    diamond = [(+1,  0), ( 0, +1), (-1,  0), ( 0, -1)]
    pre(1.0==winding_number(diamond, (0, 0)), 'winding number error: inside')
    pre(0.0==winding_number(diamond, (0, 5)), 'winding number error: outside')
    pre(2.0==winding_number(diamond+diamond, (0, 0)), 'winding number error: double')

    '''
        . . . . . . . . . . .
        . . . . . . . . . . .
        . . . . . d . . . . .
        . . . . . . . . . . .
        . . . . . . r . . . r
        . . d . . + . . d . .
        . . . . . . r . . . r
        . . . . . . . . . . .
        . . . . . d . . . . .
        . . . . . . . . . . .
        . . . . . . . . . . .
    '''
    diamond =   [(+3,  0), ( 0, +3), (-3,  0), ( 0, -3)]
    rectangle = [( 5, +1), ( 1, +1), ( 1, -1), ( 5, -1)]
    joined =    [( 5, +1), ( 0, +3), (-3,  0), ( 0, -3), ( 5, -1)] 
    pre(joined==join_cycles(diamond, rectangle), 'join cycles error: diamond and rectangle')
    
    '''
        . . . . . . . . . . .
        . . . . p o . . . . .
        . . o . . . . . o . .
        . . . . . . . . . . .
        . . . . . . . . . . p
        . o p . . + . . . o .
        . . . . . . . . . . p
        . . . . . . . . . . .
        . . o . . . . . o . .
        . . . . p o . . . . .
        . . . . . . . . . . .
    '''
    octagon = [(+3, +3), ( 0, +4), (-3, +3), (-4,  0), (-3, -3), ( 0, -4), (+3, -3), (+4,  0)]
    pentagon= [( 5, +1), (-1, +4), (-3,  0), (-1, -4), ( 5, -1)]
    joined =  [( 5, +1), (+3, +3), ( 0, +4), (-1, +4), (-3, +3), (-4,  0), (-3, -3), (-1, -4), ( 0, -4), (+3, -3), ( 5, -1)]
    pre(joined==join_cycles(octagon, pentagon), 'join cycles error: diamond and rectangle')

    print(CC + '@G all cycle join tests passed! @D ')
