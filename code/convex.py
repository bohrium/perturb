''' author: samtenka
    change: 2020-01-19
    create: 2020-01-19
    descrp: compute convex hull in two dimensions (for nice plotting of
            two-dimensional uncertainties).
            CAUTION: we do not handle degenerate edges at all
'''

import numpy as np
from utils import pre, CC

#=============================================================================#
#           0. TRIGONOMETRY                                                   #
#=============================================================================#

def cross_product(v, w):
    '''
        Compute the cross product of two vectors.  This quantity is positive
        when w sits to v's left (that is, when a sweep from v to w is a
        counterclockwise motion); it transforms as a pseudo-scalar.
    '''
    return v[0]*w[1] - v[1]*w[0]  

def dot(v, w):
    '''
        Compute the standard inner product on the standard plane.
    '''
    return v[0]*w[0] + v[1]*w[1]  

def mag(v):
    '''
        Compute the standard norm on the standard plane.
    '''
    return np.sqrt(dot(v, v))

def cos_angle(v, w):
    '''
        Compute cos(angle between v and w).  Note that this loses the sign
        information of that angle.
    '''
    return dot(v, w) / (mag(v) * mag(w)) 

def angle(v, w):
    '''
        Compute a signed angle of the sweep from v to w.
    '''
    cp = cross_product(v, w)
    pre(cp!=0, 'unable to compute angle in degenerate cases')
    al = cos_angle(v, w)

    angle = np.arcsin(cp / (mag(v) * mag(w)))
    if al < 0:   
        angle = np.sign(angle)*np.pi - angle   

    return angle

def edge_classifier(edge, pt):
    '''
        Which side of the given directed edge is (B <-- A) the given point?
        The possibilities are 'left', 'right', 'above', 'on', or 'below': 

            rrrrrrrrrrrrrrrrrrrrrrrrrrrrr
            rrrrrrr   rrrrrrr   rrrrrrrrr  
            aaaaaaa B ooooooo A bbbbbbbbb
            lllllll   lllllll   lllllllll
            lllllllllllllllllllllllllllll
    '''
    x, y = pt
    (ax, ay), (bx, by) = edge 
    cp = cross_product((bx-ax, by-ay), (x-ax, y-ay)) 
    return ('left' if 0 < cp else 'right' if cp < 0 else 'edge') 

def right_turn_possible(ab, cd):
    '''
        Does the segment (a --> b) not only intersect the segment (c --> d)
        but in fact permit a right turn from a to that intersection to d? 

        POSITIVE EXAMPLE:       NEGATIVE EXAMPLE:       NEGATIVE EXAMPLE:

                 d                       c                d                 
                 ^                       |                ^                 
                 |                       |                |                 
          b <----+---- a          b <----+---- a          |   b <----- a    
                 |                       |                |                 
                 |                       v                |                 
                 c                       d                c                 
    '''
    (a,b), (c,d) = ab, cd
    return (
        edge_classifier(ab, c)=='left'  and
        edge_classifier(ab, d)=='right' and
        edge_classifier(cd, a)=='right' and
        edge_classifier(cd, b)=='left'
    )

#=============================================================================#
#           1. CYCLES                                                         #
#=============================================================================#

def get_edge(cycle, idx):
    '''
        Get the `idx`th edge of a cycle, wrapping around as necessary. 
    '''
    return (
        cycle[(idx  )%len(cycle)],
        cycle[(idx+1)%len(cycle)]
    )

def normal_form_of_cycle(cycle):
    '''
        Return a canonical version of the cycle: do this by rotating the
        cycle's representation so that its initial entry is the point that is
        rightmost (and to break ties, uppermost) among all points in the cycle.
    '''
    i = cycle.index(max(cycle))
    return cycle[i:] + cycle[:i]

def winding_number(cycle, pt):
    '''
        Compute the number of times `cycle` winds counterclockwise around `pt`
    '''
    x, y = pt
    return int(round(sum(
        angle((ax-x, ay-y), (bx-x, by-y))
        for i in range(len(cycle))
        for (ax, ay), (bx, by) in [(cycle[i], cycle[(i+1)%len(cycle)])]
    )/(2 * np.pi)))

def cycle_encloses(cycle, pt):
    '''
        Is `pt` strictly inside the region delimited by `cycle`? 
    '''
    return 0 != winding_number(cycle, pt)

def join_cycles(lhs, rhs):
    '''
        Approximate the boundary of the union of the interiors of the two given
        cycles `lhs` and `rhs` We assume that this union is (connected and)
        simply connected; that the cycles are sufficiently fine-grained and
        smooth so that each edge of one intersects at most one edge of the
        other; and, as always, that the cycles avoid self-intersection, are
        oriented counterclockwise, and minimally represent their image.
    '''
    for ri, rhs_pt in enumerate(rhs): 
        if cycle_encloses(lhs, rhs_pt): continue
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
            if new_pt == rhs_pt: break
            new_cycle.append(new_pt)
        elif len(right_turns)==1:
            oth_idx, oth_edge = right_turns[0]
            new_pt = oth_edge[1]
            cur_idx = oth_idx
            cur_cycle, oth_cycle = oth_cycle, cur_cycle 
            new_cycle.append(cur_edge[1])
            if new_pt == rhs_pt: break
            new_cycle.append(new_pt)
        else:
            pre(len(right_turns)<=1,
                'assumption of at most one interesection violated!'
            )
        cur_idx += 1

    return normal_form_of_cycle(new_cycle)

#=============================================================================#
#           2. CONVEX HULLS                                                   #
#=============================================================================#

def get_shadow_idxs(pt, hull):
    '''
    '''
    shadow_idxs = []
    for i in range(len(hull)):
        edge = get_edge(hull, i)
        if edge_classifier(edge, pt)=='right':
            shadow_idxs.append(i)
    return shadow_idxs

def update_hull(pt, hull, shadow_idxs):
    '''
    '''
    pre(shadow_idxs, 'vacuous update detected!')
    new_hull=[]
    for i in range(len(hull)): 
        pre_inner = ((i-1)%len(hull) in shadow_idxs)
        post_inner= (i in shadow_idxs)
        if (not post_inner) and pre_inner:
            new_hull.append(pt)
        if not (post_inner and pre_inner):
            new_hull.append(hull[i])
    return new_hull

def get_convex_hull(pts):
    '''
    '''
    pts = list(set(pts))
    pre(2<= len(pts), 'want at least 2 points')
    hull = [pts[0], pts[1]]
    for pt in pts: 
        shadow_idxs = get_shadow_idxs(pt, hull)
        if not shadow_idxs: continue
        hull = update_hull(pt, hull, shadow_idxs)
    return normal_form_of_cycle(hull)

#=============================================================================#
#           3. DEMONSTRATION AND TESTING                                      #
#=============================================================================#

if __name__=='__main__':
    #-------------------------------------------------------------------------#
    #               3.0 test trigonometry                                     #
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
    #               3.1 test convex hull                                      #
    #-------------------------------------------------------------------------#

    diamond = [(+1,  0), ( 0, +1), (-1,  0), ( 0, -1)]
    pre([] == get_shadow_idxs((0.0, 0.0), diamond), 'inner edge error: empty')
    pre([0] == get_shadow_idxs((0.8, 0.8), diamond), 'inner edge error: single')
    pre([0, 3] == get_shadow_idxs((2.0, 0.0), diamond), 'inner edge error: range')
    
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
    hull = [f,a,b,c,e]
    pre(hull==get_convex_hull(pts), 'convex hull error: pentagon')


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
    hull = [d,a,c]
    pre(hull==get_convex_hull(pts), 'convex hull error: triangle')

    #'''
    #    b . . . d
    #    . a . . .
    #    . . . . .
    #    . . . . .
    #    . . . . c
    #'''
    #a = (-1,  1)
    #b = (-2,  2)
    #c = ( 2, -2)
    #d = ( 2,  2)
    #
    #pts = [a,b,c,d]
    #hull = [d, b, c]
    #pre(hull==get_convex_hull(pts), 'convex hull error: degenerate')
    print(CC + '@G all convex hull tests passed! @D ')

    #-------------------------------------------------------------------------#
    #               3.2 test cycle joining                                    #
    #-------------------------------------------------------------------------#

    diamond = [(+1,  0), ( 0, +1), (-1,  0), ( 0, -1)]
    pre(1.0==winding_number(diamond, (0, 0)), 'winding number error: inside')
    pre(0.0==winding_number(diamond, (0, 5)), 'winding number error: outside')
    pre(2.0==winding_number(diamond+diamond, (0, 0)), 'winding number error: double')

    '''
        . . . . . . . . . . .
        . . . . . . . . . . .
        . . . . . d . . . . .
        . . . . . h . . . . .
        . . . . h . . . . . h
        . . d . . + . . d . .
        . . . . h . . . . . h
        . . . . . h . . . . .
        . . . . . d . . . . .
        . . . . . . . . . . .
        . . . . . . . . . . .
    '''
    diamond = [(+3,  0), ( 0, +3), (-3,  0), ( 0, -3)]
    hexagon = [( 5, +1), ( 0, +1), (-1, +1), (-1, -1), ( 0, -1), ( 5, -1)]
    joined  = [( 5, +1), ( 0, +1), ( 0, +3), (-3,  0), ( 0, -3), (+3,  0),  ( 5, -1)]
    pre(joined==join_cycles(diamond, hexagon), 'join cycles error: diamond and hexagon')
    
    '''
        . . . . . . . . . . .
        . . . . . . . . . . .
        . . . . . o . . . . .
        . . . o . d . o . . .
        . . . . . . . . . . .
        d . o . . + . . o . d
        . . . . . . . . . . .
        . . . o . d . o . . .
        . . . . . o . . . . .
        . . . . . . . . . . .
        . . . . . . . . . . .
    '''
    octagon = [(+2, +2), ( 0, +3), (-2, +2), (-3,  0), (-2, -2), ( 0, -3), (+2, -2), (+3,  0)]
    diamond = [(+5,  0), ( 0, +2), (-5,  0), ( 0, -2)]
    joined  = [(+5,  0), ( 0, +2), (+2, +2), ( 0, +3), (-2, +2), (-3,  0), (-5,  0), ( 0, -2), (-2, -2), ( 0, -3), (+2, -2), (+3,  0)] 
    pre(joined==join_cycles(octagon, diamond), 'join cycles error: octagon and diamond')

    #'''
    #    . . . . . . . . . . .
    #    . . . . . . . . . . .
    #    . . . . a @ . . . . .
    #    . . . o . . . @ . . .
    #    . . . . . . . . . . .
    #    . . o . a + . . @ . .
    #    . . . . . . . . . . .
    #    . . . o . . . @ . . .
    #    . . . . . o a . . . .
    #    . . . . . . . . . . .
    #    . . . . . . . . . . .
    #'''
    #oo      = [(+2, +2), ( 0, +3), (-2, +2), (-3,  0), (-2, -2), ( 0, -3), (+2, -2), (+3,  0)]
    #aa      = [( 3,  0), ( 2,  2), ( 0,  3), (-1,  3), (-1,  0), ( 1, -3), ( 2, -2)]
    #pre(joined==join_cycles(oo, aa), 'join cycles error: degenerate')

    print(CC + '@G all cycle join tests passed! @D ')
