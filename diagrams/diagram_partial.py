''' author: samuel tenka
    change: 2019-05-29 
    create: 2019-03-25 
    descrp: render SGD diagrams   
'''

import matplotlib.pyplot as plt
import numpy as np
from skimage.draw import circle, circle_perimeter_aa, line_aa

def draw_disk_aa(img, row, col, rad, color=(0.0, 0.0, 0.0)):
    expanded_color = np.expand_dims(np.expand_dims(np.array(color), 0), 0)
    rr, cc, val = circle_perimeter_aa(row, col, rad)
    img[rr, cc, :] = np.expand_dims(val, 2)
    img[rr, cc, :] = img[rr, cc, :] * (1.0 - expanded_color)
    img[rr, cc, :] = 0.95 - img[rr, cc, :]
    rr, cc = circle(row, col, rad)
    img[rr, cc, :] = expanded_color

def draw_line_aa(img, row_s, col_s, row_e, col_e, color=(0.0, 0.0, 0.0)):
    expanded_color = np.expand_dims(np.expand_dims(np.array(color), 0), 0)
    rr, cc, val = line_aa(row_s, col_s, row_e, col_e)
    img[rr, cc, :] = np.expand_dims(val, 2)
    img[rr, cc, :] = img[rr, cc, :] * expanded_color

def draw_arc_aa(img, row_a, row_b, col_a, col_b, curve):
    cent_c = (col_a+col_b)/2.0
    radius = curve * ((row_b-row_a)**2 + (col_b-col_a)**2)**0.5

    old_row, old_col = row_a, col_a
    for col in list(np.arange(col_a, col_b, 2.5)) + [col_b]:
        t = float(col - col_a) / (col_b - col_a)
        dr = radius * (0.25 - (t-0.5)**2)**0.5 
        row = row_a + float(row_b-row_a)*t - dr 

        rr, cc, val = line_aa(old_row, old_col, int(row), int(col))
        img[rr, cc, :] = np.minimum(img[rr, cc, :], 1.0 - np.expand_dims(val, 2))
        old_row, old_col = int(row), int(col)

def draw_blob_aa(img, row, col_a, col_b, curve, thick = 16, outline=False):
    cent_r = int(row - abs(col_a - col_b)/curve)
    cent_c = int((col_a+col_b)/2.0)
    radius = int(((cent_r-row)**2 + (cent_c-col_a)**2)**0.5)

    for c in list(np.arange(col_a, col_b, 0.5)) + [col_b]:
        r = cent_r + (radius**2 - (c-cent_c)**2)**0.5 
        rr, cc = circle(r, c, thick)
        img[rr, cc, :] = np.maximum(0.1, 0.995 * img[rr, cc, :])

    if not outline: return
    for c in list(np.arange(col_a, col_b, 0.5)) + [col_b]:
        r = cent_r + (radius**2 - (c-cent_c)**2)**0.5 
        rr, cc = circle(r, c, thick/2)
        img[rr, cc, :] = np.minimum(1.0, 1.01 * img[rr, cc, :]) 

black = (0.0, 0.0, 0.0)
red   = (0.8, 0.2, 0.2)
green = (0.2, 0.8, 0.2)
blue  = (0.2, 0.2, 0.8)
gold  = (0.9, 0.7, 0.0)
colors = [red, green, blue, gold]

RADIN = 8 
RADOUT = 12

def draw_partial(parts, partial_arcs, filename, outline, partial_scale=0.5): 
    assert len(parts) <= len(colors)

    nb_nodes = sum(len(p) for p in parts)
    height = 80 + (nb_nodes-1)*40
    width  = 80 + (nb_nodes-1)*80
    baseline = height//2

    img = np.ones((height, width, 3), dtype=np.float32)
    for p,color in zip(parts, colors):
        for s,e in zip(p, p[1:]):
            draw_blob_aa(img, baseline, 40+80*s, 40+80*e, abs(s-e), outline=outline)
        #for i in p:
        #    R, C = baseline, 40+80*i
        #    draw_disk_aa(img, R, C, RADIN, color)

    js = []

    for aa in partial_arcs:
        if len(aa)==1:
            i, = aa
            while True:
                j = i + partial_scale * (2.0 * np.random.random() - 1.0)
                if int(40+80*i) == int(40+80*j):
                    continue
                for jj in js: 
                    if abs(jj-j) < 2.0 * partial_scale / (len(partial_arcs)+1.0):
                        break
                else:
                    break
            js.append(j)

            vshift = 40.0 * (2.0 * partial_scale**2 - (j-i)**2)**0.5

            for dr in [-1,0,1]: 
                for dc in [-1,0,1]: 
                    draw_arc_aa(img, curve = 0.1,
                        row_a = baseline          + dr,
                        row_b = baseline - vshift + dr, 
                        col_a = int(40+80*i) + dc,
                        col_b = int(40+80*j) + dc,
                    )
        elif len(aa)==2:
            i,j = aa
            for dr in [-1,0,1]: 
                for dc in [-1,0,1]: 
                    draw_arc_aa(img, curve = 0.1 * 2.0**abs(i-j),
                        row_a = baseline + dr,
                        row_b = baseline + dr, 
                        col_a = int(40+80*i) + dc,
                        col_b = int(40+80*j) + dc,
                    )

    for p,color in zip(parts, colors):
        #for s,e in zip(p, p[1:]):
        #    draw_blob_aa(img, baseline, 40+80*s, 40+80*e, abs(s-e), outline=outline)
        for i in p:
            R, C = baseline, 40+80*i
            draw_disk_aa(img, R, C, RADIN, color)

    plt.imsave(filename, img)

for outline in [False, True]:
    for pp, gg in (
        #([[0]], [[0]]),
        #([[0]], [[0],[0]]),
        #([[0]], [[0],[0],[0]]),
        #([[0,1]],   [[0],[1]]),
        #([[0,1,2]], [[0],[1],[2]]),
        #
        #([[0,1,2,3]], [[0],[1],[2],[3]]),
        #([[0,1,2,3,4]], [[0],[1],[2],[3],[4]]),
        #([[0,1]], [[0],[1],[1]]),
        #([[0,1]], [[0],[0],[1],[1]]),
        #([[0,1]], [[0],[1],[1],[1]]),
        #
        #([[0,1,2],[3],[4]], [[0,1],[2,3],[3,4]]),
        #([[0,1,2],[3],[4]], [[0,3],[1,3],[2,4]]),
        #
        ([[0,1]], [[0],[1]]),
        ([[0,1],[2,3]], [[0,2],[1,3]]),
        ([[0,1]], [[0,1]]),
        ([[0],[1,2,3]], [[0,2],[1,2],[2,3]]),
        ([[0]], [[0]]),
        ([[0,1]], [[0,1],[1]]),
    ):
        c_nm = ('c' if outline else '')
        p_nm = '({})'.format('-'.join(''.join(str(s) for s in p) for p in pp))
        t_nm = '({})'.format('-'.join(''.join(str(s) for s in g) for g in gg)) 
        nm = 'MOO{}{}{}.png'.format(c_nm, p_nm, t_nm)
        print(nm)
        draw_partial(parts=pp, partial_arcs=gg, filename=nm, outline=outline)

'''
For example, the generalized diagrams $\sdia{MOOc(01)(0-1)}$ or
$\sdia{MOOc(01-23)(02-13)}$ may appear in this computation.

For example, the generalized diagrams $\sdia{MOOc(01)(01)}$ or
$\sdia{MOOc(0-123)(02-12-23)}$ may appear in this computation.

For example, the generalized diagrams $\sdia{MOOc(0)(0)}$ or
$\sdia{MOOc(01)(01-1)}$ may appear in this computation.
'''
