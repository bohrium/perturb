''' author: samuel tenka
    change: 2019-03-25 
    create: 2019-03-25 
    descrp: render SGD diagrams   
'''

import matplotlib.pyplot as plt
import numpy as np
from skimage.draw import circle, circle_perimeter_aa, line_aa
from PIL import Image, ImageDraw, ImageFont

fnt = ImageFont.truetype('Pillow/Tests/fonts/DejaVuSans.ttf', 24)

black = (0.0, 0.0, 0.0)
red   = (0.8, 0.2, 0.2)
green = (0.2, 0.8, 0.2)
blue  = (0.2, 0.2, 0.8)
gold  = (0.9, 0.7, 0.0)
colors = [red, green, blue, gold, black]

def draw_disk_aa(img, row, col, rad=7, color=black):
    expanded_color = np.expand_dims(np.expand_dims(np.array(color), 0), 0)
    rr, cc, val = circle_perimeter_aa(row, col, rad)
    img[rr, cc, :] = np.expand_dims(val, 2)
    img[rr, cc, :] = img[rr, cc, :] * (1.0 - expanded_color)
    img[rr, cc, :] = 0.95 - img[rr, cc, :]
    rr, cc = circle(row, col, rad)
    img[rr, cc, :] = expanded_color

def draw_line_aa(img, row_s, col_s, row_e, col_e, color=black):
    expanded_color = np.expand_dims(np.expand_dims(np.array(color), 0), 0)
    rr, cc, val = line_aa(row_s, col_s, row_e, col_e)
    #img[rr, cc, :] = np.expand_dims(val, 2)
    #img[rr, cc, :] = img[rr, cc, :] * expanded_color
    img[rr, cc, :] = np.minimum(img[rr, cc, :], np.expand_dims(val, 2) * expanded_color)

def draw_arc_aa(img, row_s, col_s, row_e, col_e, curve=1.0, color=black):
    length = ((row_e-row_s)**2 + (col_e-col_s)**2)**0.5  
    row_diff = (row_e-row_s)
    col_diff = (col_e-col_s)

    get_rr = lambda t: int(row_s + (row_e-row_s)*t - curve * (1.0-(2*t-1)**2)**0.5 * col_diff) 
    get_cc = lambda t: int(col_s + (col_e-col_s)*t + curve * (1.0-(2*t-1)**2)**0.5 * row_diff)  
    times = np.arange(0.0, 1.0+1e-5, 1.0/int(length/3.0))
    times = ((np.abs(2*times-1) * np.sign(2*times-1))+1)/2
    points = [(get_rr(t), get_cc(t)) for t in times] 
    for (r, c), (rr, cc) in zip(points, points[1:]):
        draw_line_aa(img, r, c, rr, cc, color)

N = 8
T = 8
MARG = 30
SPACE = 60
height = 3*MARG + N*SPACE
width  = 4*MARG + T*SPACE
get_r = lambda n: int(2*MARG + (N-n)*SPACE )
get_c = lambda t: int(2*MARG + t*SPACE)

def draw_grid(img): 
    for n in range(N+1):
        draw_line_aa(img, get_r(n), get_c(0), get_r(n), get_c(T))
    for t in range(T+1):
        draw_line_aa(img, get_r(0), get_c(t), get_r(N), get_c(t))

def draw_edges(img, nts):
    for (n, t), (nn, tt) in zip(nts, nts[1:]):
        draw_arc_aa(img, get_r(n+0.5), get_c(t+0.5), get_r(nn+0.5), get_c(tt+0.5), curve=0.2)

def draw_nodes(img, nts):
    ns = sorted(list(set(int(n) for n,t in nts if t < T-0.5)))
    for n, t in nts:
        if not (t < T-0.5): continue
        draw_disk_aa(img, get_r(n+0.5), get_c(t+0.5), color=colors[ns.index(int(n))])

def draw_diagram(img, ntss):
    for nts in ntss:
        draw_edges(img, nts)
    draw_nodes(img, [nt for nts in ntss for nt in nts])

def draw(filename): 
    img = np.ones((height, width, 3), dtype=np.float32)
    draw_grid(img)

    # an order 5 diagram:
    draw_diagram(img, [[(5, 0+0.2), (4, 2), (5, 4), (5.5    , 8    )],
                       [(5, 0-0.2),                 (5.5    , 8    )],
                       [            (6, 3), (5, 4),                 ]])

    # an order 3 diagram (high degree):
    draw_diagram(img, [[(7, 7+0.3),                 (7.5-0.5, 8    )],
                       [(7, 7+0.0),                 (7.5-0.5, 8    )],
                       [(7, 7-0.3),                 (7.5-0.5, 8    )]])

    # an order 1 diagram:
    draw_diagram(img, [[(4, 4    ),                 (4.5    , 8    )]])

    # translates of an order 2 diagram:
    draw_diagram(img, [[(1, 1    ), (0, 3),        (-0.5    , 8    )]])
    draw_diagram(img, [[(1, 4    ), (1, 7),         (0.5    , 8    )]])
    draw_diagram(img, [[(1, 0    ), (2, 4),         (1.5    , 8    )]])
    draw_diagram(img, [[(3, 5    ), (3, 6),         (2.5    , 8    )]])

    plt.imsave(filename, img)


draw('spacetime.png')

#=============================================================================

im = Image.open("spacetime.png")

draw = ImageDraw.Draw(im)
draw.text((get_c(1.5), get_r(8+0.7)), "Space Time for Pure GD", font=fnt, fill=(0,0,0,255))
for t in range(T):
    draw.text((get_c(t+0.1), get_r(0.0)), "T=%d"%t, font=fnt, fill=(0,0,0,255))
for n in range(N):
    draw.text((get_c(-0.9), get_r(n+0.7)), "N=%d"%n, font=fnt, fill=(0,0,0,255))

im.save('spacetime.png')
