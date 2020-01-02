''' author: samuel tenka
    change: 2019-12-31 
    create: 2019-03-25 
    descrp: render SGD diagrams   
'''

import matplotlib.pyplot as plt
import numpy as np
from skimage.draw import circle, circle_perimeter_aa, line_aa, rectangle
from PIL import Image, ImageDraw, ImageFont

fnt = ImageFont.truetype('Pillow/Tests/fonts/DejaVuSans.ttf', 24)

black = (0.0, 0.0, 0.0)

red   = (0.8, 0.2, 0.2)
gold  = (0.9, 0.7, 0.0)
green = (0.2, 0.8, 0.2)
cyan  = (0.0, 0.9, 0.7)
blue  = (0.2, 0.2, 0.8)
royal = (0.7, 0.0, 0.9)

colors = [red, green, blue, gold, cyan, royal, black]

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
    img[rr, cc, :] = np.minimum(img[rr, cc, :], np.expand_dims(val, 2) * expanded_color)

def draw_arrow_aa(img, row_s, col_s, row_e, col_e, color=black):
    length = ((row_e-row_s)**2 + (col_e-col_s)**2)**0.5  
    row_diff = int(2.0 * (row_e-row_s)/length)
    col_diff = int(2.0 * (col_e-col_s)/length)

    draw_line_aa(img, row_s+col_diff, col_s-row_diff, row_e+col_diff, col_e-row_diff, color)
    draw_line_aa(img, row_s-col_diff, col_s+row_diff, row_e-col_diff, col_e+row_diff, color)
    draw_line_aa(img, row_e+3*col_diff-3*row_diff, col_e-3*row_diff-3*col_diff,
                      row_e             +row_diff, col_e             +col_diff, color)
    draw_line_aa(img, row_e-3*col_diff-3*row_diff, col_e+3*row_diff-3*col_diff,
                      row_e             +row_diff, col_e             +col_diff, color)

def draw_arc_aa(img, row_s, col_s, row_e, col_e, curve=1.0, color=black):
    length = ((row_e-row_s)**2 + (col_e-col_s)**2)**0.5  
    row_diff = (row_e-row_s)
    col_diff = (col_e-col_s)

    get_rr = lambda t: int(row_s + (row_e-row_s)*t - curve * (1.0-(2*t-1)**2)**0.5 * col_diff) 
    get_cc = lambda t: int(col_s + (col_e-col_s)*t)#+ curve * (1.0-(2*t-1)**2)**0.5 * row_diff)  
    times = np.arange(0.0, 1.0+1e-5, 1.0/int(length/3.0))
    times = ((np.abs(2*times-1) * np.sign(2*times-1))+1)/2
    points = [(get_rr(t), get_cc(t)) for t in times] 
    for (r, c), (rr, cc) in zip(points, points[1:]):
        draw_line_aa(img, r, c, rr, cc, color)

N = 5
T = 15
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
        offsets = [0.0] if t%N else [-0.02, -0.01, 0.00, 0.01, 0.02]
        for o in offsets:
            draw_line_aa(img, get_r(0), get_c(t+o), get_r(N), get_c(t+o))

def draw_edges(img, nts):
    for (n, t), (nn, tt) in zip(nts, nts[1:]):
        draw_arc_aa(img, get_r(n+0.5), get_c(t+0.5), get_r(nn+0.5), get_c(tt+0.5), curve=0.2)

def fill_boxes(img, nts):
    for (n, t) in nts:
        rr, cc = rectangle((get_r(n), get_c(t)), (get_r(n+1), get_c(t+1)))
        img[rr, cc, :] = 0.9

def draw_nodes(img, nts):
    #ns = sorted(list(set(int(n) for n,t in nts)))
    for n, t in nts:
        #draw_disk_aa(img, get_r(n+0.5), get_c(t+0.5), color=colors[ns.index(int(n))])
        draw_disk_aa(img, get_r(n+0.5), get_c(t+0.5), color=colors[int(n+0.5)])

def draw_push(img, n, t, nn, tt):
    diff_n = 0.15*(nn-n)  
    diff_t = 0.15*(tt-t)
    draw_arrow_aa(img, get_r(n+0.5+diff_n), get_c(t+0.5+diff_t),
                       get_r(nn+0.5-diff_n), get_c(tt+0.5-diff_t))

def draw_diagram(img, ntss):
    for nts in ntss:
        draw_edges(img, nts)
    draw_nodes(img, [nt for nts in ntss for nt in nts])

def draw(filename): 
    img = np.ones((height, width, 3), dtype=np.float32)
    fill_boxes(img, [(
        (N-i-1)                  % N,
        i%T
    ) for i in range(max(N, T))])
    draw_grid(img)

    ## order 1 diagram push:
    #draw_diagram(img, [[(3, 4    ),                 (5.0-0.2, 4.5)]])
    #draw_diagram(img, [[(0, 4    ),                 (5.0-0.2, 5.5)]])
    #draw_push(img, 3     , 4, 0     , 4)    

    # order 2 diagram push:
    draw_diagram(img, [[(3, 0    ),                 (5.0-0.1, 3.0 )], 
                       [            (3, 2    ),     (5.0-0.1, 3.0 )]])
    draw_diagram(img, [[(3, 1-0.2),                 (5.0-0.3, 2.5 )], 
                       [            (3, 1+0.2),     (5.0-0.3, 2.5 )]])
    draw_push(img, 3     , 0, 3     , 1-0.2)    
    draw_push(img, 3     , 2, 3     , 1+0.2) 

    # order 2 diagram push:
    draw_diagram(img, [[(2, 4    ),                 (5.0-0.3, 6.0 )], 
                       [            (3, 5    ),     (5.0-0.3, 6.0 )]])
    draw_diagram(img, [[(2, 2    ),                 (5.0-0.1, 6.5 )], 
                       [            (3, 6    ),     (5.0-0.1, 6.5 )]])
    draw_push(img, 2     , 4, 2     , 2)    
    draw_push(img, 3     , 5, 3     , 6) 

    # order 2 diagram push:
    draw_diagram(img, [[(0,  1   ), (1,  7   ),     (5.0-0.2, 8.0)]])
    draw_diagram(img, [[(0,  4   ), (1,  8   ),     (5.0-0.2, 9.0)]])
    draw_push(img, 0     , 1, 0     , 4)    
    draw_push(img, 1     , 7, 1     , 8) 

    # order 2 diagram push:
    draw_diagram(img, [[(1, 10   ), (2, 11   ),     (5.0-0.2,12.0)]])
    draw_diagram(img, [[(4, 10   ), (3, 11   ),     (5.0-0.2,11.5)]])
    draw_push(img, 1     ,10, 4     ,10)    
    draw_push(img, 2     ,11, 3     ,11) 

    # order 2 diagram push:
    draw_diagram(img, [[(3, 13   ),  (3, 14),        (5.0-0.2,14.5 )]])
    draw_diagram(img, [[(1, 13   ),  (0, 14),        (5.0-0.2,15.0 )]])
    draw_push(img, 3     ,13, 1     ,13)    
    draw_push(img, 3     ,14, 0     ,14)    


    plt.imsave(filename, img)


draw('spacetime-b.png')

#=============================================================================

im = Image.open("spacetime-b.png")

draw = ImageDraw.Draw(im)
for t in range(0, T, 7):
    draw.text((get_c(t+0.1), get_r(0.0)), "t=%d"%t, font=fnt, fill=(0,0,0,255))
for n in range(0, N, 2):
    draw.text((get_c(-0.9), get_r(n+0.7)), "n=%d"%n, font=fnt, fill=(0,0,0,255))


im.save('spacetime-b.png')
