''' author: samuel tenka
    change: 2019-12-31 
    create: 2019-03-25 
    descrp: render SGD diagrams   
'''

import matplotlib.pyplot as plt
import numpy as np
from skimage.draw import circle, circle_perimeter_aa, line_aa, rectangle
from PIL import Image, ImageDraw, ImageFont

fnt = ImageFont.truetype('Pillow/Tests/fonts/DejaVuSans.ttf', 21)
fntbig = ImageFont.truetype('Pillow/Tests/fonts/DejaVuSans.ttf', 42)
fnt = fntbig

black = (0.0, 0.0, 0.0)
gray  = (0.8, 0.8, 0.8)

red   = (0.8, 0.2, 0.0)
orange= (0.6, 0.4, 0.1)
lemon = (0.4, 0.6, 0.1)
green = (0.0, 0.8, 0.2)
teal  = (0.1, 0.6, 0.4)
sky   = (0.1, 0.4, 0.6)
blue  = (0.2, 0.0, 0.8)
indigo= (0.4, 0.1, 0.6)
purple= (0.6, 0.1, 0.4)

colors = [
    orange, teal, indigo,
    lemon, sky, purple,
    red, green, blue,
    black
]

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

N = 8
dN = 7
T = 16
dT = 5 

#MARG = 30
MARG = 45
SPACE = 60
height = 3*MARG + N*SPACE
#width  = 4*MARG + T*SPACE
width  = 3*MARG + T*SPACE
get_r = lambda n: int(2*MARG + (N-n)*SPACE)
get_c = lambda t: int(2*MARG + t*SPACE)

texts = [] 

def draw_grid(img): 
    for n in range(N):  
        # horizontals:
        draw_line_aa(img, get_r(n+0.05), get_c(0), get_r(n+0.05), get_c(T), color=gray)
        draw_line_aa(img, get_r(n+0.95), get_c(0), get_r(n+0.95), get_c(T), color=gray)

        # verticals:
        for t in range(T+1):
            draw_line_aa(img, get_r(n+0.05), get_c(t), get_r(n+0.95), get_c(t), color=gray)

def queue_text(n, t, text, color=black, rot=0):
    texts.append((n, t, text, color, rot))

def draw_x(img, n, t, nn, tt, text='', color=black):
    draw_line_aa(img, get_r(n    ), get_c(t     ), get_r(nn+1.0), get_c(tt+1.0), color=color)
    draw_line_aa(img, get_r(n    ), get_c(tt+1.0), get_r(nn+1.0), get_c(t     ), color=color)
    queue_text(n, (t+tt)/2.0, text)

def fill_boxes(img, nts):
    for (n, t) in nts:
        rr, cc = rectangle((get_r(n+0.05), get_c(t)), (get_r(n+0.95), get_c(t+1)))
        img[rr, cc, :] = 0.9 

def draw_edges(img, nts):
    for (n, t), (nn, tt) in zip(nts, nts[1:]):
        draw_arc_aa(img, get_r(n+0.5), get_c(t+0.5), get_r(nn+0.5), get_c(tt+0.5), curve=0.2)

def draw_nodes(img, nts):
    #ns = sorted(list(set(int(n) for n,t in nts)))
    for n, t in nts:
        #draw_disk_aa(img, get_r(n+0.5), get_c(t+0.5), color=colors[ns.index(int(n))])
        color = colors[int(n+0.5)] if (0<=n+0.5<=N and 0<=t+0.5<=T) else black 
        draw_disk_aa(img, get_r(n+0.5), get_c(t+0.5), color=color)

def draw_diagram(img, ntss):
    for nts in ntss:
        draw_edges(img, nts)
    draw_nodes(img, [nt for nts in ntss for nt in nts])

#def draw(filename): 
#    img = np.ones((height, width, 3), dtype=np.float32)
#    fill_boxes(img, [(
#        #(i*(int(i//N) + 1)) % N,
#        i                  % N,
#        i%T
#    ) for i in range(max(N, T))])
#    draw_grid(img)
#
#    # an order 1 diagram:
#    draw_diagram(img, [[(3, 0    ),                 (5.0-0.2, 1.5  )]])
#    draw_diagram(img, [[(0, 1    ),                 (5.0-0.2, 2.5  )]])
#
#    # an order 3 diagram (high degree):
#    draw_diagram(img, [[(3, 3+0.3),                 (5.0-0.2, 4.5  )],
#                       [(3, 3+0.0),                 (5.0-0.2, 4.5  )],
#                       [(3, 3-0.3),                 (5.0-0.2, 4.5  )]])
#
#    # an order 5 diagram:
#    draw_diagram(img, [[(1, 3+0.2), (0, 4), (2, 5), (5.0-0.2, 6.5  )],
#                       [(1, 3-0.2),                 (5.0-0.2, 6.5  )],
#                       [            (1, 4), (2, 5),                 ]])
#
#    # translates of an order 2 diagram:
#    draw_diagram(img, [[(1,  6   ),                 (5.0-0.2, 9.5  )], 
#                       [            (3, 8),         (5.0-0.2, 9.5  )]])
#    draw_diagram(img, [[(4, 9    ), (0,10),         (5.0-0.2,11.5  )]])
#    draw_diagram(img, [[(4,12    ),                 (5.0-0.2,13.5  )], 
#                       [            (3,12),         (5.0-0.2,13.5  )]])
#    draw_diagram(img, [[(1,11    ), (3,13),         (5.0-0.2,14.5  )]])
#
#    plt.imsave(filename, img)


def draw_c(filename): 
    img = np.ones((height, width, 3), dtype=np.float32)
    fill_boxes(img, [(
        i                  % N,
        i%T
    ) for i in range(max(N, T))])
    draw_grid(img)

    # an order 1 diagram:
    draw_diagram(img, [[(5, 5    ),                 (6.0-0.4, 7.0  )]])

    # order 2 diagrams:
    draw_diagram(img, [[(0, 0+0.2),                 (0.0-0.8, 4.2  )], 
                       [            (0, 0-0.2),     (0.0-0.8, 4.2  )]])
    draw_diagram(img, [[(1, 1    ),                 (0.0-0.8, 5.2  )], 
                       [            (2, 2    ),     (0.0-0.8, 5.2  )]])
    draw_diagram(img, [[(3, 3    ), (4, 4    ),     (5.0-0.4, 7.0  )]])

    plt.imsave(filename, img)

def draw_d(filename): 
    img = np.ones((height, width, 3), dtype=np.float32)
    fill_boxes(img, [(
        i % N,
        i%T
    ) for i in range(max(N, T))])
    draw_grid(img)

    # an order 1 diagram:
    draw_diagram(img, [[(5,12    ),                 (6.0-0.4,14.0  )]])

    # old order 2 diagrams:
    draw_diagram(img, [[(0, 0    ),                 (0.0-0.8, 8.0  )], 
                       [            (0, 7    ),     (0.0-0.8, 8.0  )]])
    draw_diagram(img, [[(2, 2    ),                 (0.0-0.8,10.5  )], 
                       [            (1, 8    ),     (0.0-0.8,10.5  )]])
    draw_diagram(img, [[(3, 3    ), (4,11    ),     (5.0-0.4,14.0  )]])

    # new order 2 diagram:
    draw_diagram(img, [[(6, 6    ), (6,13    ),     (7.0-0.4,14.0  )]])

    plt.imsave(filename, img)

def draw_e(filename): 
    img = np.ones((height, width, 3), dtype=np.float32)
    fill_boxes(img, [(
        (i * (1 if int(i//N)==0 else 3)) % N,
        i%T
    ) for i in range(max(N, T))])
    draw_grid(img)

    draw_diagram(img, [[(3, 3-0.3), (3, 3+0.3),     (N-0.3  , 5.0  )]])
    draw_x(img, 2.8, 2.8, 3.2, 3.2)
    queue_text(2.75,3.5,  'intra-cell edges\nare forbidden')

    draw_diagram(img, [[(6, 6    ), (6,  10   ),     (N-0.3  , 11.0  )]])
    queue_text(6.25, 8.5, 'okay embedding')

    draw_diagram(img, [[( 4,12 ), (7,  13   ),     (N-0.3  , 15.0  )]])
    queue_text(4.25,  12.5  , 'okay embedding')

    plt.imsave(filename, img)

def draw_f(filename): 
    img = np.ones((height, width, 3), dtype=np.float32)
    fill_boxes(img, [(
        (2*i) % N, 
        i%T
    ) for i in range(max(N, T))])
    fill_boxes(img, [(
        (2*i+1) % N, 
        i%T
    ) for i in range(max(N, T))])
    draw_grid(img)

    # an order 4 diagram:
    draw_diagram(img, [[(5, 2+0.2),         (1,12), (N-3.3, T+0.2)],
                       [(5, 2-0.2),                 (N-3.3, T+0.2)],
                       [            (2, 5), (1,12),              ]])
    queue_text(5.5 ,   2+1.5, 'this cell...', color=colors[5])
    queue_text(5.5 ,   7.25 , '...affects...', color=colors[5])
    queue_text(8.25,  10    , '...and affects...', color=colors[5])

    queue_text(2   ,   5+1  , 'this cell...', color=colors[2])
    queue_text(3.5 ,   9    , '...affects...', color=colors[2])

    queue_text(1   ,  12+1  , '...this cell, which...', color=colors[1])
    queue_text(3.5 ,  16    , '...affects...', color=colors[1])

    queue_text(N-1.2 ,T-2.5 , '...the test\nmeasurement')

    plt.imsave(filename, img)

FILE_NM = 'spacetime-e.png' 
title = "Singleton Batches with Shuffling"
#title = "Size-Two Batches without Shuffling"
draw_e(FILE_NM)

#=============================================================================

im = Image.open(FILE_NM)
draw = ImageDraw.Draw(im)

def centered_text(n, t, s, c=black, rot=0):
    ll = max(map(len, s.split('\n')))
    #t = t - 0.095*len(s)
    t = t - 2*0.095*ll
    draw.text((get_c(t), get_r(n)), s, font=fnt,
        fill=(int(255*c[0]), int(255*c[1]), int(255*c[2]), 255),
        align='center'
    )

#centered_text(N+0.7, T/2.0, title)
centered_text(N+1.4, T/2.0, title)
for t in range(0, T, dT):
    centered_text(0.0, t+0.5, "t=%d"%t)
for n in range(0, N, dN):
    #centered_text(n+0.7, -0.5, "n=%d"%n)
    centered_text(n+0.7, -0.9, "n=%d"%n)

for (n, t, s, c, rot) in texts:
    centered_text(n, t, s, c, rot)

im.save(FILE_NM)
