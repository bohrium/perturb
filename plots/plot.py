''' author: samuel tenka
    change: 2019-03-25 
    create: 2019-03-25 
    descrp: render SGD diagrams   
'''

import matplotlib.pyplot as plt
import numpy as np
from skimage.draw import circle, circle_perimeter_aa, line_aa

def draw_line_aa(img, row_s, col_s, row_e, col_e, color=(0.0, 0.0, 0.0)):
    expanded_color = np.expand_dims(np.expand_dims(np.array(color), 0), 0)
    rr, cc, val = line_aa(row_s, col_s, row_e, col_e)
    img[rr, cc, :] = np.expand_dims(val, 2)
    img[rr, cc, :] = img[rr, cc, :] * expanded_color

black = (0.0, 0.0, 0.0)
red   = (0.8, 0.2, 0.2)
green = (0.2, 0.8, 0.2)
blue  = (0.2, 0.2, 0.8)
gold  = (0.9, 0.7, 0.0)
colors = [red, green, blue, gold]

def draw(filename, margin=8): 
    img = np.ones((256, 256, 3), dtype=np.float32)
    draw_line_aa(img, 8, 8, 256-8, 8, color=black)
    draw_line_aa(img, 256-8, 8, 256-8, 256-8, color=black)
    plt.imsave(filename, img)

draw('blank.png')
