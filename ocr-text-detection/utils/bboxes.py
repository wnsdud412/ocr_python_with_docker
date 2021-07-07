import numpy as np
from numpy.linalg import norm
eps = 1e-10


def polygon_to_rbox3(xy):
    # two points at the center of the left and right edge plus heigth
    tl, tr, br, bl = xy
    # length of top and bottom edge
    dt, db = tr-tl, bl-br
    # height is mean between distance from top to bottom right and distance from top edge to bottom left
    h = (norm(np.cross(dt, tl-br)) + norm(np.cross(dt, tr-bl))) / (2*(norm(dt)+eps))
    p1 = (tl + bl) / 2.
    p2 = (tr + br) / 2. 
    return np.hstack((p1,p2,h))


def rbox3_to_polygon(rbox):
    x1, y1, x2, y2, h = rbox
    alpha = np.arctan2(x1-x2, y2-y1)
    dx = -h*np.cos(alpha) / 2.
    dy = -h*np.sin(alpha) / 2.
    xy = np.reshape([x1-dx,y1-dy,x2-dx,y2-dy,x2+dx,y2+dy,x1+dx,y1+dy], (-1,2))
    return xy

