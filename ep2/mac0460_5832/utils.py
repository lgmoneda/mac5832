# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import PIL
import scipy.ndimage as mm

def read_img(filename):
    img = np.asarray(PIL.Image.open(filename))
    if img.dtype == bool:
        return img
    return img > 127

def config_ax(ax, W, H):
    ax.set_xlim(0,W)
    ax.set_ylim(0,H)
    ax.xaxis.set_major_locator(plt.MultipleLocator(1.0))
    ax.yaxis.set_major_locator(plt.MultipleLocator(1.0))
    ax.grid(which='major', axis='x', linewidth=0.75, linestyle='-', color='0.75')
    ax.grid(which='major', axis='y', linewidth=0.75, linestyle='-', color='0.75')
    ax.set_xticklabels([])
    ax.set_yticklabels([])

def draw_img(img):
    H, W = img.shape
    
    if W <= 20:
        ax = plt.gca()
        config_ax(ax, W, H)
        ax.imshow(img, cmap="binary", interpolation='none', extent= [0, W, 0, H])
        plt.show()
        return
    
    # desenha imagens maiores plotando sem o grid
    plt.imshow(img, cmap='binary', interpolation='none')
    plt.show()
    
def draw_img_pair(img1, img2, figsz=(10.4, 4.8) ):
    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=figsz)
    
    H1, W1 = img1.shape
    H2, W2 = img2.shape
    if W1 <= 20 and W2 <= 20:
        config_ax(ax1, W1, H1)
        ax1.imshow(img1, cmap="binary", interpolation='none', extent=[0, W1, 0, H1])
        config_ax(ax2, W2, H2)
        ax2.imshow(img2, cmap="binary", interpolation='none', extent=[0, W2, 0, H2])
        plt.show()
        return
    ax1.imshow(img1, cmap='binary', interpolation='none')
    ax2.imshow(img2, cmap='binary', interpolation='none')
    plt.show()

def intersect_img(img1, img2):
    i = np.minimum(img1, img2)
    return i.astype(img1.dtype)

def union_img(img1, img2):
    u = np.maximum(img1, img2)
    return u.astype(img1.dtype)

def sub_img(f1, f2):
    return np.clip(f1 - f2, 0,1)

def invert_img(img):
    ret = np.copy(img)
    ret[img == 0] = 1
    ret[img == 1] = 0
    return ret

# A criação dos structuring elements e interface para as funções de morfologia
# do scipy foram copiadas de https://github.com/dennisjosesilva/vpi/
############# Structure elements creation  ##################
def se_disk(r=1):
    v = np.arange(-r, r+1)
    x = np.resize(v, (len(v), len(v)))
    y = np.transpose(x)
    be = np.sqrt(x*x + y*y)<=(r+0.5)
    return be >= 1
    
def se_cross(r=1):
    cross = np.array([[0,1,0],[1,1,1],[0,1,0]])
    if r > 1:
        shape = (3*r-(r-1), 3*r-(r-1))
        se  = np.zeros(shape)
        center = (shape[0] // 2, shape[1] // 2)
        indices = np.arange(-1, 2)

        for i in np.arange(3):
            for j in np.arange(3):
                se[indices[i]+center[0], indices[j]+center[1]] = cross[i,j]

        return dilation(se, cross, r-1)
    
    return cross

def se_box(r=1):
    if r == 1:
        return np.ones((3,3), np.bool)
    else:
        return np.ones((3*r-(r-1), 3*r-(r-1)), np.bool)

############# Morphology operators  ######################
def dilation(f, b=se_cross(), iterations=1):
    return mm.binary_dilation(f, b, iterations)

def erosion(f, b=se_cross(), iterations=1):
    return mm.binary_erosion(f, b, iterations)

def opening(f, b=se_cross()):
    return mm.binary_opening(f, b)

def closing(f, b=se_cross()):
    return mm.binary_closing(f,b)
