#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 19 16:08:33 2019

@author: claudriel
"""

import sys
import os
import matplotlib
from pylab import *
from scipy.ndimage import measurements
from PIL import Image
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
from PIL import ImageStat
import holopy as hp
from holopy.core.process import bg_correct, subimage, normalize,center_find
import cv2
import imutils
from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
from scipy.sparse import csgraph
from matplotlib.offsetbox import AnchoredText
import time

"""
All the functions that you want :)
"""

def rebin(a, shape, pixcombine):
    """
    Reshape an array (matrix) to a give size using either the sum, mean or median of the
    pixels binned.
    Note that the old array dimensions have to be multiples of the new array
    dimensions.
    Parameters
    ----------
    a: :class:`.Image` or :class:`.VectorGrid`
        Array to reshape (combine pixels)
    shape: (int, int)
        New size of array
    pixcombine: str
        The method to combine the pixels with. Choices are sum, mean and median
        
    Returns
    -------
    reshaped_array: :class:`.Image` or :class:`.VectorGrid`
        Matrix with the new shape binned
    """
    sh = shape[0], a.shape[0]//shape[0], shape[1], a.shape[1]//shape[1]
    if pixcombine == 'sum':
        reshaped_array = a.reshape(sh).sum(-1).sum(1)
    elif pixcombine == 'mean':
        reshaped_array = a.reshape(sh).mean(-1).mean(1)
    elif pixcombine == 'median':
        reshaped_array = a.reshape(sh).median(-1).median(1)

    return reshaped_array
        


def calcolo_hologram_BINNING(cartella, name, pixel_size, lim, binsize, pixcombine):
    """
    Open the image with the correspective path, it calculates the center of the
    hologram and it prints it on the console.
    Then cut the image with a fixed dimension around the center. So the new 
    center of the image is fixed.
    Finally it rebins the image.
    
    Warning: to find the center you have to open the image with holopy function.
    But you can't rebbined DataArray. So you had to open it also with PIL.
    !!!!Maybe you can correct this in a second time!!!!
    Parameters
    ----------
    cartella: str
        Name of the folder of the image
    name: str
        Number within the name of the image (without type)
    pixel_size: float
        Value of the pixel size (um)
    lim: int
        Value of the half shape of the new matrix. It will be the new center
    binsize: int
        Value of the new reshape
    pixcombine: str
        The method to combine the pixels with. Choices are sum, mean and median
        
    Returns
    -------
    data_holo: :class:`.Image` or :class:`.VectorGrid`
        Data reshaped of the hologram
    """
    raw_holo = hp.load_image("../Campioni/Flusso/"+cartella+"/img_correct/img_" + name + ".tiff", spacing = pixel_size)  
    hp.show(raw_holo)
    plt.show()
    
    im = Image.open("../Campioni/Flusso/"+cartella+"/img_correct/img_" + name + ".tiff").convert("L")
    I  = np.asarray(im)
    
    centro = center_find(raw_holo, centers=1, threshold=0.3, blursize=6.0)
    print(centro)
    data_holo = I[int(centro[0]-lim) : int(centro[0]+lim), int(centro[1]-lim) : int(centro[1]+lim)]
    
    data_holo = rebin(data_holo, ((binsize, binsize)), pixcombine)
    lim = lim/2
    
    hp.show(data_holo)
    plt.show()
    
    return(data_holo)
    

      
    
def find_the_new_lim(center_x, center_y, lim,lim1,lim2, N):
    if center_x < center_y:
        if center_x - lim < 0:
            lim1 = int(center_x)
        if center_x + lim > N:
            lim1 = int(N - center_x)
        if center_y - lim < 0:
            lim2 = int(center_y)
        if center_y + lim > N:
            lim2 = int(N - center_y)
                                    
        if lim1 < lim2:
            lim = lim1
        else:
            lim = lim2
            
    else:
        if center_x - lim < 0:
            lim1 = int(center_x)
        if center_x + lim >N:
            lim1 = int(N - center_x)
        if center_y - lim < 0:
            lim2 = int(center_y)
        if center_y + lim > N:
            lim2 = int(N- center_y)
        
        if lim1 < lim2:
            lim = lim1
        else:
            lim = lim2                                  
    return(lim)
    
    
def make_the_fold(nome_cartella):      
    try: 
        os.stat(nome_cartella) 
    except: 
        os.makedirs(nome_cartella)
    return(0)
