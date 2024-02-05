#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
import numpy as np
import holopy as hp

def rebin(a, shape, pixcombine):
    """
    Reshape an array (matrix) to a give size using either the sum, mean or median of the
    pixels binned.
    Note that the old array dimensions have to be multiples of the new array
    dimensions.
    
    Args
    ------------------------------------------------------
    a (:class:`.Image` or :class:`.VectorGrid`):
        Array to reshape (combine pixels)
    shape (int, int):
        New size of array
    pixcombine (str):
        The method to combine the pixels. Choices are sum, mean and median
        
    Returns
    ------------------------------------------------------
    reshaped_array (:class:`.Image` or :class:`.VectorGrid`):
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
        

    
def find_the_new_lim(center_x, center_y, lim, lim1, lim2, Lx, Ly):
    """[summary]

    Args:
    ------------------------------------------------------
    center_x (int):
        X center of the hologram
    center_y (int):
        Y center of the hologram
    lim (int or float):
        Initial half length of the hologram
    lim1 (int or float):  
        Check for the orizontal extrema
    lim2 (int or float):  
        Check for the vertical extrema
    N (int):
        Size of the hologram

    Returns:
    ------------------------------------------------------
    lim(float):
        New half length of the hologram
    """    
    if center_x < center_y:
        if center_x - lim < 0:
            lim1 = int(center_x)
        if center_x + lim > Lx:
            lim1 = int(Lx - center_x)
        if center_y - lim < 0:
            lim2 = int(center_y)
        if center_y + lim > Ly:
            lim2 = int(Ly - center_y)
                                    
        if lim1 < lim2:
            lim = lim1
        else:
            lim = lim2
            
    else:
        if center_x - lim < 0:
            lim1 = int(center_x)
        if center_x + lim > Lx:
            lim1 = int(Lx - center_x)
        if center_y - lim < 0:
            lim2 = int(center_y)
        if center_y + lim > Ly:
            lim2 = int(Ly- center_y)
        
        if lim1 < lim2:
            lim = lim1
        else:
            lim = lim2                                  
    return lim
    
    
def make_the_fold(name_fold):
    """
    Make a non-existing fold

    Args:
    ------------------------------------------------------
    nome_cartella (str):
        Name of the fold

    Returns:
    ------------------------------------------------------
    0
    """     
    try: 
        os.stat(name_fold) 
    except: 
        os.makedirs(name_fold)
    return 0
