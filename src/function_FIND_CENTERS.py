#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import holopy as hp
from HoloSoft import *

def filter_deconv(Lx, Ly, img, pix_size, A, B):
    """
    Template matching between the normalized hologram and a prior function
    
    Args:
    -------------------------------------------------------
        Lx (int):
            X-size of the hologram
        Ly (int):
            Y-size of the hologram
        img (:class:`.Image` or :class:`.VectorGrid`):
            Matrix of the contrast hologram
        pix_size (float):
            Value of the pixel size (um)
        A (float):
            First parameter of the function
        B (float):
            Second parameter of the function


    Returns:
    -------------------------------------------------------
        map_centers (:class:`.Image` or :class:`.VectorGrid`):
            Matrix of the deconvolved hologram
    """
    h = Ly/Lx
    x  = (np.arange(Ly)+.5 - Ly/2.) /(Ly-1.)
    y  = (np.arange(Lx)+.5 - Lx/2.) /(Lx-1.)
    X, Y = np.meshgrid(x, y)
    
    R  = np.sqrt((h*X)**2. + Y**2.) # * pix_size
      
    # centri = center_find(img, centers=2, threshold=A, blursize=B)                               
    # if (centri[0]-centri[1]).all()<50:
    #     centri = center_find(img, centers=1, threshold=A, blursize=B)
    # map_centers = centri
    K =  np.exp(-(R**2)/A)*(np.cos(R**2/B))
    im_f = np.fft.fft2(np.fft.fftshift(img))
    kk_f = np.fft.fft2(np.fft.fftshift(K))
    conv_f = im_f * kk_f
    map_centers = np.abs(np.fft.fftshift(np.fft.ifft2(conv_f)))**2


    return map_centers
