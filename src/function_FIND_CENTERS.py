#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 15:43:16 2020

@author: claudriel
"""

import numpy as np

def filter_deconv(N, img, pix_size, A, B):
    onesvec = np.ones(N)
    inds  = (np.arange(N)+.5 - N/2.) /(N-1.)
    X = np.outer(onesvec, inds)
    Y = np.transpose(X)
    R  = np.sqrt(X**2. + Y**2.) # * pix_size

    K =  np.exp(-(R**2)/A)*(np.cos(R**2/B))
    #K = 1/(R**2.)

    im_f = np.fft.fft2(np.fft.fftshift(img))
    kk_f = np.fft.fft2(np.fft.fftshift(K))

    conv_f = im_f * kk_f

    map_centers = np.abs(np.fft.fftshift(np.fft.ifft2(conv_f)))**2

    return map_centers