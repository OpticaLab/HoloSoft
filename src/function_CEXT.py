#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt
from holopy.scattering import Sphere
from holopy.scattering import calc_cross_sections
from scipy.signal import argrelextrema
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import InsetPosition


def matrix_R(N, pixel_size):
    """
    Measurements of the positions matrix and of the angles matrix

    Args
    -------------------------------------------------------
    N (int):
        Shape of the hologram
    pixel_size (float):
        Value of the pixel size (um)

    Returns
    -------------------------------------------------------
    R (:class:`.Image` or :class:`.VectorGrid`):
       Matrix of positions 
    A (:class:`.Image` or :class:`.VectorGrid`):
       Matrix of angles (°) 
    """
    onesvec = np.ones(N)
    inds = (np.arange(N)+.5 - N/2.) / (N-1.)
    X = np.outer(onesvec, inds)
    Y = np.transpose(X)
    R = np.sqrt(X**2. + Y**2.) * pixel_size * N
    A = -np.arctan2(X, Y)*180/np.pi+180
    return R, A


def Integration_tw_square(holo, lim, pixel_size):
    """
    Integration of the hologram from the center point to the edges throw square

    Args
    -----------------------------------------------------
    holo (:class:`.Image` or :class:`.VectorGrid`):
       Hologram in function of x,y
    lim (int):
        Half length of the hologram (center of the hologram)
    pixel_size (float):
        Value of the pixel size (um)

    Returns
     -----------------------------------------------------
    Integral_array (float or list of floats):
       Integration of the hologram tw square 
    """
    Integral_array = np.array([])
    for r in np.arange(0, int(lim), 1):
        Integral = np.sum(
            holo[int(lim-r):int(lim+r+1), int(lim-r):int(lim+r+1)])*pixel_size**2
        Integral_array = np.append(Integral_array, Integral)
    return Integral_array


def Cext_tw_integration(medium_index, illum_wavelen, illum_polarization, Integral_array,  name_graph, data, x_sec):
    """
    Plot of the integration of the hologram. 
    By this you can have 
    1) Cext
    2) The real part of S(0) with the Optical Theorem

    Args
    -----------------------------------------------------
    Integral_array (float or list of floats):
       Integration of the hologram, can be tw circle or square 
    graph name (str):
        Name of the image savedù
    data (str):
        If the object is calibrated you may want a measurement of expected Cext 

    Returns
    ------------------------------------------------------
    y[0]: (float)
        Value of the Cext
    """
    Integral_array = -Integral_array[:]
    x = np.arange(0, len(Integral_array), 1)

 
    envelope_sup = argrelextrema(Integral_array[:], np.greater)[0]
    envelope_min = argrelextrema(Integral_array[:], np.less)[0]

    envelope_min = envelope_min[envelope_min > 45]
    envelope_sup = envelope_sup[envelope_sup > 35]

    if len(envelope_sup) > 0 and len(envelope_min) > 0:
        if len(envelope_sup) > len(envelope_min):
            x2 = envelope_sup[0:len(envelope_min)]
            y = ((Integral_array[envelope_sup[0:len(envelope_min)]] -
                  Integral_array[envelope_min])/2+Integral_array[envelope_min])
        if len(envelope_sup) < len(envelope_min):
            x2 = envelope_sup
            y = ((Integral_array[envelope_sup]-Integral_array[envelope_min[0:len(
                envelope_sup)]])/2+Integral_array[envelope_min[0:len(envelope_sup)]])
        if len(envelope_sup) == len(envelope_min):
            x2 = envelope_sup
            y = ((Integral_array[envelope_sup] -
                  Integral_array[envelope_min])/2+Integral_array[envelope_min])
        
        x1 = np.arange(5, envelope_sup[0]-10, 1)
        x3 = np.arange(5, envelope_sup[0]-10, 1)
        y2 = np.ones(len(x3))*((Integral_array[envelope_sup[0]] -
                                Integral_array[envelope_min[0]])/2+Integral_array[envelope_min[0]])

        plt.figure(figsize=(12, 8))
        plt.plot(x, Integral_array, 'b', linewidth=2, alpha=0.5)

        if data == 'poli':
            plt.plot(x1, np.ones(len(x1)) *
                     x_sec[2].values, '<--g', label='Cext Expected')

        plt.plot(
            envelope_sup, Integral_array[envelope_sup], '-.k', linewidth=1.5, label='Envelope')
        plt.plot(envelope_min,
                 Integral_array[envelope_min], '-.k', linewidth=1.5)
        plt.plot(x3, y2, '<--r', label='Cext Obtained')
        plt.plot(x2, y, '--.r', linewidth=2)
        
        plt.title('Cext = {:.2f}'.format(y[0]))
        plt.xlabel('x (pixel)', fontsize=20)
        plt.ylabel('Integration ($\mu m^2$)', fontsize=20)
        plt.legend(fontsize=18)
        plt.tick_params(axis='both', which='both', labelsize=18)
        plt.savefig(name_graph)
        plt.clf()
        plt.close()

    else:
        y = np.array([0])

    return (y[0]) 
