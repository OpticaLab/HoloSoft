#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from pylab import *
from scipy.ndimage import measurements
from PIL import Image
import cv2
import imutils
from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import holopy as hp


def propagation_module(z, rec_vol, limx, limy, num):
    """
    Calculates the module of the hologram along the optical axis, only in the 
    center of the hologram.

    Args
    -------------------------------------------------------
    rec_vol (:class:`.Image` or :class:`.VectorGrid`):
       Matrix field in function of x,y,z
    z (float or list of floats):
       Distance to propagate. 
    lim (int):
        Half length of the hologram (center of the hologram)

    Returns
    -------------------------------------------------------
    module_arr (np.array):
       The module of the hologram progagated to a distance z from its current
       location calculated at the center of the hologram.
    """
    module_arr = np.array([])
    for j in range(0, len(z)):
        if num != 0:
            module = np.abs(rec_vol[limx-num:limx+num, limy-num:limy+num, j])
            module = np.mean(module)
        else:
            module = np.abs(rec_vol[limx, limy, j])
        module_arr = np.append(module_arr, module)

    return module_arr


def propagation_phase(phase, p, z, limx, limy, num):
    """
    Calculates the phase of the hologram along the optical axis, only in the 
    center of the hologram.

    Args
    -------------------------------------------------------
    phase (:class:`.Image` or :class:`.VectorGrid`):
        Phase of the Hologram in function of x,y,z
    p (float or list of floats):
        Reference wave, that hasn't scattered
    z  (float or list of floats):
       Distance to propagate. 
    lim (int):
        Half length of the hologram (center of the hologram)

    Returns
    -------------------------------------------------------
    phase_arr (float or list of floats):
       The phase of the hologram progagated to a distance z from its current
       location calculated at the center of the hologram with respect to the
       reference wave.
    """
    p_arr = np.array([])
    phase_arr = np.array([])

    only_phase = np.array([])

    for j in range(0, len(z)):
        if num != 0:
            diff = phase[limx-num:limx+num, limy-num:limy+num, j] - p[j]
            diff = np.mean(diff)
            only_phase = np.append(only_phase, np.mean(
                phase[limx-num:limx+num, limy-num:limy+num, j]))

        else:
            diff = phase[limx:limx, limy:limy, j] - p[j]
            only_phase = np.append(only_phase, np.mean(
                phase[limx:limx, limy:limy, j]))

        phase_arr = np.append(phase_arr, diff)

        p_arr = np.append(p_arr, p[j])

        phase_arr[phase_arr > np.pi] = 0
        phase_arr[phase_arr < -np.pi] = 0

    return phase_arr, p_arr, only_phase


def maximum_minimum(array, z):
    """
    Calculates the max and min value of an array

    Args
    -------------------------------------------------------
    array (float or list of floats):
        Array of which you want calculate the extremes
    z (float or list of floats):
        Distance to propagate  

    Returns
    --------------------------------------------------------
    d_max (float):
        Distance[pixels] at which the array have the maximum value
    d_min (float):
        Distance[pixels] at which the array have the minimun value
    z_max (int):
        Array position at which the array have the maximum value
    z_min (int):
        Array position at which the array have the minium value    
    """
    max_array = np.amax(array)
    d_max = z[np.where(array == max_array)[0]]
    z_max = np.where(array == max_array)[0]

    min_array = np.amin(array)
    d_min = z[np.where(array == min_array)[0]]
    z_min = np.where(array == min_array)[0]
    return d_max, d_min, z_max, z_min


def plot_twin_propagation(z, module_arr, phase_arr, directory_graph):
    """
    Calculates the plot of the propagation of the hologram along the optical
    axis and at the center of the hologram both studying the module and the phase of the field. 

    Args
    -------------------------------------------------------
    z (float or list of floats):
        Distance to propagate  
    module_arr (float or list of floats):
        Array of the intensity of the field propagated
    phase_arr (float or list of floats):
        Array of the phase of the field propagated
    directory_graph (str):
        Path where the graph is saved         

    Returns
    -------------------------------------------------------
    0: the graph is saved authomatically.
        By the graph the point of discontinuity can be seen and it can be possible
        calculate the z position of the particle     
    """
    fig, ax1 = plt.subplots()
    ax1.plot(z, module_arr, '-b*', label='module')

    ax1.set_xlabel('z($\mu$m)')
    ax1.set_ylabel('|U|', color='b')
    ax1.tick_params('y', colors='b')

    ax2 = ax1.twinx()
    phase_arr[phase_arr == 0] = None
    ax2.plot(z, phase_arr, '-r*', label='phase')
    ax2.set_ylabel('$\phi$(U)', color='r')
    ax2.tick_params('y', colors='r')

    plt.title("Propagation")
    plt.savefig(directory_graph)
    plt.clf()
    plt.close()
    return 0



def midpoint(ptA, ptB):
    """
    Calculates the middle point of two point 

    Args
    -------------------------------------------------------
    ptA (int):
       Position A
    ptB (int):
       Position B

    Returns
    --------------------------------------------------------
    The middle point: (float)
    """
    return (ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5


def object_dimension(directory_obj, pixel_size,  lim, area, dim, name_save):
    """
    Calculates the diameter of an object, spherical-shaped or not.

    It first performs edge detection, then performs a dilation + erosion to
    close gaps between object edges.
    Then for each object in the image, it calcalutes the contourns of the
    minimum box (minimum rectangle that circumvent the object) and it sorts
    them from left-to-right (allowing us to extract our reference object).
    It unpacks the ordered bounding box and computes the midpoint between the
    top-left and top-right coordinates, followed by the midpoint between
    bottom-left and bottom-right coordinates.
    Finally it computes the Euclidean distance between the midpoints.

    Args
    -------------------------------------------------------
    directory_obj (str):
       Path of the directory of the image of the object reconstructed at the 
       focal point.
    pixel_size (float):
        Value of the pixel size (um)
    lim (int):
        Value of the half shape of the new matrix (the new center)
    name_save (str):
        Name of the image saved

    Returns
    -------------------------------------------------------
    dimS (float):
        Value of the the smaller diameter
    dimL (float):
        Value of the the longer diameter
    ratio (float):
        Value of the ratio of the two diameters
    """
    image = cv2.imread(directory_obj, cv2.IMREAD_GRAYSCALE)
    _, threshold = cv2.threshold(image, 110, 255, cv2.THRESH_BINARY)
    contours= cv2.findContours(threshold.copy(), cv2.RETR_TREE,
                                  cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(contours)

    dimS = np.array([])
    dimL = np.array([])
    ratio_array = np.array([])

    if cnts != []:
        # sort the contours from left-to-right
        # (cnts, _) = contours.sort_contours(cnts)
        n = 0
        dimS = np.array([])
        dimL = np.array([])
        ratio_array = np.array([])

        for c in cnts:
            # compute the rotated bounding box of the contour
            orig = image.copy()
            box = cv2.minAreaRect(c)
            box = cv2.cv.BoxPoints(
                box) if imutils.is_cv2() else cv2.boxPoints(box)
            box = np.array(box, dtype="int")
        # order the points in the contour such that they appear in top-left
            box = perspective.order_points(box)
        # Compute the midpoint
            (tl, tr, br, bl) = box
            (tltrX, tltrY) = midpoint(tl, tr)
            (blbrX, blbrY) = midpoint(bl, br)
            (tlblX, tlblY) = midpoint(tl, bl)
            (trbrX, trbrY) = midpoint(tr, br)

        # compute the Euclidean distance between the midpoints
        # dA  variable will contain the height distance (pixels)
        # dB  will hold our width distance (pixels).
            dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
            dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))

            if dA != 0 and dB != 0:
                # compute the size of the object
                dimA = dA * pixel_size
                dimB = dB * pixel_size

                major = max(dimA, dimB)
                minor = min(dimA,dimB)
                ratio = minor/major

                plt.plot([tltrX,blbrX],[tltrY,blbrY], color = "yellow", linewidth=3)
                plt.plot([tlblX,trbrX] ,[tlblY, trbrY], color = "lime", linewidth=3)
                plt.imshow(orig, cmap = "gray")  
                plt.text(dim -20,10,"major:"+ str(round(major,2))+" $\mathrm{\mu}$m", color="w")
                plt.text(dim-20,20,"minor:"+str(round(minor,2))+" $\mathrm{\mu}$m", color="w")
                plt.colorbar()
                plt.axis('off')
                plt.savefig(name_save+'_'+str(n)+'.pdf')
                plt.close()

               
            else:
                major = 0
                minor= 0
                ratio = 0
                dimS = np.append(dimS, minor)
                dimL = np.append(dimL, major)
                ratio_array = np.append(ratio_array, ratio)

            n = n+1
            dimS = np.append(dimS, minor)
            dimL = np.append(dimL, major)
            ratio_array = np.append(ratio_array, ratio)
    else:
        major = 0
        minor= 0
        ratio = 0
        dimS = np.append(dimS, minor)
        dimL = np.append(dimL, major)
        ratio_array = np.append(ratio_array, ratio)

    return dimS, dimL, ratio_array
