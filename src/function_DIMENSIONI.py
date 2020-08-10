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
    module_arr=np.array([])   
    for j in range(0,len(z)):        
        if num !=0:
            module=np.abs(rec_vol[limx-num:limx+num,limy-num:limy+num,j])
            module=np.mean(module)    
        else:
            module = np.abs(rec_vol[limx,limy,j])
        module_arr=np.append(module_arr,module)
        
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
    p_arr=np.array([])
    phase_arr=np.array([])

    only_phase=np.array([])
    
    for j in range(0,len(z)):
        if num !=0:
            diff=phase[limx-num:limx+num, limy-num:limy+num, j] - p[j]
            diff= np.mean(diff)
            only_phase = np.append(only_phase,np.mean(phase[limx-num:limx+num, limy-num:limy+num, j])  )
        
        else:
            diff = phase[limx:limx, limy:limy, j] - p[j]
            only_phase = np.append(only_phase,np.mean(phase[limx:limx, limy:limy, j])  )
            
        phase_arr=np.append(phase_arr,diff)
        
        p_arr = np.append(p_arr,p[j])
        
        phase_arr[phase_arr>3] = 0
        phase_arr[phase_arr<-3] = 0
              
     
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
    d_max = z[np.where(array== max_array)[0]]
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
    ax1.plot(z,module_arr, '-b*', label='module')
    
    ax1.set_xlabel('z($\mu$m)')
    ax1.set_ylabel('|U|', color='b')
    ax1.tick_params('y', colors='b') 
    
    ax2 = ax1.twinx()
    phase_arr[phase_arr == 0] = None
    ax2.plot(z,phase_arr,'-r*', label='phase')
    ax2.set_ylabel('$\phi$(U)', color='r')
    ax2.tick_params('y', colors='r')
   
    plt.title("Propagation")
    plt.savefig(directory_graph)
    plt.clf()
    plt.close()
    return 0


# def treshold(z, phase, p):
#     """
#     Calculates the phase of the hologram, with the respect of the reference
#     wave, at the initial position of the particle.
    
#     Args
#     -------------------------------------------------------
#     z: (int)
#         Position of the particle object
#     phase: (float or list of floats) 
#         Array of the phase of the field propagated
#     p: (float or list of floats) 
#         Reference wave, that hasn't scattered
        
#     Returns
#     -------------------------------------------------------
#     diff: (:class:`.Image` or :class:`.VectorGrid`)
#         Matrix of the phase hologram at the plane of the focus (object position)
#     """
#     p = p[z] * np.ones((int(len(phase/2)),int(len(phase/2))))
#     diff=phase[:,:,z] -p
#     return diff



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


def object_dimension(directory_obj, pixel_size,  lim, area, name_save):
    """
    Calculates the diameters of an object, not circular.
    
    It first performs edge detection, then performs a dilation + erosion to
    close gaps in between object edges.
    Then for each object in the image, it calcaluates the contourns of the
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
    image = cv2.imread(directory_obj)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #perform the countour
    gray = cv2.GaussianBlur(gray, (7, 7), 0) 
    edged = cv2.Canny(gray, 50, 100)
    edged = cv2.dilate(edged, None, iterations=1)
    edged = cv2.erode(edged, None, iterations=1)
    # find contours in the edge map
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)    
    cnts = imutils.grab_contours(cnts)

    dimUno = np.array([])
    dimDue = np.array([])
    ratio_array = np.array([])
    
    
    
    if cnts != []:
    # sort the contours from left-to-right
        (cnts, _) = contours.sort_contours(cnts)
        n=0
        dimUno = np.array([])
        dimDue = np.array([])
        ratio_array = np.array([])
        
        for c in cnts:
        # compute the rotated bounding box of the contour
            orig = image.copy()
            box = cv2.minAreaRect(c)
            box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
            box = np.array(box, dtype="int")
        # order the points in the contour such that they appear in top-left
            box = perspective.order_points(box)
        # Compute the midpoint
            (tl, tr, br, bl) = box
            (tltrX, tltrY) = midpoint(tl, tr)
            (blbrX, blbrY) = midpoint(bl, br)
            (tlblX, tlblY) = midpoint(tl, bl)
            (trbrX, trbrY) = midpoint(tr, br)
        # draw lines between the midpoints
            cv2.line(orig, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)), (5, 0, 255), 2)
            cv2.line(orig, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)), (255, 0, 255), 2)
        # compute the Euclidean distance between the midpoints
        # dA  variable will contain the height distance (pixels)
        # dB  will hold our width distance (pixels).
            dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
            dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))
            
            if dA !=0 and dB!=0:
                # compute the size of the object
                dimA = dA * pixel_size
                dimB = dB * pixel_size
        
                diff = dA - dB
                if diff < 0:
                    ratio = dA/dB
                    dimS = dimA
                    dimL = dimB
                else:
                    ratio = dB/dA
                    dimS = dimB
                    dimL = dimA
                    
                if (dimS<100) and (dimS>0) and (ratio>0) and (dimL<100)  and (dimL>0) and (len(area) <10) and (all(area < 500)):
                    cv2.putText(orig, "{:.1f}um".format(dimA),(int(tltrX - 5), int(tltrY - 5)), cv2.FONT_HERSHEY_SIMPLEX,0.25, (100, 100,100),1)
                    cv2.putText(orig, "{:.1f}um".format(dimB),(int(trbrX + 8), int(trbrY+ 10)), cv2.FONT_HERSHEY_SIMPLEX,0.25, (100, 100,100), 1)
                    result = Image.fromarray((orig).astype('uint8')) 
                    result.save(name_save+'_'+str(n)+'.pdf')
                else:
                    dimS=np.nan
                    dimL=np.nan
                    ratio=np.nan
                    dimUno = np.append(dimUno,dimS)
                    dimDue = np.append(dimDue,dimL)
                    ratio_array = np.append(ratio_array,ratio)
            else: 
                dimS=np.nan
                dimL=np.nan
                ratio=np.nan
                dimUno = np.append(dimUno,dimS)
                dimDue = np.append(dimDue,dimL)
                ratio_array = np.append(ratio_array,ratio)
                
            n=n+1
            dimUno = np.append(dimUno,dimS)
            dimDue = np.append(dimDue,dimL)
            ratio_array = np.append(ratio_array,ratio)
    else:
        dimS=np.nan
        dimL=np.nan
        ratio=np.nan
        dimUno = np.append(dimUno,dimS)
        dimDue = np.append(dimDue,dimL)
        ratio_array = np.append(ratio_array,ratio)
        
    return dimUno, dimDue, ratio_array
        
    
def simmetry(holo_cut, R, A, freq, lim, graph_name): 
    """Function to find the simmetry of the object startin from the hologram path

    Args
    -------------------------------------------------------
        holo_cut (:class:`.Image` or :class:`.VectorGrid`):
            Matrix data of the feature of interest of hologram
        R (:class:`.Image` or :class:`.VectorGrid`):
            Matrix of positions
        A (:class:`.Image` or :class:`.VectorGrid`):
            Matrix of angles (°)
        freq (int):
            Step of the angle
        lim (int):
            Half length of the hologram (center of the hologram)
        graph_name (str):
            Name of the graph saved

    Returns
    -------------------------------------------------------
        0
    """    
    angle = 30
    holo_forma= holo_cut[0,:,:].values
    x_forma = np.arange(0,360,30)
    x_forma2 = np.arange(0,360,angle-10)
    
    mxt1 = np.where(R<=freq*60,holo_forma,0)
    mxt2 = np.where(R <= freq*180,holo_forma,0)
    mxt2 = np.where(R > freq*60,mxt2,0)
    if R[0,int(lim)]<freq*300:
        mxt3 = np.where(R<=R[0,int(lim)],holo_forma,0)
    else:
        mxt3 = np.where(R<=freq*300,holo_forma,0)#R[0,int(lim)]
    mxt3 = np.where(R>freq*180,mxt3,0)
                                    
                                    
    value1 = np.array([])
    value2 = np.array([])
    value3 = np.array([])
    for i_forma in x_forma:
        mtx1 = np.where((A<=i_forma+angle)&(A>=i_forma),mxt1,0)
        value1 = np.append(value1, np.mean(np.abs(mtx1)))
        mtx2 = np.where((A<=i_forma+angle)&(A>=i_forma),mxt2,0)
        value2 = np.append(value2, np.mean(np.abs(mtx2)))
    for i_forma in x_forma2:   
        mtx3 = np.where((A<=i_forma+angle-10)&(A>=i_forma),mxt3,0)
        value3 = np.append(value3, np.mean(np.abs(mtx3)))
                                        
                                        
    value1 = value1/np.amax(value1)
    value2 = value2/np.amax(value2)
    value3 = value3/np.amax(value3)
    
    plt.plot(x_forma,value1,'-*',label = 'First Rings')
    plt.ylim(0,1.1)#
    plt.title('Simmetry Object')
    plt.xlabel('Angle (°C)')
    plt.ylabel('Intensity Norm.')  
    
    plt.plot(x_forma,value2,'-*', label = 'Second Rings')
    plt.ylim(0,1.1)
    plt.title('Simmetry Object')
    plt.xlabel('Angle (°C)')
    plt.ylabel('Intensity Norm.')
                                    
    plt.plot(x_forma2,value3,'-*', label = 'Third Rings')
    plt.ylim(0,1.1)
    plt.title('Simmetry Object')
    plt.xlabel('Angle (°C)')
    plt.ylabel('Intensity Norm.')
    plt.legend()
    plt.title(str("{:.2f}".format(np.std(value1)))+', '+str("{:.2f}".format(np.std(value2)))+', '+ str("{:.2f}".format(np.std(value3))))
    plt.savefig( graph_name)
    plt.clf()
    plt.close()
    
    return 0

def plot_bello(z, holo_cut, illum_wavelen ,medium_index, lim, nome_graph):
    """
    Calculates the plot of the propagation of the hologram along the optical
    axis and at the center of the hologram both studing the intensity of the
    field and the phase of the field. 
    
    Args
    -------------------------------------------------------
    z (float or list of floats):
        Distance to propagate  
    holo_cut (:class:`.Image` or :class:`.VectorGrid`):
        Matrix data of the feature of interest of hologram
    illum_wavelen (float):
        Source wavelength
    medium index (str):
        Refraction index of the medium      
    lim (int):
        Half length of the hologram (center of the hologram)
    nome_graph (str): 
        Name of the graph saved
        
    Returns
    -------------------------------------------------------
    0   
    """
    holo_cut = holo_cut +1
    rec_vol = hp.propagate(holo_cut, z, illum_wavelen = illum_wavelen, medium_index = medium_index)
                                                
    modulo = propagation_module(z, rec_vol,int(lim),int(lim), 5)
                                      
    phase = np.angle(rec_vol)
    onda_riferimento = np.angle(np.e**((-1j*2*np.pi*z/(illum_wavelen/medium_index)))) 
    fase, only_p, only_phase= propagation_phase(phase, onda_riferimento, z, int(lim),int(lim), 1 ) #int(centro[0]),int(centro[1])
                                                
    fase = np.nan_to_num(fase) 
    plot_twin_propagation(0,z, modulo, fase, nome_graph)                                
    holo_cut = holo_cut -1
    
    return 0
    
    
   