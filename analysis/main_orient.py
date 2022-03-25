#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
This program analyses polystyrene holograms for test the setup. 
"""

import os
from pylab import *
import numpy as np
import matplotlib.pyplot as plt
import holopy as hp
from holopy.scattering import Sphere
from holopy.scattering import calc_cross_sections
from scipy.signal import argrelextrema
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from scipy.ndimage import gaussian_filter
from PIL import Image
import argparse
from scipy.ndimage import measurements
from skimage.restoration import unwrap_phase
from HoloSoft import *

"""
Parser for run all the folds. Class args for one fold.

"""

#parser = argparse.ArgumentParser("Mineral Dust")
#parser.add_argument('-fd','--folder_dir', required=True, type=str, help="Folder Directory")
#parser.add_argument('-sd','--stack_dir', required=True, type=str, help="Working Stack Directory")
#parser.add_argument('-pf','--path_file', required=True, type = str, help='Name file path.dat')
#parser.add_argument('-np','--number_path', required=True, type = str, help='Number file path.dat')
#parser.add_argument('-st', '--std_dev', required=True, type=float, help="Standard deviation cut off")
#parser.add_argument('-msk', '--mask_tresh', required=True, type=float, help="Mask treshold")
#parser.add_argument('-par1', '--par1_deconv', required=True, type=float, help="Parameter for the filter deconvolution func.")
#parser.add_argument('-par2', '--par2_deconv', required=True, type=float, help="Parameter for the filter deconvolution func.")


#args = parser.parse_args()


class args:
    folder_dir = "1"
    std_dev = 0.016
    mask_tresh = 0.4
    ray = 1
    par1_deconv = 0.0509
    par2_deconv = 0.00090

sample = args.folder_dir+"/"

data_path = sample
data_path_list= os.listdir(data_path)
data_path_list.sort()

start =0
end = 20
N = 1024
numero = 1
contatore_mediana = 0

file = args.path_file
make_the_fold(file)
name_file = file +'/'+ args.number_path+'_'+ args.stack_dir+ ".dat"  #file dove salvo dati
dati = open(name_file,'w+')

name_file2 = file +'/'+ args.number_path+'_'+ args.stack_dir+ "_spex.dat"  #file dove salvo dati
dati2 = open(name_file2,'w+')



def object_dim(directory_obj, pixel_size,  lim, area, name_save):
   
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
            dx = np.abs(tltrX-blbrX)
            dy = np.abs(tlblY-trbrY)
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
                   
                if (dimS<100) and (dimS>0) and (ratio>0) and (dimL<100)  and (dimL>0) and (len(area) <= 10) and (all(area < 500)):
                    cv2.putText(orig, "{:.1f}um".format(dimA),(int(tltrX - 5), int(tltrY - 5)), cv2.FONT_HERSHEY_SIMPLEX,0.25, (100, 100,100),1)
                    cv2.putText(orig, "{:.1f}um".format(dimB),(int(trbrX + 8), int(trbrY+ 10)), cv2.FONT_HERSHEY_SIMPLEX,0.25, (100, 100,100), 1)
                    result = Image.fromarray((orig).astype('uint8')) 
                    result.save(name_save+'_'+str(n)+'.pdf')
                else:
                    dimS= 0
                    dimL= 0
                    ratio= 0
                    dx = 0
                    dy = 0
                    dimUno = np.append(dimUno,dimS)
                    dimDue = np.append(dimDue,dimL)
                    ratio_array = np.append(ratio_array,ratio)
            else: 
                dimS= 0
                dimL= 0
                ratio= 0
                dx = 0
                dy = 0
                dimUno = np.append(dimUno,dimS)
                dimDue = np.append(dimDue,dimL)
                ratio_array = np.append(ratio_array,ratio)
                
            n=n+1
            dimUno = np.append(dimUno,dimS)
            dimDue = np.append(dimDue,dimL)
            ratio_array = np.append(ratio_array,ratio)
    else:
        dimS= 0
        dimL= 0
        ratio= 0
        dx = 0
        dy = 0
        dimUno = np.append(dimUno,dimS)
        dimDue = np.append(dimDue,dimL)
        ratio_array = np.append(ratio_array,ratio)
        
    return dimUno, dimDue, ratio_array, dx, dy

#############################################################################
##############################################################################
############################################################################

"Image caracteristics"
medium_index = 1.33
pixel_size = 0.265 # pix_size is given in um
illum_wavelen = 0.6328
illum_polarization =(0,1)
k=np.pi*2/(illum_wavelen/medium_index)
Lx = 1024
Ly = 1280

lim = 0
area=np.array([5])
for ciclo in np.arange(1,int(len(data_path_list)/end)+1,1):
    print (start)
    for i in data_path_list[start:end]:
        print(i)
        integral =sample+cartella+'integral/'
        total = 'total/'
        make_the_fold(total)
        
        for j in range(0,4):
            try:
                dimA1, dimB1, ratio1, dx, dy = object_dim(integral+str(numero)+"mask_"+str(j)+"_"+str(os.path.splitext(i)[0])+".tiff", pixel_size, int(lim), area, total+str(numero)+"obj_"+str(j)+"_"+str(os.path.splitext(i)[0])+".pdf")
            except FileNotFoundError:
                 print('no')
                  
        print (str(dimA1)+' '+str(dimB1) +' '+str(ratio1) +' '+str(dx)+' '+str(dy),file=dati)
                                                    

dati.close() 
