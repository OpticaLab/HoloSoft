#!/usr/bin/python3
# -*- coding: utf-8 -*-

from pylab import *
import numpy as np
import matplotlib.pyplot as plt
import holopy as hp
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
import argparse
from HoloSoft import *

parser = argparse.ArgumentParser("Find Centers")
parser.add_argument('-fd','--folder_dir', required=True, type=str, help="Folder Directory")
parser.add_argument('-sd','--stack_dir', required=True, type=str, help="Working Stack Directory")
parser.add_argument('-im','--image_name', required=True, type=str, help="Image name")
parser.add_argument('-med','--median_name', required=True, type=str, help="Median name")
parser.add_argument('-par1', '--par1_deconv', required=True, type=float, help="Parameter for the filter deconvolution func.")
parser.add_argument('-par2', '--par2_deconv', required=True, type=float, help="Parameter for the filter deconvolution func.")

args = parser.parse_args()

#class args:
#    folder_dir = "RICE/135/"
#    stack_dir = "1/2/"
#    image_name = "IMG_0041.tiff"
#    median_name = "median_20_40.tiff"
#    par1_deconv = 0.0509
#    par2_deconv = 0.00090


medium_index = 1.33
pixel_size = 0.236
illum_wavelen = 0.6328
illum_polarization =(0,1)
k=np.pi*2/(illum_wavelen/medium_index)
r = 2
Lx = 1024
Ly = 1280


data_path = args.folder_dir + args.stack_dir +"dati/" + args.image_name
bg_holo = hp.load_image(args.folder_dir + args.stack_dir +"mediana/" + args.median_name , spacing = pixel_size)
  
N = 1024

raw_holo = hp.load_image(data_path, spacing = pixel_size)
data_holo = raw_holo/(bg_holo+1)
print(np.std(data_holo))
data_holo = data_holo/(np.amax(data_holo))*255

data_holo = data_holo[0,0:N,0+128:N+128]    #The image must be square for the FFT        
data_holo = data_holo -1

t = 0.3
centri = filter_deconv(N, data_holo, pixel_size, args.par1_deconv, args.par2_deconv) 
centri = centri/np.amax(centri)
image_max = ndi.maximum_filter(centri, size=50, mode='constant')
coordinates = peak_local_max(centri, min_distance=200, threshold_abs=t)
       
fig, axes = plt.subplots(1, 3, figsize=(10, 5), sharex=True, sharey=True)
ax = axes.ravel()
            
im1 = ax[0].imshow(data_holo, cmap=plt.cm.gray)            
ax[0].set_title('Original')            

im2 = ax[1].imshow(centri, cmap=plt.cm.gray)                        
ax[1].set_title('Maximum filter')

im3 = ax[2].imshow(data_holo, cmap=plt.cm.gray)
ax[2].autoscale(False)
ax[2].plot(coordinates[:, 1], coordinates[:, 0], 'r.')
ax[2].set_title('Peak local max')

fig.tight_layout()
plt.show()           
plt.clf()
plt.close()
