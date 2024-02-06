#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
This program analyses ice core dust holograms  
"""

import os
from pylab import *
import numpy as np
import matplotlib.pyplot as plt
import holopy as hp
from scipy.signal import argrelextrema
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from scipy.ndimage import gaussian_filter
from PIL import Image
import argparse
from scipy.ndimage import measurements
from holopy.scattering import calc_scat_matrix, Sphere
from HoloSoft import *

"""
Parser for run all the folds. Class args for one fold.
"""
parser = argparse.ArgumentParser("Mineral Dust")
parser.add_argument('-fd','--folder_dir', required=True, type=str, help="Folder Directory")
parser.add_argument('-sd','--stack_dir', required=True, type=str, help="Working Stack Directory")
parser.add_argument('-pf','--path_file', required=True, type = str, help='Name file path.dat')
parser.add_argument('-np','--number_path', required=True, type = str, help='Number file path.dat')
parser.add_argument('-wvl', '--laser_wvl', required=True, type=float, help="wvl of laser")
parser.add_argument('-pix', '--pixel_size', required=True, type=float, help="Pixel size")
parser.add_argument('-interval', '--interval', required=True, type=int, help="Interval of the median")
parser.add_argument('-st', '--std_dev', required=True, type=float, help="Standard deviation cut off")
parser.add_argument('-lim', '--lim_img', required=True, type=float, help="Shape cut image")
parser.add_argument('-zed', '--prop_z', required=True, type=int, help="Start range for propagation in z")
parser.add_argument('-msk', '--mask_tresh', required=True, type=float, help="Mask treshold")
parser.add_argument('-par1', '--par1_deconv', required=True, type=float, help="Parameter for the filter deconvolution func.")
parser.add_argument('-par2', '--par2_deconv', required=True, type=float, help="Parameter for the filter deconvolution func.")
parser.add_argument('-dimx', '--dim_img_x', required=True, type=int, help="X-dimension of raw image")
parser.add_argument('-dimy', '--dim_img_y', required=True, type=int, help="Y-dimension of raw image")
parser.add_argument('-sample', '--sample_name', required=True, type=str, help="Polystyrene or not")
parser.add_argument('-ray', '--ray', required=True, type=float, help="Ray polystyrene, if not ray = 0")
parser.add_argument('-area', '--area_max', required=True, type=int, help="Area max threshold to save")
parser.add_argument('-cext', '--cext_min', required=True, type=float, help="Cext min threshold to save")


args = parser.parse_args()

#class args:
#    folder_dir = ""
#    stack_dir = ""
#    end = 
#    path_file = ""
#    number_path = ""
#    std_dev = 
#    msk_tresh = 
#    par1_deconv = 
#    par2_deconv = 


fold = args.stack_dir +"/"
sample =  args.folder_dir + "/"
sample_name= args.folder_dir+"_"+args.stack_dir

data_path = sample+fold+"data/"
data_path_list= os.listdir(data_path)
data_path_list.sort()

bg_path = sample + fold + "bg/"
bg_path_list = os.listdir(bg_path)
bg_path_list.sort()

cont_bg = 0
cont_data = 0
numb = 1 # check on holograms number
ray = args.ray

file1 = args.path_file + 'single/'
file2 = args.path_file + 'double/'
file3 = args.path_file + 'triple/'

make_the_fold(file1)
make_the_fold(file2)
make_the_fold(file3)

name_file = file1 + args.number_path+'_'+ args.stack_dir+ ".dat"  #file where save data with one particles in the same cut image
name_file_2 = file2 + args.number_path+'_'+ args.stack_dir+ "_double.dat"  #file where save data with two particles
name_file_3 = file3 + args.number_path+'_'+ args.stack_dir+ "_triple.dat"  #file where save data with three particles

dati = open(name_file,'w+')
dati_2 = open(name_file_2,'w+')
dati_3 = open(name_file_3,'w+')

#############################################################################
##############################################################################
############################################################################

"Image caracteristics"
medium_index = 1.33
pixel_size = args.pixel_size #given in um
illum_wavelen = args.laser_wvl
illum_polarization =(0,1)
k=np.pi*2/(illum_wavelen/medium_index)

Lx = args.dim_img_x
Ly = args.dim_img_y

if args.sample_name == 'polystyrene':
     distant_sphere = Sphere(r=ray, n=1.59)
     x_sec = calc_cross_sections(
         distant_sphere, medium_index, illum_wavelen, illum_polarization)
else:
    x_sec = 0

for i in data_path_list:
        
        """
        --OPEN IMG--
        """
        
        raw_holo = hp.load_image(data_path + data_path_list[cont_data], spacing = pixel_size,medium_index =medium_index ,illum_wavelen=illum_wavelen, illum_polarization=illum_polarization)
        
        if cont_data == 0:
            bg_holo = hp.load_image(bg_path + bg_path_list[cont_bg], spacing = pixel_size,medium_index =medium_index ,illum_wavelen=illum_wavelen, illum_polarization=illum_polarization)
        
        else:
            if np.abs(int(data_path_list[cont_data][4:8]) - int(data_path_list[cont_data-1][4:8])) < args.interval:
                
                if int(int(data_path_list[cont_data][4:8])/args.interval) == int(int(data_path_list[cont_data-1][4:8])/args.interval):
                    cont_bg  = cont_bg -1
                    bg_holo = hp.load_image(bg_path + bg_path_list[cont_bg], spacing = pixel_size,medium_index =medium_index ,illum_wavelen=illum_wavelen, illum_polarization=illum_polarization)
                
                else:
                    bg_holo = hp.load_image(bg_path + bg_path_list[cont_bg], spacing = pixel_size,medium_index =medium_index ,illum_wavelen=illum_wavelen, illum_polarization=illum_polarization)
            else:
                bg_holo = hp.load_image(bg_path + bg_path_list[cont_bg], spacing = pixel_size,medium_index =medium_index ,illum_wavelen=illum_wavelen, illum_polarization=illum_polarization)
        raw_holo = raw_holo*np.mean(bg_holo).values/np.mean(raw_holo).values
        
        if np.amin(bg_holo) != 0:
            data_holo_orig = raw_holo / (bg_holo)
        else:
            data_holo_orig = raw_holo/(bg_holo+0.01)
             
        print(i,bg_path_list[cont_bg] )
     
        graph_center = sample+fold+'centers/'
        integral =sample+fold+'integral/'
        make_the_fold(graph_center)
        make_the_fold(integral)
        
        """
        --DECONVOLUTION FILTER--
            Find the centers
        """
            
        center_holo = data_holo_orig[0,:,:]    #The image must be square for the FFT        
        center_holo = center_holo -1
        
        t = 0.3
        centers_p = filter_deconv(Lx, Ly, center_holo, pixel_size, args.par1_deconv, args.par2_deconv) 
        centers_p = centers_p/np.amax(centers_p)
        image_max = ndi.maximum_filter(centers_p, size=30, mode='constant')
        coordinates = peak_local_max(centers_p, min_distance=50, threshold_abs=t)

        """
        --DELETE NULL IMG--
        """
     
        st = np.std(data_holo_orig)
        if (np.shape(coordinates)[0]<4) and (np.shape(coordinates)[0]>0) and (st >= args.std_dev):  #If it is too many coordinates: either too many holograms or noisy image.
            img = center_holo
                
            for j in np.arange(0,len(coordinates),1):
                    """
                    --CUT THE IMAGE--
                    """
                 
                    center_x_0 = coordinates[j][0]
                    center_y_0 = coordinates[j][1]

                    lim =args.lim_img
                    lim1=args.lim_img
                    lim2=args.lim_img
            
                    lim = find_the_new_lim(center_x_0, center_y_0, lim, lim1, lim2, Lx-1, Ly-1)
                    
                    if lim >= 150:
                         data_holo = data_holo_orig -1
                         holo_cut_0 = data_holo[:,int(center_x_0 -lim) : int(center_x_0 +lim+1), int(center_y_0 -lim)  : int(center_y_0 +lim+1)  ] 
                         center_refinment = center_find(holo_cut_0, centers=1, threshold=0.5, blursize=2.0)    
                         
                         center_x = center_refinment[0]
                         center_y = center_refinment[1]
                             
                         Lx2 = lim*2
                         Ly2 = Lx2 +1
                             
                         lim =lim
                         lim1=lim
                         lim2=lim
                         
                         lim = find_the_new_lim(center_x, center_y, lim, lim1, lim2, Lx2, Ly2)
                         holo_cut = holo_cut_0[:,int(center_x-lim) : int(center_x+lim+1), int(center_y-lim)  : int(center_y+lim+1)  ]
                         
                         plt.figure(0) #find center
                         plt.figsize = (30,8)
                         plt.imshow(holo_cut[0,:,:], cmap = 'viridis')
                         plt.axis('off')
                         plt.savefig(integral+str(numb)+'img_'+str(j)+'_'+str(os.path.splitext(i)[0])+".pdf")                                
                         plt.clf()
                         plt.close()
                         
                         plt.figure(1)
                         fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharex=True, sharey=True)
                         ax = axes.ravel()
                         im1 = ax[0].imshow(img, cmap=plt.cm.gray)
                         ax[0].set_title('Original, '+'{:.4f}'.format(st.values))
                         im3 = ax[1].imshow(img, cmap=plt.cm.gray)
                         ax[1].autoscale(False)
                         ax[1].plot(coordinates[:, 1], coordinates[:, 0], 'r.', alpha = 0.5)
                         ax[1].set_title('Peak local max')
                         fig.tight_layout()
    
                         fig.savefig(graph_center+'centers_'+str(i))
                         plt.clf()
                         plt.close()
                            
                         holo = holo_cut[0,:,:]-np.mean(holo_cut).values
                         #R , A = matrix_R(int(lim*2)+1, pixel_size)
                            
                         """
                         --CEXT--tw--INTEGRATION --square
                         
                         """
                         Integration_square = Integration_tw_square(holo, lim, pixel_size)                                   
                         Cext_tw_Integration_Square = Cext_tw_integration(medium_index, illum_wavelen, illum_polarization, Integration_square, integral+str(numb)+'cext_'+str(j)+'_'+str(os.path.splitext(i)[0])+".pdf", args.sample_name, x_sec)
                         print('Cext Integrale=', Cext_tw_Integration_Square)

                         """
                         --DIMENSIONI--
                         
                         """
                         z = np.linspace(args.prop_z,args.prop_z+500, 100)
                         rec_vol = hp.propagate(holo_cut, z, illum_wavelen = illum_wavelen, medium_index = medium_index)
                                       
                         modulo = propagation_module(z, rec_vol,int(lim),int(lim), 5)**2
                         max_d_modulo, min_d_modulo, max_zarray_modulo, min_zarray_modulo = maximum_minimum(gaussian_filter(modulo, sigma =2), z)
                         
                         phase = np.angle(rec_vol)
                         ref = np.angle(np.e**((-1j*2*np.pi*z/(illum_wavelen/medium_index)))) 
                         fase, only_p, only_phase= propagation_phase(phase, ref, z, int(lim),int(lim), 1 ) 
                                       
                         fase = np.nan_to_num(fase)
                         max_d, min_d, max_zarray, min_zarray = maximum_minimum(fase, z)
                         
                         plot_twin_propagation( z, modulo, fase, integral+str(numb)+"propagation_"+str(j)+"_"+str(os.path.splitext(i)[0])+".pdf")       
                         peaks = argrelextrema(gaussian_filter(modulo, sigma =2),np.greater)[0]
                         
                         for p in np.arange(0, len(peaks),1):
                           if modulo[peaks[p]]<0.65*np.amax(modulo):
                               peaks[p]=9999
                         peaks = peaks[peaks != 9999]
                         
                         if peaks.size == 4:
                           focal_position = int(abs((peaks[1]-peaks[2]))/2 + peaks[1]) 
                         elif peaks.size>1:
                           focal_position = int(abs((peaks[0]-peaks[1]))/2 + peaks[0]) 
                           
                           if peaks.size>2:
                               peaks = argrelextrema(gaussian_filter(modulo, sigma = 4),np.greater)[0]
                               for p in np.arange(0, len(peaks),1):
                                   if modulo[peaks[p]]<0.85*np.amax(modulo):
                                       peaks[p]=9999
                               peaks = peaks[peaks != 9999]
                               
                               if peaks.size >1:
                                   focal_position = int(abs((peaks[0]-peaks[1]))/2 + peaks[0]) 
                               else:
                                   if peaks.size>0:
                                       focal_position = peaks[0] 
                                   else:
                                       focal_position = 0
                         else :
                           if peaks.size>0:
                               focal_position = peaks[0] 
                           else:
                               focal_position = 0
                           
                         if modulo[99]>0.6*np.amax(modulo) and peaks.size>0 :
                           focal_position = int(abs((peaks[0]-99))/2 + peaks[0]) 
                         
                         dim = 50
                         obj= np.abs(rec_vol[:,:,focal_position])
                         lenx = np.linspace(dim,dim +2/pixel_size,3)
                         plt.plot(lenx,np.ones(3)*5,'-w', linewidth=5)
                         plt.text(124,15, r'2$\mu m$', {'color': 'w', 'fontsize': 12})
                         plt.imshow(obj[int(lim)-dim :int(lim)+dim +1,int(lim)-dim :int(lim)+dim +1],cmap='gray')
                         plt.colorbar()
                         plt.axis('off')
                         plt.savefig(integral+str(numb)+"modulo_nofilter_"+str(j)+"_"+str(os.path.splitext(i)[0])+".pdf")                                        
                         plt.clf()
                         plt.close()

                         obj=gaussian_filter(obj, sigma=0)                                   
                         obj= obj/np.amax(obj)
                         mask= (obj[int(lim)-dim :int(lim)+dim+1 ,int(lim)-dim :int(lim)+dim +1]>args.mask_tresh).astype(int)
                         lw,num = measurements.label(mask)
                         area = (measurements.sum(mask, lw, range(num + 1)))*pixel_size*pixel_size
                         print('area=', area)
                         mask = mask*255
                         mask = mask.astype(int)
                         result = Image.fromarray((mask).astype('uint8'))
                         result.save(integral+str(numb)+"mask_"+str(j)+"_"+str(os.path.splitext(i)[0])+".tiff")

                         if len(area)<10:
                              dimA1, dimB1, ratio1 = object_dimension(integral+str(numb)+"mask_"+str(j)+"_"+str(os.path.splitext(i)[0])+".tiff", pixel_size, int(lim), area, dim, integral+str(numb)+"obj_"+str(j)+"_"+str(os.path.splitext(i)[0]))
                         else:
                              dimA1= 0;dimB1= 0;ratio1=0;
                              
                         """
                         --SAVE--DATA
                         
                         """
                            
                         if (any(area > args.area_max)) or (Cext_tw_Integration_Square <=args.cext_min) :
                              print('delete')
                              
                              os.remove(integral+str(num)+'img_'+str(j)+'_'+str(os.path.splitext(i)[0])+".pdf")
                              os.remove(integral+str(num)+"propagation_"+str(j)+"_"+str(os.path.splitext(i)[0])+".pdf")
                              os.remove(integral+str(num)+"modulo_nofilter_"+str(j)+"_"+str(os.path.splitext(i)[0])+".pdf")
                              os.remove(integral+str(num)+"mask_"+str(j)+"_"+str(os.path.splitext(i)[0])+".tiff")
                                
                              try:
                                   if np.size(dimA1)>1:
                                        for k in np.arange(0,len(dimA1),1):
                                             os.remove(integral+str(num)+"obj_"+str(j)+"_"+str(os.path.splitext(i)[0])+'_'+str(k)+".pdf")
                              except FileNotFoundError:
                                   print('no obj')
                                    
                              try:
                                   os.remove(integral+str(num)+'cext_'+str(j)+'_'+str(os.path.splitext(i)[0])+".pdf")
                              except FileNotFoundError:
                                   print('no cext')
                                   
                         else:
                                if len(area)>2:  
                                    if len(area)>3 :
                                        print(i)
                                        print('write')  
                                        try :
                                            print (str(i)+' '+str(focal_position)+' '+str(center_x_0)+' '+str(center_y_0)+' '+str(area[1])+' '+str(area[2])+' '+str(area[3])+' '+str(dimA1[0])+' '+str(dimB1[0])+' '+str(dimA1[1])+' '+str(dimB1[1])+' '+str(dimA1[2])+' '+str(dimB1[2])+' '+str(Cext_tw_Integration_Square),file=dati_3)
                                            
                                        except (TypeError,IndexError) as e:
                                            print (str(i)+' '+str(focal_position)+' '+str(center_x_0)+' '+str(center_y_0)+' '+str(area[1])+' '+str(area[2])+' '+str(area[3])+' '+str(dimA1[0])+' '+str(dimB1[0])+'  '+str(Cext_tw_Integration_Square),file=dati_3)
                                            
                                    else:
                                        print(i)
                                        print('write')
                                        try :
                                            print (str(i)+' '+str(focal_position)+' '+str(center_x_0)+' '+str(center_y_0)+' '+str(area[1])+' '+str(area[2])+' '+str(dimA1[0])+' '+str(dimB1[0])+' '+str(dimA1[1])+' '+str(dimB1[1])+' '+str(Cext_tw_Integration_Square),file=dati_2)
                                     
                                        except (TypeError,IndexError) as e:
                                            print (str(i)+' '+str(focal_position)+' '+str(center_x_0)+' '+str(center_y_0)+' '+str(area[1])+' '+str(area[2])+' '+' '+str(dimA1[0])+' '+str(dimB1[0])+'  '+str(Cext_tw_Integration_Square)+' '+str(andamento_y[len(andamento_y)-1]),file=dati_2)
                                    
                                else:
                                    print(i)
                                    print('write')  
                                    print (str(i)+' '+str(z[focal_position])+' '+str(center_x_0)+' '+str(center_y_0)+' '+str(lim)+' '+str(area[1])+' '+str(dimA1[0])+' '+str(dimB1[0])+' '+str(Cext_tw_Integration_Square)+' '+str(dx)+' '+str(dy),file=dati)
                            
                      
        numb =numb+1                                  
        cont_bg = cont_bg +1
        cont_data = cont_data +1

dati.close() 
dati_2.close()
dati_3.close()
datax.close()
datay.close()
dataz.close()
