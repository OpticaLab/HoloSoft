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
parser.add_argument('-interval', '--interval', required=True, type=int, help="Interval bg")
parser.add_argument('-st', '--std_dev', required=True, type=float, help="Standard deviation cut off")
parser.add_argument('-lim', '--lim_img', required=True, type=float, help="Shape cut image")
parser.add_argument('-zeta', '--prop_z', required=True, type=int, help="Start range for propagation in z")
parser.add_argument('-msk', '--mask_tresh', required=True, type=float, help="Mask treshold")
parser.add_argument('-par1', '--par1_deconv', required=True, type=float, help="Parameter for the filter deconvolution func.")
parser.add_argument('-par2', '--par2_deconv', required=True, type=float, help="Parameter for the filter deconvolution func.")
parser.add_argument('-dimx', '--dim_img_x', required=True, type=int, help="X-dimension of raw image")
parser.add_argument('-dimy', '--dim_img_y', required=True, type=int, help="Y-dimension of raw image")
parser.add_argument('-sample', '--sample_name', required=True, type=str, help="Polystyrene or not")
parser.add_argument('-sample2', '--sample_name2', required=True, type=str, help="Two type of find centers")
parser.add_argument('-ray', '--ray', required=True, type=float, help="Ray polystyrene, if not ray = 0")
parser.add_argument('-area', '--area_max', required=True, type=int, help="Area max threshold for save")
parser.add_argument('-cext', '--cext_min', required=True, type=float, help="Cext min threshold for save")


args = parser.parse_args()

#class args:
#    folder_dir = ""
#    stack_dir = ""
#    end = 20
#    path_file = ""
#    number_path = ""
#    std_dev = 
#    msk_tresh = 
#    par1_deconv = 
#    par2_deconv = 


fold = args.stack_dir +"/"
sample =  args.folder_dir + "/"
sample_name= args.folder_dir+"_"+args.stack_dir

data_path = sample+fold+"dati/"
data_path_list= os.listdir(data_path)
data_path_list.sort()

bg_path = sample + fold + "bg/"
bg_path_list = os.listdir(bg_path)
bg_path_list.sort()

cont_bg = 0
cont_data = 0
numero = 1 #check numbers holograms
ray =  args.ray

file1 = args.path_file + 'singole/'
file2 = args.path_file + 'doppie/'
file3 = args.path_file + 'triple/'

make_the_fold(file1)
make_the_fold(file2)
make_the_fold(file3)

name_file = file1 + args.number_path+'_'+ args.stack_dir+ ".dat"  #file dove salvo dati
name_file_2 = file2 + args.number_path+'_'+ args.stack_dir+ "_doppie.dat"  #file dove salvo dati
name_file_3 = file3 + args.number_path+'_'+ args.stack_dir+ "_triple.dat"  #file dove salvo dati

dati = open(name_file,'w+')
dati_2 = open(name_file_2,'w+')
dati_3 = open(name_file_3,'w+')
#############################################################################
##############################################################################
############################################################################

"Image caracteristics"
medium_index = 1.33
pixel_size = 0.2625 #0.2596 # pix_size is given in um
illum_wavelen = 0.6328
illum_polarization =(0,1)
k=np.pi*2/(illum_wavelen/medium_index)

Lx = 1024
Ly = 1280
N = 1024

for ciclo in np.arange(1,int(len(data_path_list)/end)+1,1):
    for i in data_path_list[start:end]:

        """
        --OPEN IMG--
        """
        
        raw_holo = hp.load_image(data_path + i, spacing = pixel_size,medium_index =medium_index ,illum_wavelen=illum_wavelen, illum_polarization=illum_polarization)
        bg_holo = hp.load_image(bg_path + bg_path_list[cont_bg], spacing = pixel_size,medium_index =medium_index ,illum_wavelen=illum_wavelen, illum_polarization=illum_polarization)
        data_holo = raw_holo / (bg_holo)
        
        print(i, bg_path_list[cont_bg])

        graph_centri = sample+fold+'riconoscimento_centri/'
        integral =sample+fold+'integral/'
        make_the_fold(graph_centri)
        make_the_fold(integral)
        
        """
        --DECONVOLUTION FILTER--
            Faccion il template matching dell'immagine con una funzione a priori
        """
            
        center_holo = data_holo[0,0:N,0+128:N+128]    #The image must be square for the FFT        
        center_holo = center_holo -1
        
        t = 0.3
        centri = filter_deconv(N, center_holo, pixel_size, args.par1_deconv, args.par2_deconv ) 
        centri = centri/np.amax(centri)
        image_max = ndi.maximum_filter(centri, size=50, mode='constant')
        coordinates = peak_local_max(centri, min_distance=50, threshold_abs=t)
        
        
        """
        --DELETE NULL IMG--
        Elimino le immagini con una DEV STD inferiore di un tot, che cambierà 
        per ogni data-set. 
        Elimino le immagini senza centri
        """
    
        if np.shape(coordinates)[0]<5:  #se sono troppe coordinate: o troppi ologrammi o immagine rumorosa
            
            if np.shape(coordinates)[0]>0: #deve almeno trovarne una
                st = np.std(data_holo)
                if st >= args.std_dev: 
                    img = center_holo
                    
                    if (np.shape(coordinates)[0]>1) and (coordinates[0][0] - coordinates[1][0]<100 or coordinates[0][1] - coordinates[1][1]<100):
                        coordinates = np.array([coordinates[0]])
                        
                    for j in np.arange(0,len(coordinates),1): #ciclo sulle coordinate
                        
                            """
                            --CUT THE IMAGE--
                            attorno ad ogni ologramma in ogni immagine. Salvo solo quelle
                            con std dev > dello stesso tot di prima
                            """
                            center_x = coordinates[j][0]
                            center_y = coordinates[j][1]
        
                            lim =500
                            lim1=500
                            lim2=500
                    
                            lim = find_the_new_lim(center_x, center_y, lim, lim1, lim2, N)
                        
                            if lim > 200: #img troppo piccole scartate
                                raw_holo = hp.load_image(data_path + i, spacing = pixel_size) #la riapro, ho bisogno ora della terza dimensione
                                data_holo = raw_holo/(bg_holo)
                                
                                data_holo = data_holo[:,0:N,0+128:N+128] 
                                data_holo = data_holo -1
                                holo_cut = data_holo[:,int(center_x-lim) : int(center_x+lim), int(center_y-lim) : int(center_y+lim)] #taglio img

                                centri_correct = center_find(holo_cut, centers=1, threshold=0.5, blursize=3.0)  #trovo i centri in modo più preciso e taglio di nuovo
                                center_x = centri_correct[0]
                                center_y = centri_correct[1]
                                
                                lim_nuovo = lim
                                lim1 = lim
                                lim2 = lim
                                lim_nuovo = find_the_new_lim(center_x, center_y, lim_nuovo,lim1,lim2, lim*2)
                                lim = lim_nuovo
                                
                                holo_cut = holo_cut[:,int(center_x-lim) : int(center_x+lim), int(center_y-lim) : int(center_y+lim)]
                                
                                if len(holo_cut[0,:,:]) % 2 == 0: #img deve essere di pixel dispari
#                                    lim = lim-0.5
                                    holo_cut = holo_cut[:,int(lim-lim+1) : int(lim+lim), int(lim-lim+1) : int(lim+lim)]
                                    lim = len(holo_cut[0,:,:])/2  
                                                                  
                                if np.shape(holo_cut)[1]>200 and np.shape(holo_cut)[2]>200:
                                    plt.figure(0) #find center
                                    plt.figsize = (30,8)
                                    plt.imshow(holo_cut[0,:,:], cmap = 'viridis')
                                    plt.axis('off')
                                    plt.savefig(integral+str(numero)+'img_'+str(j)+'_'+str(os.path.splitext(i)[0])+".pdf")                                
                                    plt.clf()
                                    plt.close()
                                
                                    
                                    plt.figure(1)
                                    fig, axes = plt.subplots(1, 3, figsize=(10, 5), sharex=True, sharey=True)
                                    ax = axes.ravel()
                                    im1 = ax[0].imshow(img, cmap=plt.cm.gray)
                                    ax[0].set_title('Original, '+'{:.4f}'.format(st.values))
                                    im2 = ax[1].imshow(centri, cmap=plt.cm.gray)
                                    ax[1].plot(coordinates[:, 1], coordinates[:, 0], 'r.', alpha = 0.5)
                                    ax[1].set_title('Template Matching')
                                    im3 = ax[2].imshow(img, cmap=plt.cm.gray)
                                    ax[2].autoscale(False)
                                    ax[2].plot(coordinates[:, 1], coordinates[:, 0], 'r.', alpha = 0.5)
                                    ax[2].set_title('Peak local max')
                                    fig.tight_layout()
                                    fig.savefig(graph_centri+'confronto_'+str(i))
                                    plt.clf()
                                    plt.close()
#                                    
#                                    
                                    holo =holo_cut[0,:,:]-np.mean(holo_cut).values
                                    R , A= matrix_R(int(lim*2), pixel_size)
                                    Integrale_circle, media = media_angolare(R, holo, pixel_size,int(lim))
                                    freq = (R[int(lim)][int(lim)+1]-R[int(lim)][int(lim)])
                                    x_fit_1 = np.arange(freq, R[int(lim)][0]+freq,freq)
#                                        
#                                       
                                    """
                                    --CEXT--tw--INTEGRATION --square
                                    
                                    """
                                    
                                    Integration_square = Integration_tw_square(holo, lim, pixel_size)                                   
                                    Cext_tw_Integration_Square = Cext_tw_integration(Integration_square, ray, integral+str(numero)+'cext_'+str(j)+'_'+str(os.path.splitext(i)[0])+".pdf", "rice")
                                    print('Cext Integrale=', Cext_tw_Integration_Square)
#                                    
                                   
                                
                                    """
                                    --DIMENSIONI--
                                    
                                    """
#                                
                               # prop_plot = plot_bello(holo_cut,illum_wavelen ,medium_index, lim,  integral+str(numero)+'_'+str(j)+"propagation.png" )
                                 
                                    z = np.linspace(100,650, 150) #np.linspace(100,600, 100)
                                    rec_vol = hp.propagate(holo_cut, z, illum_wavelen = illum_wavelen, medium_index = medium_index)
                                                    
                                    modulo = propagation_module(z, rec_vol,int(lim),int(lim), 5)# int(centro[0]),int(centro[1]),
                                    max_d_modulo, min_d_modulo, max_zarray_modulo, min_zarray_modulo = maximum_minimum(gaussian_filter(modulo, sigma =2), z)
                                    
                                    phase = np.angle(rec_vol)
                                    onda_riferimento = np.angle(np.e**((-1j*2*np.pi*z/(illum_wavelen/medium_index)))) 
                                    fase, only_p, only_phase= propagation_phase(phase, onda_riferimento, z, int(lim),int(lim), 1 ) #int(centro[0]),int(centro[1])
                                                    
                                    fase = np.nan_to_num(fase)
                                    max_d, min_d, max_zarray, min_zarray = maximum_minimum(fase, z)
#               
                                    plot_twin_propagation( z, modulo, fase, integral+str(numero)+"propagation_"+str(j)+"_"+str(os.path.splitext(i)[0])+".pdf")
                                    
                                    picchi = argrelextrema(gaussian_filter(modulo, sigma =2),np.greater)#75#47
#                                    plt.plot(gaussian_filter(modulo[3:197], sigma =2))
                                    picchi = picchi[0]
                                    for p in np.arange(0, len(picchi),1):
                                        if modulo[picchi[p]]<0.45*np.amax(modulo):
                                            picchi[p]=9999
                                    picchi = picchi[picchi != 9999]
                                   
                                    if  len(picchi) >1:
                                        fuoco = int(abs((picchi[0]-picchi[1]))/2 + picchi[0]) 
                                        
                                        if len(picchi) >2:
                                           if abs(picchi[0]-picchi[1]) < 20:
                                               fuoco = int(abs((picchi[1]-picchi[2]))/2 + picchi[1]) 
                                           else:
                                               fuoco = int(abs((picchi[0]-picchi[1]))/2 + picchi[0]) 
                                    else :
                                        fuoco =max_zarray_modulo[0] 
                                                          
                                #print(picchi, fuoco)
                                
                                    obj= np.abs(rec_vol[:,:,fuoco])
                                    lenx =np.arange(132,139.7,0.5)
                                    plt.plot(lenx,np.ones(len(lenx))*20,'-w', linewidth=5)
                                    plt.text(124,15, r'2$\mu m$', {'color': 'w', 'fontsize': 12})
                                    plt.imshow(obj[int(lim)-80:int(lim)+80,int(lim)-80:int(lim)+80],cmap='gray')
                                    plt.colorbar()
                                    plt.axis('off')
                                    plt.savefig(integral+str(numero)+"modulo_nofilter_"+str(j)+"_"+str(os.path.splitext(i)[0])+".pdf")                                        
                                    plt.clf()
                                    plt.close()
                                
                                    obj=gaussian_filter(obj, sigma=2) #per 2um 1
                                                #                                       
                                    obj= obj/np.amax(obj)
                                    mask= (obj[int(lim)-80:int(lim)+80,int(lim)-80:int(lim)+80]>0.65).astype(int) #per 2um 0.65 o 0-67
                                    lw,num = measurements.label(mask)
                                    area = (measurements.sum(mask, lw, range(num + 1)))*pixel_size*pixel_size
                                    print('area=', area)
                                    mask = mask*255
                                    mask = mask.astype(int)
                                    result = Image.fromarray((mask).astype('uint8'))
                                    result.save(integral+str(numero)+"mask_"+str(j)+"_"+str(os.path.splitext(i)[0])+".tiff")
                                    
                                    dimA1, dimB1, ratio1 = object_dimension(integral+str(numero)+"mask_"+str(j)+"_"+str(os.path.splitext(i)[0])+".tiff", pixel_size, int(lim), area, integral+str(numero)+"obj_"+str(j)+"_"+str(os.path.splitext(i)[0]))
                                    
                                
                                    """
                                    --SAVE--DATA
                                        
                                    """
                                    
                                    if (any(area > 500)) or (len(area)>10) or (Cext_tw_Integration_Square <=0) :
                                        print('delete')

                                        os.remove(integral+str(numero)+'img_'+str(j)+'_'+str(os.path.splitext(i)[0])+".pdf")
                                        #os.remove(graph_centri+'confronto_'+str(i))
                                        os.remove(integral+str(numero)+"propagation_"+str(j)+"_"+str(os.path.splitext(i)[0])+".pdf")
                                        os.remove(integral+str(numero)+"modulo_nofilter_"+str(j)+"_"+str(os.path.splitext(i)[0])+".pdf")
                                        os.remove(integral+str(numero)+"mask_"+str(j)+"_"+str(os.path.splitext(i)[0])+".tiff")
                                        
                                        try:
                                            for k in np.arange(0,len(dimA1),1):
                                                os.remove(integral+str(numero)+"obj_"+str(j)+"_"+str(os.path.splitext(i)[0])+'_'+str(k)+".pdf")
                                        except FileNotFoundError:
                                            print('no obj')
                                            
                                        try:
                                            os.remove(integral+str(numero)+'cext_'+str(j)+'_'+str(os.path.splitext(i)[0])+".pdf")
                                        except FileNotFoundError:
                                            print('no cext')   
                                            
                                    elif (Cext_tw_Integration_Square >0) and (len(area)>1) and (len(area)<10):
                                
                                        if len(area)>2:
                                            
                                            if len(area)>3 :
                                                print(i)
                                                print('scritto')  
                                                try :
                                                    print (str(i)+' '+str(fuoco)+' '+str(center_x)+' '+str(center_y)+' '+str(area[1])+' '+str(area[2])+' '+str(area[3])+' '+str(dimA1[0])+' '+str(dimB1[0])+' '+str(dimA1[1])+' '+str(dimB1[1])+' '+str(dimA1[2])+' '+str(dimB1[2])+' '+str(Cext_tw_Integration_Square),file=dati_3)
    
                                                except (TypeError,IndexError) as e:
                                                    print (str(i)+' '+str(fuoco)+' '+str(center_x)+' '+str(center_y)+' '+str(area[1])+' '+str(area[2])+' '+str(area[3])+' '+str(dimA1[0])+' '+str(dimB1[0])+'  '+str(Cext_tw_Integration_Square),file=dati_3)
                                                
                                            else:
                                                print(i)
                                                print('scritto')
                                                try :
                                                    print (str(i)+' '+str(fuoco)+' '+str(center_x)+' '+str(center_y)+' '+str(area[1])+' '+str(area[2])+' '+str(dimA1[0])+' '+str(dimB1[0])+' '+str(dimA1[1])+' '+str(dimB1[1])+' '+str(Cext_tw_Integration_Square),file=dati_2)
                                             
                                                except (TypeError,IndexError) as e:
                                                    print (str(i)+' '+str(fuoco)+' '+str(center_x)+' '+str(center_y)+' '+str(area[1])+' '+str(area[2])+' '+' '+str(dimA1[0])+' '+str(dimB1[0])+'  '+str(Cext_tw_Integration_Square),file=dati_2)
                                            
                                        else:
                                            print(i)
                                            print('scritto')  
                                            print (str(i)+' '+str(fuoco)+' '+str(center_x)+' '+str(center_y)+' '+str(area[1])+' '+str(dimA1[0])+' '+str(dimB1[0])+' '+str(Cext_tw_Integration_Square),file=dati)
                                            
     
                                    else:
                                        print("cext or area fuori range")

        numero =numero+1                                  
    start = start +20 
    end = end +20  
    cont_bg = cont_bg +1
#
dati.close() 
dati_2.close()
dati_3.close()
