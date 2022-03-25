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


#Parser for run all the folds. Class args for one fold.
parser = argparse.ArgumentParser("Mineral Dust")
parser.add_argument('-fd','--folder_dir', required=True, type=str, help="Folder Directory")
parser.add_argument('-sd','--stack_dir', required=True, type=str, help="Working Stack Directory")
parser.add_argument('-pf','--path_file', required=True, type = str, help='Name file path.dat')
parser.add_argument('-np','--number_path', required=True, type = str, help='Number file path.dat')
parser.add_argument('-pix', '--pixel_size', required=True, type=float, help="Pixel size")
parser.add_argument('-interval', '--interval', required=True, type=int, help="Intervallo mediana")
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
#    folder_dir = "RICE/135"
#    stack_dir = "1"
#    end = 20
#    path_file = "DAT_RICE/135"
#    numero_path = "1"
#    std_dev = 0.015
#    msk_tresh = 0.65
#    par1_deconv = 0.0509
#    par2_deconv = 0.00090


cartella = args.stack_dir +"/"
sample =  args.folder_dir + "/"
sample_name= args.folder_dir+"_"+args.stack_dir

data_path = sample+cartella+"dati/"
data_path_list= os.listdir(data_path)
data_path_list.sort()

bg_path = sample + cartella + "mediana/"
bg_path_list = os.listdir(bg_path)
bg_path_list.sort()

contatore_mediana = 0
contatore_data = 0
numero = 1 #controllo sul numero di ologrammi
raggio = args.ray

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

andamento_file_x = file1 + args.number_path+'_'+ args.stack_dir+ "_andamento_x.dat"  #file dove salvo dati
andamento_file_y = file1 + args.number_path+'_'+ args.stack_dir+ "_andamento_y.dat"  #file dove salvo dati
andamento_file_z = file1 + args.number_path+'_'+ args.stack_dir+ "_andamento_z.dat"  #file dove salvo dati
datax = open(andamento_file_x,'w+')
datay = open(andamento_file_y,'w+')
dataz = open(andamento_file_z,'w+')
#############################################################################
##############################################################################
############################################################################

"Image caracteristics"
medium_index = 1.33
pixel_size = args.pixel_size #0.2625 # pix_size is given in um
illum_wavelen = 0.6328
illum_polarization =(0,1)
k=np.pi*2/(illum_wavelen/medium_index)

Lx = args.dim_img_x
Ly = args.dim_img_y
N = 1024



if args.sample_name == 'poli':
     distant_sphere = Sphere(r=raggio, n=1.59)
     x_sec = calc_cross_sections(
         distant_sphere, medium_index, illum_wavelen, illum_polarization)

     detector = hp.detector_points(theta = np.linspace(0, np.pi, 100), phi = 0)
     distant_sphere = Sphere(r=args.ray, n=1.59)
     matr = calc_scat_matrix(detector, distant_sphere, medium_index, illum_wavelen)
        
     S = np.abs(matr[0,0,0].values)
else:
    x_sec = 0
    S = 0

for i in data_path_list:
        
        """
        --OPEN IMG--
        """
        
        raw_holo = hp.load_image(data_path + data_path_list[contatore_data], spacing = pixel_size,medium_index =medium_index ,illum_wavelen=illum_wavelen, illum_polarization=illum_polarization)
        
        if contatore_data == 0:
            bg_holo = hp.load_image(bg_path + bg_path_list[contatore_mediana], spacing = pixel_size,medium_index =medium_index ,illum_wavelen=illum_wavelen, illum_polarization=illum_polarization)
        
        else:
            if np.abs(int(data_path_list[contatore_data][4:8]) - int(data_path_list[contatore_data-1][4:8])) < args.interval:
                
                if int(int(data_path_list[contatore_data][4:8])/args.interval) == int(int(data_path_list[contatore_data-1][4:8])/args.interval):
                    contatore_mediana  = contatore_mediana -1
                    bg_holo = hp.load_image(bg_path + bg_path_list[contatore_mediana], spacing = pixel_size,medium_index =medium_index ,illum_wavelen=illum_wavelen, illum_polarization=illum_polarization)
                
                else:
                    bg_holo = hp.load_image(bg_path + bg_path_list[contatore_mediana], spacing = pixel_size,medium_index =medium_index ,illum_wavelen=illum_wavelen, illum_polarization=illum_polarization)
            else:
                bg_holo = hp.load_image(bg_path + bg_path_list[contatore_mediana], spacing = pixel_size,medium_index =medium_index ,illum_wavelen=illum_wavelen, illum_polarization=illum_polarization)
        raw_holo = raw_holo*np.mean(bg_holo).values/np.mean(raw_holo).values
        
        if np.amin(bg_holo) != 0:
            data_holo_orig = raw_holo / (bg_holo)
        else:
            data_holo_orig = raw_holo/(bg_holo+0.01)
        print(i,bg_path_list[contatore_mediana] )
        graph_centri = sample+cartella+'riconoscimento_centri/'
        integral =sample+cartella+'integral/'
        graph = sample+cartella +'graph/'
        make_the_fold(graph_centri)
        make_the_fold(integral)
       
    
        
        """
        --DECONVOLUTION FILTER--
            Faccion il template matching dell'immagine con una funzione a priori
        """
            
        center_holo = data_holo_orig[0,:,:]    #The image must be square for the FFT        
        center_holo = center_holo -1
        
        t = 0.3
        centri = filter_deconv(Lx, Ly, center_holo, pixel_size, args.par1_deconv, args.par2_deconv, S, args.sample_name2, illum_wavelen, medium_index ) 
      
        
        centri = centri/np.amax(centri)
        image_max = ndi.maximum_filter(centri, size=30, mode='constant')
        coordinates = peak_local_max(centri, min_distance=50, threshold_abs=t)

        
        """
        --DELETE NULL IMG--
        Elimino le immagini con una DEV STD inferiore di un tot, che cambierà 
        per ogni data-set. 
        Elimino le immagini senza centri
        """
        st = np.std(data_holo_orig)
        if (np.shape(coordinates)[0]<4) and (np.shape(coordinates)[0]>0) and (st >= args.std_dev):  #se sono troppe coordinate: o troppi ologrammi o immagine rumorosa e deve almeno trovarne una
            img = center_holo
                
            for j in np.arange(0,len(coordinates),1): #ciclo sulle coordinate
                
                    """
                    --CUT THE IMAGE--
                    attorno ad ogni ologramma in ogni immagine. Salvo solo quelle
                    con std dev > dello stesso tot di prima
                    """
                    center_x_0 = coordinates[j][0]
                    center_y_0 = coordinates[j][1]

                    lim =args.lim_img
                    lim1=args.lim_img
                    lim2=args.lim_img
            
                    lim = find_the_new_lim(center_x_0, center_y_0, lim, lim1, lim2, Lx-1, Ly-1)
                    
                    if lim >= 150: #img troppo piccole scartate
                        data_holo = data_holo_orig -1
                        holo_cut = data_holo[:,int(center_x_0 -lim) : int(center_x_0 +lim+1), int(center_y_0 -lim)  : int(center_y_0 +lim+1)  ] #taglio img
                        
                    #     centri2 = center_find(holo_cut_0, centers=1, threshold=0.5, blursize=2.0)                               
                        
                    #     center_x = centri2[0]
                    #     center_y = centri2[1]
                        
                    #     Lx2 = lim*2
                    #     Ly2 = Lx2 +1
                        
                    #     lim =lim
                    #     lim1=lim
                    #     lim2=lim
                
                    #     lim = find_the_new_lim(center_x, center_y, lim, lim1, lim2, Lx2, Ly2)
                    #     holo_cut = holo_cut_0[:,int(center_x-lim) : int(center_x+lim+1), int(center_y-lim)  : int(center_y+lim+1)  ] #taglio img
                       
                        
                        if lim >= 150:
                            plt.figure(0) #find center
                            plt.figsize = (30,8)
                            plt.imshow(holo_cut[0,:,:], cmap = 'viridis')
                            plt.axis('off')
                            plt.savefig(integral+str(numero)+'img_'+str(j)+'_'+str(os.path.splitext(i)[0])+".pdf")                                
                            plt.clf()
                            plt.close()
     
                            plt.figure(1)
                            fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharex=True, sharey=True)
                            ax = axes.ravel()
                            im1 = ax[0].imshow(img, cmap=plt.cm.gray)
                            ax[0].set_title('Original, '+'{:.4f}'.format(st.values))
                            # im2 = ax[1].imshow(centri, cmap=plt.cm.gray)
                            # ax[1].plot(coordinates[:, 1], coordinates[:, 0], 'r.', alpha = 0.5)
                            # ax[1].set_title('Template Matching')
                            im3 = ax[1].imshow(img, cmap=plt.cm.gray)
                            ax[1].autoscale(False)
                            ax[1].plot(coordinates[:, 1], coordinates[:, 0], 'r.', alpha = 0.5)
                            ax[1].set_title('Peak local max')
                            fig.tight_layout()
    
                            fig.savefig(graph_centri+'confronto_'+str(i))
                            plt.clf()
                            plt.close()
    
                            holo = holo_cut[0,:,:]#-np.mean(holo_cut).values
                            R , A = matrix_R(int(lim*2)+1, pixel_size)
                            Integrale_circle, media = media_angolare(R, holo, pixel_size,int(lim)) #media angolare
                            freq = (R[int(lim)][int(lim)+1]-R[int(lim)][int(lim)])
                            x_fit_1 = np.arange(freq, R[int(lim)][0]+freq,freq)
                            
                            """
                            --CEXT--tw--INTEGRATION --square
                            
                            """
                            Integration_square = Integration_tw_square(holo, lim, pixel_size)                                   
                            Cext_tw_Integration_Square,andamento_x, andamento_y= Cext_tw_integration(medium_index, illum_wavelen, illum_polarization, Integration_square, raggio, integral+str(numero)+'cext_'+str(j)+'_'+str(os.path.splitext(i)[0])+".pdf", args.sample_name, x_sec)
                            print('Cext Integrale=', Cext_tw_Integration_Square); print('Cabs=',andamento_y[len(andamento_y)-1])
                            
                            
                            """
                            5) --FORMA--
                            
                            """   
                            angle = 10
                            holo_forma= holo_cut[0,:,:].values
                            x_forma = np.arange(0,360,angle)
                           
                            a = int(lim/3)
                            b = int(lim/3)*2
                            c = int(lim)
                            mxt1 = np.where(R<=freq*a, holo_forma,0)    
                            mxt2 = np.where(R <= freq*b, holo_forma,0)
                            mxt2 = np.where(R > freq*a, mxt2,0)
                            
                            # if R[0,int(lim)]<freq*300:
                            mxt3 = np.where(R<=R[0,c],holo_forma,0)
                            mxt4 = np.where(R <=R[0,c],holo_forma,0)
                            # else:
                                # mxt3 = np.where(R<=freq*300,holo_forma,0)
                                # mxt4 = np.where(R <=freq*300,holo_forma,0)#R[0,int(lim)]
                            mxt3 = np.where(R>freq*b,mxt3,0)
                            value1 = np.array([])
                            value2 = np.array([])
                            value3 = np.array([])
                            value4 = np.array([])
                            for k in x_forma:
                                mtx1 = np.where((A<=k+angle)&(A>=k),mxt1,0)
                                value1 = np.append(value1, np.mean(np.abs(mtx1)))
                                mtx2 = np.where((A<=k+angle)&(A>=k),mxt2,0)
                                value2 = np.append(value2, np.mean(np.abs(mtx2)))
                                mtx4 = np.where((A<=k+angle)&(A>=k),mxt4,0)
                                value4 = np.append(value4, np.mean(np.abs(mtx4)))   
                                mtx3 = np.where((A<=k+angle)&(A>=k),mxt3,0)
                                value3 = np.append(value3, np.mean(np.abs(mtx3)))
    
                            value1 = gaussian_filter(value1, sigma=1)/np.amax(gaussian_filter(value1, sigma=1))
                            value2 =gaussian_filter(value2, sigma=1)/np.amax(gaussian_filter(value2, sigma=1))
                            value3 = gaussian_filter(value3, sigma=1)/np.amax(gaussian_filter(value3, sigma=1))
                            value4 = gaussian_filter(value4, sigma=1)/np.amax(gaussian_filter(value4, sigma=1))


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
                            
                            plt.plot(x_forma,value3,'-*', label = 'Third Rings')
                            plt.ylim(0,1.1)
                            plt.title('Simmetry Object')
                            plt.xlabel('Angle (°C)')
                            plt.ylabel('Intensity Norm.')
                            plt.legend()
                            plt.savefig(integral+str(numero)+'_'+str(j)+'_forma3.png')
                            plt.clf()
                            plt.close()
                               
                            plt.plot(x_forma,value4,'-*')
                            plt.ylim(0,1.1)
                            plt.title('Simmetry Object')
                            plt.xlabel('Angle (°C)')
                            plt.ylabel('Intensity Norm.')
                            plt.savefig(integral+str(numero)+'_'+str(j)+'_forma.png')
                            plt.clf()
                            plt.close()
                            
                            
                            
                            """
                            --DIMENSIONI--
                            
                            """
                            z = np.linspace(args.prop_z,args.prop_z+500, 100)
                            rec_vol = hp.propagate(holo_cut, z, illum_wavelen = illum_wavelen, medium_index = medium_index)
                                            
                            modulo = propagation_module(z, rec_vol,int(lim),int(lim), 5)**2# int(centro[0]),int(centro[1]),
                            max_d_modulo, min_d_modulo, max_zarray_modulo, min_zarray_modulo = maximum_minimum(gaussian_filter(modulo, sigma =2), z)
                            
                            phase = np.angle(rec_vol)
                            onda_riferimento = np.angle(np.e**((-1j*2*np.pi*z/(illum_wavelen/medium_index)))) 
                            fase, only_p, only_phase= propagation_phase(phase, onda_riferimento, z, int(lim),int(lim), 1 ) #int(centro[0]),int(centro[1])
                                            
                            fase = np.nan_to_num(fase)
                            max_d, min_d, max_zarray, min_zarray = maximum_minimum(fase, z)
    
                            plot_twin_propagation( z, modulo, fase, integral+str(numero)+"propagation_"+str(j)+"_"+str(os.path.splitext(i)[0])+".pdf")       
                            picchi = argrelextrema(gaussian_filter(modulo, sigma =2),np.greater)#75#47
                           
                            
                            picchi = picchi[0]
                            rumore = np.amax(modulo)
                            
                            for p in np.arange(0, len(picchi),1):
                                if modulo[picchi[p]]<0.65*np.amax(modulo):
                                    picchi[p]=9999
                            picchi = picchi[picchi != 9999]
                            
                            if picchi.size ==4:
                                fuoco = int(abs((picchi[1]-picchi[2]))/2 + picchi[1]) 
                            elif picchi.size>1:
                                fuoco = int(abs((picchi[0]-picchi[1]))/2 + picchi[0]) 
                                
                                if picchi.size>2:
                                    
                                    picchi = argrelextrema(gaussian_filter(modulo, sigma =4),np.greater)
                                    picchi = picchi[0]
                                    
                                    for p in np.arange(0, len(picchi),1):
                                        if modulo[picchi[p]]<0.85*np.amax(modulo):
                                            picchi[p]=9999
                                    picchi = picchi[picchi != 9999]
                                    
                                    if picchi.size >1:
                                        fuoco = int(abs((picchi[0]-picchi[1]))/2 + picchi[0]) 
                                    else:
                                        if picchi.size>0:
                                            fuoco = picchi[0] 
                                        else:
                                            fuoco = 0
                            else :
                                if picchi.size>0:
                                    fuoco = picchi[0] 
                                else:
                                    fuoco = 0
                                
                            if modulo[99]>0.6*np.amax(modulo) and picchi.size>0 :
                                fuoco = int(abs((picchi[0]-99))/2 + picchi[0]) 
                            
                            
                            obj= np.abs(rec_vol[:,:,fuoco])**2
                            lenx =np.arange(132,139.7,0.5)
                            plt.plot(lenx,np.ones(len(lenx))*20,'-w', linewidth=5)
                            plt.text(124,15, r'2$\mu m$', {'color': 'w', 'fontsize': 12})
                            plt.imshow(obj[int(lim)-80:int(lim)+80,int(lim)-80:int(lim)+80],cmap='gray')
                            plt.colorbar()
                            plt.axis('off')
                            plt.savefig(integral+str(numero)+"modulo_nofilter_"+str(j)+"_"+str(os.path.splitext(i)[0])+".pdf")                                        
                            plt.clf()
                            plt.close()
                        
                            obj=gaussian_filter(obj, sigma=0) #per 2um 1
                                        #                                       
                            obj= obj/np.amax(obj)
                            mask= (obj[int(lim)-80:int(lim)+80,int(lim)-80:int(lim)+80]>args.mask_tresh).astype(int) #per 2um 0.65 o 0-67
                            lw,num = measurements.label(mask)
                            area = (measurements.sum(mask, lw, range(num + 1)))*pixel_size*pixel_size
                            print('area=', area)
                            mask = mask*255
                            mask = mask.astype(int)
                            result = Image.fromarray((mask).astype('uint8'))
                            result.save(integral+str(numero)+"mask_"+str(j)+"_"+str(os.path.splitext(i)[0])+".tiff")
                            
                            dimA1, dimB1, ratio1, dx, dy, = object_dimension(integral+str(numero)+"mask_"+str(j)+"_"+str(os.path.splitext(i)[0])+".tiff", pixel_size, int(lim), area, integral+str(numero)+"obj_"+str(j)+"_"+str(os.path.splitext(i)[0]))  
                        
                            """
                            --SAVE--DATA
                                
                            """
                            
                            if (any(area > args.area_max)) or (len(area)>10) or (Cext_tw_Integration_Square <=args.cext_min) :
                                print('delete')
        
                                os.remove(integral+str(numero)+'img_'+str(j)+'_'+str(os.path.splitext(i)[0])+".pdf")
                                os.remove(integral+str(numero)+"propagation_"+str(j)+"_"+str(os.path.splitext(i)[0])+".pdf")
                                os.remove(integral+str(numero)+"modulo_nofilter_"+str(j)+"_"+str(os.path.splitext(i)[0])+".pdf")
                                os.remove(integral+str(numero)+"mask_"+str(j)+"_"+str(os.path.splitext(i)[0])+".tiff")
                                os.remove(integral+str(numero)+'_'+str(j)+'_forma.png')
                                os.remove(integral+str(numero)+'_'+str(j)+'_forma3.png')
                                
                                try:
                                    for k in np.arange(0,len(dimA1),1):
                                        os.remove(integral+str(numero)+"obj_"+str(j)+"_"+str(os.path.splitext(i)[0])+'_'+str(k)+".pdf")
                                except FileNotFoundError:
                                    print('no obj')
                                    
                                try:
                                    os.remove(integral+str(numero)+'cext_'+str(j)+'_'+str(os.path.splitext(i)[0])+".pdf")
                                except FileNotFoundError:
                                    print('no cext')   
                                    
                            elif (Cext_tw_Integration_Square >args.cext_min) and (len(area)>1) and (len(area)<10):
                               
                                # np.savetxt(datay,[andamento_forma2],fmt='%1.10f')
                                
                                
                                if len(area)>2:
                                    
                                    if len(area)>3 :
                                        print(i)
                                        print('scritto')  
                                        
                                        try :
        
                                            print (str(i)+' '+str(fuoco)+' '+str(center_x_0)+' '+str(center_y_0)+' '+str(area[1])+' '+str(area[2])+' '+str(area[3])+' '+str(dimA1[0])+' '+str(dimB1[0])+' '+str(dimA1[1])+' '+str(dimB1[1])+' '+str(dimA1[2])+' '+str(dimB1[2])+' '+str(Cext_tw_Integration_Square)+' '+str(np.std(value1))+' '+str(np.std(value2))+' '+str(np.std(value3))+' '+str(np.std(value4)),file=dati_3)
                                            
                                        except (TypeError,IndexError) as e:
                                            print (str(i)+' '+str(fuoco)+' '+str(center_x_0)+' '+str(center_y_0)+' '+str(area[1])+' '+str(area[2])+' '+str(area[3])+' '+str(dimA1[0])+' '+str(dimB1[0])+'  '+str(Cext_tw_Integration_Square)+' '+str(np.std(value1))+' '+str(np.std(value2))+' '+str(np.std(value3))+' '+str(np.std(value4)),file=dati_3)
                                            
                                    else:
                                        print(i)
                                        print('scritto')
                                        try :
                                            print (str(i)+' '+str(fuoco)+' '+str(center_x_0)+' '+str(center_y_0)+' '+str(area[1])+' '+str(area[2])+' '+str(dimA1[0])+' '+str(dimB1[0])+' '+str(dimA1[1])+' '+str(dimB1[1])+' '+str(Cext_tw_Integration_Square)+' '+str(np.std(value1))+' '+str(np.std(value2))+' '+str(np.std(value3))+' '+str(np.std(value4)),file=dati_2)
                                     
                                        except (TypeError,IndexError) as e:
                                            print (str(i)+' '+str(fuoco)+' '+str(center_x_0)+' '+str(center_y_0)+' '+str(area[1])+' '+str(area[2])+' '+' '+str(dimA1[0])+' '+str(dimB1[0])+'  '+str(Cext_tw_Integration_Square)+' '+str(np.std(value1))+' '+str(np.std(value2))+' '+str(np.std(value3))+' '+str(np.std(value4))+' '+str(andamento_y[len(andamento_y)-1]),file=dati_2)
                                    
                                else:
                                    print(i)
                                    print('scritto')  
                                    print (str(i)+' '+str(z[fuoco])+' '+str(center_x_0)+' '+str(center_y_0)+' '+str(lim)+' '+str(area[1])+' '+str(dimA1[0])+' '+str(dimB1[0])+' '+str(Cext_tw_Integration_Square)+' '+str(dx)+' '+str(dy)+' '+str(np.std(value1))+' '+str(np.std(value2))+' '+str(np.std(value3))+' '+str(np.std(value4)),file=dati)
                                    
                                    np.savetxt(dataz,[value4],fmt='%1.10f')
                                    np.savetxt(datax,[andamento_x],fmt='%1.10f')
                                    np.savetxt(datay,[andamento_y],fmt='%1.10f')
                            else:
                                print("cext or area fuori range")
                                os.remove(integral+str(numero)+'img_'+str(j)+'_'+str(os.path.splitext(i)[0])+".pdf")
                                os.remove(integral+str(numero)+"propagation_"+str(j)+"_"+str(os.path.splitext(i)[0])+".pdf")
                                os.remove(integral+str(numero)+"modulo_nofilter_"+str(j)+"_"+str(os.path.splitext(i)[0])+".pdf")
                                os.remove(integral+str(numero)+"mask_"+str(j)+"_"+str(os.path.splitext(i)[0])+".tiff")
                                os.remove(integral+str(numero)+'_'+str(j)+'_forma.png')
                                os.remove(integral+str(numero)+'_'+str(j)+'_forma3.png')
                       
        numero =numero+1                                  
        
        contatore_mediana = contatore_mediana +1
        contatore_data = contatore_data +1

dati.close() 
dati_2.close()
dati_3.close()
datax.close()
datay.close()
dataz.close()