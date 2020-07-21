#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 17:12:37 2020

@author: claudriel
"""
import numpy as np

def media_angolare_mask(R, holo, pixel_size, lim):
    
    total_aver = np.array([])
    Integrale_array = np.array([])

    freq = (R[int(lim)][int(lim)+1]-R[int(lim)][int(lim)])
    x = np.arange(freq*2, R[int(lim)][0]+freq*2,freq*2)
    for i in x:
       
        Integrale = np.sum(np.where(R<=i,holo.values,0))*pixel_size**2#dati nei due cerchi
        Integrale_array = np.append(Integrale_array, Integrale)
        
        restrict = np.where(R<i, holo.values,0)
        restrict2 = np.mean(np.where(R>=i-freq, restrict, 0))

        total_aver = np.append(total_aver, restrict2)

    return (Integrale_array, total_aver)

def Area(list_radii, dati, holo, pixel_size):
    """
    Calculates the angular medium of an hologram from a data file with three
    columns: data[0] is the x cordinates, data[1] is the y cordinates and 
    data[3] is the respectively hologram intensity.
    It sums the value within two ray and then it average them.
    The function cycles on ray value 
    Parameters
    ----------
    list_radii: np.array, int
        List of "ray" at which the angular medium is calculate
    dati: float by a data file
       The file.data saved for calculate the extinction coefficient 
        
    Returns
    -------
    total_aver: np.array
        Array of the angular medium of the image. 
    """
    
    total_aver = np.array([])
    restrict2=np.array([])
    restrict=np.array([])
    Integrale_array = np.array([])
    for i in list_radii:
        restrict2 = dati[(np.sqrt(dati[:,0]**2+dati[:,1]**2))<i/pixel_size] #dati nei due cerchi
        restrict = restrict2[(np.sqrt(restrict2[:,0]**2+restrict2[:,1]**2))>=(i-1)/pixel_size]
        
        aver = np.sum(restrict[:,2])/len(restrict[:,2])
        total_aver=np.append(total_aver,aver)
        
        Integrale = np.sum(restrict2[:,2]*pixel_size**2)
        Integrale_array = np.append(Integrale_array, Integrale)
        
    return (total_aver, Integrale_array)


def create_file_dat(data_holo, lim, name_file):
    """
    This writes on a file.dat the x, y index and the relative intensity
    value of an hologram image in three different columns. 
    Parameters
    ----------
    data_holo: :class:`.Image` or :class:`.VectorGrid`
        Matrix data of the hologram, that can be binned or not
    lim: int
        Value of the half shape of the new matrix. It will be the new center
    name_file: str
        Name of the file dat to write on
        
    Returns
    -------
    dati: .dat
        The file.dat with x, y, and intensity value in three different columns
    """
    leng = lim*2
    dati = open('name_file','w+')
    for i in range(0,leng):
        for j in range(0,leng):
            print ( str(str((i-lim))+" "+str((j-lim))+" "+str(data_holo[i][j].values)),file=dati)
    dati.close() 
    
    dati = np.loadtxt(open("name_file","rb")) 
    return(dati)

def create_file_dat_BINNING(data_holo, lim, name_file):
    """
    This writes on a file.dat the x, y index and the relative intensity
    value of an hologram image in three different columns. 
    Parameters
    ----------
    data_holo: :class:`.Image` or :class:`.VectorGrid`
        Matrix data of the hologram, that can be binned or not
    lim: int
        Value of the half shape of the new matrix. It will be the new center
    name_file: str
        Name of the file dat to write on
        
    Returns
    -------
    dati: .dat
        The file.dat with x, y, and intensity value in three different columns
    """
    leng = lim*2
    dati = open('name_file','w+')
    for i in range(0,leng):
        for j in range(0,leng):
            print ( str(str((i-lim))+" "+str((j-lim))+" "+str(data_holo[i][j])),file=dati)
    dati.close() 
    
    dati = np.loadtxt(open("name_file","rb")) 
    return(dati)    
    
def mediana_plus_soglia(directory_sample, directory_save_bg, directory_save_correct, dev, Lx, Ly, pixel):
    """
    Calculates the medians of a data set and subtracs them to each image.
    Parameters
    ----------
    directory_sample: str
       Path of the directory of the data set
    directory_save_bg: str
        Path of the directory where it saves the background (the medians)
    directory_save_correct: str
       Path of the directory where it saves the final coorected images
    a = int
        Initaila range of the time you want repeat the median
    b = int
        Final range of the time you want repeat the median
    N = int
        Number of the data set lengh
    
    Returns
    -------
    0 : The images are automatically saved in the path and they are ready to
    the use
    """  
    img_list= os.listdir(directory_sample)
    img_list.sort()
    
    for i in range(0,len(img_list)):
        I_array = []

        I_array.append(Image.open(directory_sample + img_list[i]).convert("L"))
    I_array = np.array([np.asarray(im) for im in I_array])      
    img_median = np.median(I_array,axis=0)
                  
    result = Image.fromarray(img_median.astype('uint8'))
    result.save(directory_save_bg+'/img_'+'.tiff')
        
    for j in range(0,len(img_list)):
        print(j)
            
        im = hp.load_image(directory_sample+img_list[j], spacing= pixel) 
                       
        I_correct = im - img_median
        minimo = np.abs(np.amin(I_correct))
        I_correct = I_correct + minimo
            
            
        #print(np.std(I_correct[0]))
#            plt.imshow(I_correct)
#            plt.show()
#            plt.close()
            
        if np.std(I_correct)> dev:
            start = time.clock()
            centro = center_find(I_correct, centers=1, threshold=0.3, blursize=6.0)
            elapsed = time.clock()
            elapsed = elapsed - start
           
            if elapsed > 6:
                print(j, ':time troppo lungo')
                os.remove(directory_sample +img_list[j])
               
            else:
                lim= 300
                
                if centro[0] < centro[1]:
                    if centro[0] - lim < 0:
                        lim = int(centro[0])
                    if centro[0] + lim > Lx:
                        lim = int(Lx - centro[0])
                    if centro[1] - lim < 0:
                        lim = int(centro[1])
                    if centro[1] + lim > Ly:
                        lim = int(Ly - centro[1])
            
                else:
                    if centro[0] - lim < 0:
                        lim = int(centro[0])
                    if centro[0] + lim > Lx:
                        lim = int(Lx - centro[0])
                    if centro[1] - lim < 0:
                        lim = int(centro[1])
                    if centro[1] + lim > Ly:
                        lim = int(Ly - centro[1])
                       
                if lim < 100:
                    print(j, ':lim troppo piccolo')
                    os.remove(directory_sample +img_list[j])
                    
                else:
                    print('ok')
                    hp.save_image(directory_save_correct +'/img_' + str(j)+ '.tiff', I_correct)
#                    I_correct=I_correct.values*255
#                    I_correct = I_correct.astype(int)
#                    result = Image.fromarray((I_correct).astype('uint8'))
#                    result.save(directory_save_correct + str(j) + '.tiff')
                    
        else:
            print(j, ':dev std troppo bassa')
            os.remove(directory_sample +img_list[j])
            
            print(np.std(I_correct[0]))
       
       
    print('Le mediane sono state calcolate e le immagini corrette')
    return (0)


def mediana_only_dev(directory_sample, directory_save_bg, directory_save_correct, dev, Lx, Ly, pixel):
    """
    Calculates the medians of a data set and subtracs them to each image.
    Parameters
    ----------
    directory_sample: str
       Path of the directory of the data set
    directory_save_bg: str
        Path of the directory where it saves the background (the medians)
    directory_save_correct: str
       Path of the directory where it saves the final coorected images
    a = int
        Initaila range of the time you want repeat the median
    b = int
        Final range of the time you want repeat the median
    N = int
        Number of the data set lengh
    
    Returns
    -------
    0 : The images are automatically saved in the path and they are ready to
    the use
    """  
    img_list= os.listdir(directory_sample)
    img_list.sort()
    I_array = []
    for i in range(0,len(img_list)):
        

        I_array.append(Image.open(directory_sample + img_list[i]).convert("L"))
    I_array = np.array([np.asarray(im) for im in I_array])      
    img_median = np.median(I_array,axis=0)
                  
    result = Image.fromarray(img_median.astype('uint8'))
    result.save(directory_save_bg+'/img_'+'.tiff')
    print('La media è stata calcolata')
        
    for j in range(0,len(img_list)):
        print(j)
            
        im = hp.load_image(directory_sample+img_list[j], spacing= pixel) 
                       
        I_correct = im - img_median
        minimo = np.abs(np.amin(I_correct))
        I_correct = I_correct + minimo
            
            
        print(np.std(I_correct[0]))
#            plt.imshow(I_correct)
#            plt.show()
#            plt.close()
            
        if np.std(I_correct) < dev:
            
            os.remove(directory_sample +img_list[j])
        else:
            hp.save_image(directory_save_correct +'/img_' + str(j)+ '.tiff', I_correct)
       
       
    print('Le mediane sono state calcolate e le immagini corrette')
    return (0)



def median(directory_sample, directory_save_bg, directory_save_correct):
    """
    Calculates the median of a data set and subtracs them to each image.
    Parameters
    ----------
    directory_sample: str
       Path of the directory of the data set
    directory_save_bg: str
        Path of the directory where it saves the background (the medians)
    directory_save_correct: str
       Path of the directory where it saves the final coorected images    
    Returns
    -------
    0 : The images are automatically saved in the path and they are ready to
    the use
    """  
    img_list= os.listdir(directory_sample)
    img_list.sort()
    I_array = []
    
    for k in range(0, len(img_list)):
        I_array.append(Image.open(directory_sample + img_list[k]).convert("L"))
    I_array = np.array([np.asarray(im) for im in I_array])      
    img_median = np.median(I_array,axis=0)
                  
    result = Image.fromarray(img_median.astype('uint8'))
    result.save(directory_save_bg+'/median.tiff')
    print('La media è stata calcolata')
    
    dev =np.array([])
    for j in range(0, len(img_list)):
        print(j)
            
        im = Image.open(directory_sample + img_list[j]).convert("L")
        I = np.asarray(im)
        
#        I_correct = I /(img_median+0.01)*255
        I_correct = I - img_median
        minimo = np.ones((1024,1280))*np.abs(np.amin(I_correct))
        I_correct = I_correct + minimo
#        plt.imshow(I_correct)
#        plt.show()
#        plt.clf()
#        plt.close()
#        
        std = (np.std(I_correct[0]))
#        print(std)
        result = Image.fromarray(I_correct.astype('uint8'))
        result.save(directory_save_correct +'/img_'+ str(j) + '.tiff')
             
        dev = np.append(std, dev)
    print('Le immagini sono state corrette') 
    return (dev)

def mean(directory_sample, directory_save_bg, directory_save_correct):
    """
    Calculates the means of a data set and subtracs them to each image.
    Parameters
    ----------
    directory_sample: str
       Path of the directory of the data set
    directory_save_bg: str
        Path of the directory where it saves the background (the medians)
    directory_save_correct: str
       Path of the directory where it saves the final coorected images    
    Returns
    -------
    0 : The images are automatically saved in the path and they are ready to
    the use
    """  
    img_list= os.listdir(directory_sample)
    img_list.sort()
    I_array = []
    
    for k in range(0, len(img_list)):
        I_array.append(Image.open(directory_sample + img_list[k]).convert("L"))
    I_array = np.array([np.asarray(im) for im in I_array])      
    img_median = np.mean(I_array,axis=0)
                  
    result = Image.fromarray(img_median.astype('uint8'))
    result.save(directory_save_bg+'.tiff')
    print('La media è stata calcolata')
    
    for j in range(0, len(img_list)):
        print(j)
            
        im = Image.open(directory_sample + img_list[j]).convert("L")
        I = np.asarray(im)
            
        I_correct = I - img_median
        minimo = np.ones((1024,1280))*np.abs(np.amin(I_correct))
        I_correct = I_correct + minimo
            
        result = Image.fromarray(I_correct.astype('uint8'))
        result.save(directory_save_correct + str(j) + '.tiff')
    print('Le immagini sono state corrette')      
    return (0)


def mean_ratio(directory_sample, directory_save_bg, directory_save_correct):
    """
    Calculates the means of a data set and subtracs them to each image.
    Parameters
    ----------
    directory_sample: str
       Path of the directory of the data set
    directory_save_bg: str
        Path of the directory where it saves the background (the medians)
    directory_save_correct: str
       Path of the directory where it saves the final coorected images    
    Returns
    -------
    0 : The images are automatically saved in the path and they are ready to
    the use
    """  
    img_list= os.listdir(directory_sample)
    img_list.sort()
    I_array = []
    
    for k in range(0, len(img_list)):
        I_array.append( Image.open(directory_sample + img_list[k]).convert("L"))
    I_array = np.array([np.asarray(im) for im in I_array])
      
    img_median = np.mean(I_array,axis=0)
    
    img_median = img_median+0.01
    #plt.imshow(img_median)
    
    
    result = Image.fromarray(img_median.astype('uint8'))
    result.save(directory_save_bg+'/mediana.tiff')
    print('La media è stata calcolata')

    for j in range(0, len(img_list)):
        print(j)
            
        im = Image.open(directory_sample + img_list[j]).convert("L")
        
        I = np.asarray(im)
            
        I_correct = (I / img_median+0.8)
        
        I_correct = I_correct/(np.amax(I_correct))*255
                    
        result = Image.fromarray(I_correct.astype('uint8'))
        result.save(directory_save_correct +'/img_'+ str(j) + '.tiff')
    print('Le immagini sono state corrette')      
    return (0)