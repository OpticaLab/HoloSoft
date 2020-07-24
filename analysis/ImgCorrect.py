#!/usr/bin/python3

"""
Created on Sun May 19 16:21:39 2019

@author: claudriel

Script for correct the raw-images from the background. 
The program takes all the N images in the directory path and it calculates 
the median every "end".
At each set of images, it subtracts the respective median. 
Yuo can save them in another fold.
Yuo can also set a cut off for delete empty image.

"""
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import argparse

"""
Parser for run all the folds. Class args for one fold.

"""
parser = argparse.ArgumentParser("Programma Ologrammi")
parser.add_argument('-fd','--folder_dir', required=True, type=str, help="Folder Directory")
parser.add_argument('-sd','--stack_dir', required=True, type=str, help="Working Stack Directory")
parser.add_argument('-sv', '--save_img', action='store_true', help="Save correct images" )
parser.add_argument('-del', '--delete_img', action='store_true', help="Delete medians" )
parser.add_argument('-N', '--num', required=True, type=int, help="Number of total images")
parser.add_argument('-end', '--end', required=True, type=int, help="Median cycle")
parser.add_argument('-st', '--std_dev', required=True, type=int, help="Standard deviation cut off")

args = parser.parse_args()


#class args:
#    folder_dir = "RICE/1"
#    stack_dir = "1"
#    save_img = True
#    delete_img = True     
#    num = 501
#    end = 20
#    std_dev = 0.0138


sample = args.folder_dir +"/" +args.stack_dir+"/dati3/"
path_list= os.listdir(sample)
   
img_list= os.listdir(sample)
img_list.sort()
        
directory_save_bg = args.folder_dir +"/" +args.stack_dir+"/mediana"   
directory_save_correct = args.folder_dir + "/" +args.stack_dir+"/img_correct"

try: 
	os.stat(directory_save_bg) 
except: 
	os.makedirs(directory_save_bg)

if args.save_img:   
    try: 
        os.stat(directory_save_correct)  
    except: 
        os.makedirs(directory_save_correct) 
         
        
start = 0
end = args.end
N = args.num
std_array = np.array([])

for i in np.arange(0, len(N)/end, 1):
    I_array = []
    
    for k in range(start, end):
        I_array.append(Image.open(sample + img_list[k]).convert("L"))
    I_array = np.array([np.asarray(im) for im in I_array])      
    img_median = np.median(I_array,axis=0)
    result = Image.fromarray(img_median.astype('uint8'))
    result.save(directory_save_bg+'/median_'+format(start,"04")+'_'+format(end,"04")+'.tiff')
    
    std_array = np.array([])
    for j in range(start, end):
        im = Image.open(sample + img_list[j]).convert("L")
        I = np.asarray(im)
        I_correct = (I)/(img_median) 
        st = np.std(I_correct)
        std_array = np.append(std_array,st)
        
        if args.save_img == "save":
            I_correct = I_correct/np.amax(I_correct)*255
            result = Image.fromarray(I_correct.astype('uint8'))
            result.save(directory_save_correct +'/img_'+ str(j)+'.tiff')
            print('Le immagini sono state corrette')
    
    if args.delete_img:
        if all(k < args.std_dev for k in std_array) :
            for m in range(start, end):
                os.remove(sample + img_list[m])
            os.remove(directory_save_bg+'/median_'+format(start,"04")+'_'+format(end,"04")+'.tiff')
            print("la media è stata cancellata")
    
    start = start +20
    end = end +20
