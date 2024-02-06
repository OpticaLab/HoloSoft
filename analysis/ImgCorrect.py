#!/usr/bin/python3

"""
Script for correct the raw-images from the background. 
The program takes all the N images in the directory path and it calculates 
the median every "end".
At each set of images, it subtracts the respective median. 
Yuo can save them in another fold.
Yuo can also set a cut off for delete empty image.

"""
import os
import numpy as np
from PIL import Image
import argparse

"""
Parser for run all the folds. Class args for one fold.

"""
parser = argparse.ArgumentParser("Correct Image")
parser.add_argument('-fd','--folder_dir', required=True, type=str, help="Folder Directory")
parser.add_argument('-sd','--stack_dir', required=True, type=str, help="Working Stack Directory")
parser.add_argument('-sv', '--save_img', action='store_true', help="Save correct images" )
parser.add_argument('-del', '--delete_img', action='store_true', help="Delete medians" )
parser.add_argument('-st', '--std_dev', required=True, type=float, help="Standard deviation cut off")
parser.add_argument('-stp', '--step', required=True, type=int, help="Step of average")

args = parser.parse_args()

S = True
N = False

# class args:
#     folder_dir = ""
#     stack_dir = ""
#     save_img = 
#     delete_img = 
#     std_dev =
#     step = 


for m in args.stack_dir[:]:
    print(m)
    sample = args.folder_dir + "/" + str(args.stack_dir[m])+"/data/"
    path_list = os.listdir(sample)

    img_list = os.listdir(sample)
    img_list.sort()

    directory_save_bg = args.folder_dir + "/" + str(args.stack_dir[m])+"/bg"
    directory_save_correct = args.folder_dir + "/" + str(args.stack_dir[m])+"/img_correct"

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
    end = args.step
    std_array = np.array([])

    for i in np.arange(0, len(path_list)/args.step, 1):
        print(start)
        I_array = []
        for k in range(start, end):
            
            I_array.append(Image.open(sample + img_list[k]).convert("L"))
        I_array = np.array([np.asarray(im) for im in I_array])
        img_median = np.median(I_array, axis=0)
        result = Image.fromarray(img_median.astype('uint8'))
        result.save(directory_save_bg+'/median_' + format(start, "04")+'_'+format(end, "04")+'.tiff')
        std_array = np.array([])
        
        for j in range(start, end):
            im = Image.open(sample + img_list[j]).convert("L")
            I = np.asarray(im)
            I_correct = (I)/(img_median)
            st = np.std(I_correct)
            std_array = np.append(std_array, st)
            
            if args.delete_img:
                if st < args.std_dev:
                    print("canceled",img_list[j])
                    os.remove(sample + img_list[j])

           
            if args.save_img:
                plt.imshow(I_correct, cmap = "viridis")
                plt.savefig(directory_save_correct + '/img_' + str(j)+'_'+str(st)+'.png')
                plt.clf()
                plt.close()
                # I_correct = I_correct/np.amax(I_correct)*255
                # result = Image.fromarray(I_correct.astype('uint8'))
                # result.save(directory_save_correct + '/img_' + str(j)+'_'+str(st)+'.tiff')

        if args.delete_img:
            if all(k < args.std_dev for k in std_array):
                print("canceled", '/median_' + format(start, "04")+'_'+format(end, "04"))
                os.remove(directory_save_bg+'/median_' + format(start, "04")+'_'+format(end, "04")+'.tiff')
                
        start = start + args.step
        end = end + args.step
        
