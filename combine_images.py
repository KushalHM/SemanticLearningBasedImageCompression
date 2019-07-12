# from __future__ import division
import sys
from PIL import Image
import os
import numpy as np
from util import *
import argparse
from scipy.stats import percentileofscore
from get_metrics import get_metrics


def make_quality_compression(image,roi,original,sal):
    """
    Use the MS-ROI to compress the input image at different levels
    
    Parameters
    ----------
    image : str
        Original image path
    roi : str
        MS-ROI path
    original : arr
        input image as array
    sal : arr
        ms-roi as array
    
    Returns
    -------
    str, int, int
        compressed image path, compressed image size, uncompressed omage size
    """
    output_directory        = 'static'
    threshold_pct           = 20
    jpeg_compression        = 40
    model                   = 4
    single                  = 1

    original.save(output_directory + '\\temp.png', quality=100)
    # if the size of the map is not the same original image, then blow it
    if original.size != sal.size:
        sal = sal.resize(original.size)

    sal_arr = np.asarray(sal)
    img_qualities = []
    quality_steps = [i*10 for i in range(1,11)]

    # this temp directory will be deleted, do not use this to store your files
    os.makedirs('temp_xxx_yyy')
    for q in quality_steps:
        name = 'temp_xxx_yyy\\temp_' + str(q) + '.jpg'
        original.save(name, quality=q)
        img_qualities.append(np.asarray(Image.open(name)))
        os.remove(name)
    os.rmdir('temp_xxx_yyy')
                   
    k = img_qualities[-1][:] # make sure it is a copy and not reference
    shape = k.shape 
    k.flags.writeable = True
    mx, mn = np.max(sal_arr), np.mean(sal_arr)
    sal_flatten = sal_arr.flatten()

    q_2,q_3,q_5,q_6,q_9 = map(lambda x: np.percentile(sal_arr, x), [20,30,50,60,90])

    q_a = [np.percentile(sal_arr, j) for j in quality_steps]
    low, med, high = 1, 5, 9

    for i in range(shape[0]):
        for j in range(shape[1]):
            for l in range(shape[2]):
                ss = sal_arr[i,j]

                if model == 1:
                    # model -1 
                    # hard-coded model
                    if ss > mn: qq = 9
                    else: qq = 6

                elif model == 2:
                    # model -2 
                    # linearly scaled technique
                    qq = (ss * 10 // mx) -1  + 3
                
                elif model == 3:
                    # model -3 
                    # percentile based technique
                    # qq = int(percentileofscore(sal_flatten, ss)/10)
                    for index, q_i in enumerate(q_a):
                        if ss < q_i: 
                            qq = index + 1
                            break

                elif model == 4:
                    # model -4 
                    # discrete percentile based technique
                    # if   ss < q_2: qq = 4 
                    if ss < q_2: qq = 4 
                    elif ss < q_6: qq = 6 
                    elif ss < q_9: qq = 8 
                    else: qq = 9

                elif model == 5:
                    # model -5
                    # two way percentile
                    if ss <  q_5: qq = 2
                    else: qq = 8

                elif model == 6:
                    # model -6
                    # two way percentile - higher coverage
                    if ss <  q_5: qq = 7
                    else: qq = 9
                    
                else:
                    raise Exception("unknown model number")

                if qq < low : qq = low
                if qq > high: qq = high 
                k[i,j,l] = img_qualities[qq][i,j,l]
                

    # save the original file at the given quality level
    compressed = output_directory + '\\' + '_original_' + image.split('\\')[-1] + '_' + str(jpeg_compression) + '.jpg'
    # print (compressed)
    original.save(compressed, quality=jpeg_compression)
    
    original_size = os.path.getsize(compressed)
    uncompressed_size = os.path.getsize(output_directory + '\\temp.png')
    os.remove(output_directory + '\\temp.png')

    return compressed, original_size, uncompressed_size


def combineImage(imgPath, roiPath):
    """
    Public method used to compress image based on MS-ROI
    
    Parameters
    ----------
    imgPath : str
        Original image path
    roiPath : str
        MS-ROI path
    
    Returns
    -------
    str, int, int, float, float
        compressed image path, compressed image size, uncompressed_size, PSNR, SSIM
    
    """
    original = Image.open(imgPath)
    sal = Image.open(roiPath)
    compressed, newSize, uncSize = make_quality_compression(imgPath,roiPath,original,sal)
    psnr, ssim = get_metrics(imgPath,compressed)

    return compressed, newSize, uncSize, psnr, ssim