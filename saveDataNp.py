# -*- coding: utf-8 -*-
"""
Created on Tue Apr 09 23:24:37 2018

@author: Harsh Sharma

"""

from __future__ import print_function

import os
import glob
import numpy as np
import pandas as pd

import skimage.io
import skimage.transform
from PIL import Image
from params import TrainingParams, HyperParams, CNNParams  

tparam = TrainingParams(verbose=True)  
hyper  = HyperParams(verbose=True)
    

def create_data(isTrain): 
    """
    Reads all images from the dataset and stores them as .npy files for faster processing
    
    Parameters
    ----------
    isTrain : bool
        flag to store train and test files
    """

    # read the config params
    tparam = TrainingParams(verbose=True)  
    hyper  = HyperParams(verbose=True)
    cparam = CNNParams(verbose=True)
    trainDataFileName = tparam.train_data_np
    testDataFileName = tparam.test_data_np

    # Get details of images in train and test datasets
    csvData = pd.read_pickle(tparam.data_train_path) if isTrain else pd.read_pickle(tparam.data_test_path)
    
    total = len(csvData)
    imgs = np.ndarray((total,hyper.image_h, hyper.image_w, hyper.image_c), dtype=np.float32)

    print ("\n\n\n\nStarting Process")
    fid = open(tparam.train_csv, 'w') if isTrain else open(tparam.test_csv, 'w')
    fid.write('index,image_path\n')
    # 1) Read each image
    # 2) Resize
    # 3) Add to numpy array
    for index, file in enumerate(csvData.image_path):
        fid.write(str(index)+','+file+'\n')
        img = skimage.io.imread(file).astype(np.float)
        img /= 255.0
        X = img.shape[0]
        Y = img.shape[1]
        S = min(X,Y)
        XX = int((X - S) / 2)
        YY = int((Y - S) / 2)
        if len(img.shape) == 2: img=np.tile(img[:,:,None], 3)
        imgs[index] = np.array([skimage.transform.resize( img[XX:XX+S, YY:YY+S], [224,224])])
        if index % 500 == 0:
            print('Done: {0} images'.format(index))
    
    print('Starting saving process.')
    fid.close()
    # Save files
    if (isTrain):
        np.save(trainDataFileName, imgs)
    else:
        np.save(testDataFileName, imgs)
    
    print('Saving to .npy files done.')


def load_train_data():
    print ("Loading training data")
    return np.load(tparam.train_data_np)

def load_test_data():
    print ("Loading test data")
    return np.load(tparam.test_data_np)

# For saving first time
if __name__ == '__main__':
   create_data(True)
   create_data(False)