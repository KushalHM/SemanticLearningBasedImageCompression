import skimage.io
import skimage.transform
import numpy as np
from PIL import Image

def chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))

def load_image(path, np_data=None, csv_data=None):
	# Load image from save npy file
    idx = csv_data[csv_data.image_path == path].index[0]
    return np_data[idx]


def load_image_disk(path):
	# Load image from disk
    try:
        img = skimage.io.imread( path ).astype(np.float)
        img /= 255.0
        X = img.shape[0]
        Y = img.shape[1]
        S = min(X,Y)
        XX = int((X - S) / 2)
        YY = int((Y - S) / 2)
    except:
        return Exception("You need skimage to load the image")

    # if black and white image, repeat the channels
    if len(img.shape) == 2: img=np.tile(img[:,:,None], 3)
    return skimage.transform.resize( img[XX:XX+S, YY:YY+S], [224,224] )

def load_single_image(image, np_data=None, csv_data=None):
    return np.expand_dims(load_image(image, np_data, csv_data),0)


def array2PIL(arr):
    mode = 'RGBA'
    shape = arr.shape
    arr = arr.reshape(arr.shape[0]*arr.shape[1], arr.shape[2])
    if len(arr[0]) == 3:
        arr = np.c_[arr, 255*np.ones((len(arr),1), np.uint8)]

    return Image.frombuffer(mode, (shape[1], shape[0]), arr.tostring(), 'raw', mode, 0, 1)


def normalize(x):
	# Normalize given array
    min = np.min(x)
    max = np.max(x)
    return (x-min)/(max-min)
