import numpy
from skimage import io
import tensorflow as tf
from skimage.measure import compare_ssim
import cv2
import numpy as np
from scipy import signal
from scipy.ndimage.filters import convolve
import tensorflow as tf


tf.flags.DEFINE_string('original_image', None, 'Path to PNG image.')
tf.flags.DEFINE_string('compared_image', None, 'Path to PNG image.')
FLAGS = tf.flags.FLAGS

def get_metrics(im1, im2):
    """
    Calculates and returns evaluation metrics
    
    Parameters
    ----------
    im1 : str
        First image path
    im2 : str
        Second image path
    
    Returns
    -------
    float, float
        PSNR, SSIM
    """
    return psnr(im1,im2), ssim(im1,im2)

def log10(x):
    numerator = tf.log(x)
    denominator = tf.log(tf.constant(10, dtype=numerator.dtype))
    return numerator / denominator


def psnr(im1, im2):
    try:
      # Read the image
      imageA = cv2.imread(im1)
      imageB = cv2.imread(im2)

      # Convert to numpy for faster processing
      img_arr1 = numpy.array(imageA).astype('float32')
      img_arr2 = numpy.array(imageB).astype('float32')

      # Calculate MSE and PSNR
      mse = tf.reduce_mean(tf.squared_difference(img_arr1, img_arr2))
      psnr = tf.constant(255**2, dtype=tf.float32)/mse
      result = tf.constant(10, dtype=tf.float32)*log10(psnr)
      with tf.Session():
          result = result.eval()
      return result
    except Exception as e:
      # Exception Handling
      print ("Exception in PSNR for "+im1)
      print (e)
    return 0

def ssim(im1, im2):
    score = 0
    try:
      # load the two input images
      imageA = cv2.imread(im1)
      imageB = cv2.imread(im2)

      # convert the images to grayscale
      grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
      grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)

      # compute the Structural Similarity Index (SSIM) between the two
      # images, ensuring that the difference image is returned
      (score, diff) = compare_ssim(grayA, grayB, full=True)
    except Exception as e:
      # Exception Handling
      print ("Exception in SSIM for "+im1)
      print (e)
    return score
