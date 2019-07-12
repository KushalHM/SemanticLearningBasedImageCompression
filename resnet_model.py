# Multi-structure Regions of Interest
# 
# References : 
#       CNN structure based on ResNet-50, https://github.com/piyush2896/ResNet50-Tensorflow/blob/master/model.py
#       Channel independent feature maps (3D features) using https://www.tensorflow.org/versions/r0.11/api_docs/python/nn.html#depthwise_conv2d_native 
#       GAP based on https://github.com/jazzsaxmafia/Weakly_detector/blob/master/src/detector.py

import tensorflow as tf
import numpy as np
from params import CNNParams, HyperParams

# Read the params and set others
cnn_param = CNNParams(verbose=False)
stddev  = 0.2
num_channels = 3
hyper = HyperParams(verbose=False)
n_labels = 257

# Method for debugging
def print_model_params(verbose=True):
    total_parameters = 0
    for variable in tf.trainable_variables():
        shape = variable.get_shape()
        if verbose: print("name: " + str(variable.name) + " - shape:" + str(shape))
        variable_parametes = 1
        for dim in shape:
            variable_parametes *= dim.value
        if verbose: print("variable parameters: " , variable_parametes)
        total_parameters += variable_parametes
    if verbose: print("total params: ", total_parameters)
    return total_parameters

# Create convolutional layers and depth layers for the final 3 layer
def conv2d_depth_or_not(input_, name, nonlinearity=None):
        with tf.variable_scope(name) as scope:
            
            W_shape = cnn_param.layer_shapes[name + '/W']
            b_shape = cnn_param.layer_shapes[name + '/b']
            
            W_initializer = tf.truncated_normal_initializer(stddev=stddev)
            b_initializer = tf.constant_initializer(0.0)
                
            conv_weights = tf.get_variable(name+"_W", shape=W_shape, initializer=W_initializer)
            conv_biases  = tf.get_variable(name+"_b", shape=b_shape, initializer=b_initializer)

            if name == 'depth':
                # learn different filter for each input channel
                # thus the number of input channel has to be reduced
                conv = tf.nn.depthwise_conv2d_native(input_, conv_weights, [1,1,1,1], padding='SAME')
                # conv = tf.nn.separable_conv2d(input_, conv_weights, [1,1,1,1], padding='SAME')
            else:
                conv = tf.nn.conv2d(input_, conv_weights, [1,1,1,1], padding='SAME')

            bias = tf.nn.bias_add(conv, conv_biases)
            bias = tf.nn.dropout(bias,0.7) 
            if nonlinearity is None: 
                return bias
            return nonlinearity(bias, name=name)

def get_weights(shape, name=''):
    W_initializer = tf.truncated_normal_initializer(stddev=stddev)
    return tf.get_variable(name, shape=shape, initializer=W_initializer)

def get_bias(shape, name=''):
    b_initializer = tf.constant_initializer(0.0)
    return tf.get_variable(name, shape=shape, initializer=b_initializer)

def zero_padding(X, pad=(3, 3)):
    paddings = tf.constant([[0, 0], [pad[0], pad[0]],
                            [pad[1], pad[1]], [0, 0]])
    return tf.pad(X, paddings, 'CONSTANT')

# Create convolutional layer for core resnet
def conv2D(A_prev, filters, k_size, strides, padding, name):
    if (name == 'convReplacement'):
        w1 = n_labels
        w2 = n_labels
    else:
        (m, in_H, in_W, in_C) = A_prev.shape
        w1 = in_C
        w2 = filters

    w_shape = (k_size[0], k_size[1], w1, w2)
    b_shape = (filters, )

    conv_weights = get_weights(shape=w_shape, name=name+'_W')
    b = get_bias(shape=b_shape, name=name+'_b')

    strides = [1, strides[0], strides[1], 1]

    if name == 'depth':
        # learn different filter for each input channel
        # thus the number of input channel has to be reduced
        A = tf.nn.depthwise_conv2d_native(A_prev, conv_weights, strides=strides, padding=padding, name=name)
        # conv = tf.nn.separable_conv2d(input_, conv_weights, [1,1,1,1], padding='SAME')
    else:
        A = tf.nn.conv2d(A_prev, conv_weights, strides=strides, padding=padding, name=name)
    params = {'W':conv_weights, 'b':b, 'A':A}
    return A, params

def batch_norm(X, name):
    m_, v_ = tf.nn.moments(X, axes=[0, 1, 2], keep_dims=False)
    beta_ = tf.zeros(X.shape.as_list()[3])
    gamma_ = tf.ones(X.shape.as_list()[3])
    bn = tf.nn.batch_normalization(X, mean=m_, variance=v_,
                                   offset=beta_, scale=gamma_,
                                   variance_epsilon=1e-4)
    return bn


def identity_block(X, f, filters, stage, block):
    """
    Implementing a ResNet identity block with shortcut path
    passing over 3 Conv Layers

    @params
    X - input tensor of shape (m, in_H, in_W, in_C)
    f - size of middle layer filter
    filters - tuple of number of filters in 3 layers
    stage - used to name the layers
    block - used to name the layers

    @returns
    A - Output of identity_block
    params - Params used in identity block
    """

    conv_name = 'res' + str(stage) + block + '_branch'
    bn_name = 'bn' + str(stage) + block + '_branch'

    l1_f, l2_f, l3_f = filters

    params = {}

    A1, params[conv_name+'2a'] = conv2D(X, filters=l1_f, k_size=(1, 1), strides=(1, 1),
                                        padding='VALID', name=conv_name+'2a')
    A1_bn = batch_norm(A1, name=bn_name+'2a')
    A1_act = tf.nn.relu(A1_bn)
    params[conv_name+'2a']['bn'] = A1_bn
    params[conv_name+'2a']['act'] = A1_bn

    A2, params[conv_name+'2b'] = conv2D(A1_act, filters=l2_f, k_size=(f, f), strides=(1, 1),
                                        padding='SAME', name=conv_name+'2b')
    A2_bn = batch_norm(A2, name=bn_name+'2b')
    A2_act = tf.nn.relu(A2_bn)
    params[conv_name+'2b']['bn'] = A2_bn
    params[conv_name+'2b']['act'] = A2_act

    A3, params[conv_name+'2c'] = conv2D(A2_act, filters=l3_f, k_size=(1, 1), strides=(1, 1),
                                        padding='VALID', name=conv_name+'2c')
    A3_bn=batch_norm(A3, name=bn_name+'2c')

    A3_add = tf.add(A3_bn, X)
    A = tf.nn.relu(A3_add)
    params[conv_name+'2c']['bn'] = A3_bn
    params[conv_name+'2c']['add'] = A3_add
    params['out'] = A
    return A, params


def convolutional_block(X, f, filters, stage, block, s=2):
    """
    Implementing a ResNet convolutional block with shortcut path
    passing over 3 Conv Layers having different sizes

    @params
    X - input tensor of shape (m, in_H, in_W, in_C)
    f - size of middle layer filter
    filters - tuple of number of filters in 3 layers
    stage - used to name the layers
    block - used to name the layers
    s - strides used in first layer of convolutional block

    @returns
    A - Output of convolutional_block
    params - Params used in convolutional block
    """

    conv_name = 'res' + str(stage) + block + '_branch'
    bn_name = 'bn' + str(stage) + block + '_branch'

    l1_f, l2_f, l3_f = filters

    params = {}

    A1, params[conv_name+'2a'] = conv2D(X, filters=l1_f, k_size=(1, 1), strides=(s, s),
                                        padding='VALID', name=conv_name+'2a')
    A1_bn = batch_norm(A1, name=bn_name+'2a')
    A1_act = tf.nn.relu(A1_bn)
    params[conv_name+'2a']['bn'] = A1_bn
    params[conv_name+'2a']['act'] = A1_bn

    A2, params[conv_name+'2b'] = conv2D(A1_act, filters=l2_f, k_size=(f, f), strides=(1, 1),
                                        padding='SAME', name=conv_name+'2b')
    A2_bn = batch_norm(A2, name=bn_name+'2b')
    A2_act = tf.nn.relu(A2_bn)
    params[conv_name+'2b']['bn'] = A2_bn
    params[conv_name+'2b']['act'] = A2_act

    A3, params[conv_name+'2c'] = conv2D(A2_act, filters=l3_f, k_size=(1, 1), strides=(1, 1),
                                        padding='VALID', name=conv_name+'2c')
    A3_bn=batch_norm(A3, name=bn_name+'2c')
    params[conv_name+'2c']['bn'] = A3_bn

    A_, params[conv_name+'1'] = conv2D(X, filters=l3_f, k_size=(1, 1), strides=(s, s),
                                       padding='VALID', name=conv_name+'1')
    A_bn_ = batch_norm(A_, name=bn_name+'1')

    A3_add = tf.add(A3_bn, A_bn_)
    A = tf.nn.relu(A3_add)
    params[conv_name+'2c']['add'] = A3_add
    params[conv_name+'1']['bn'] = A_bn_
    params['out'] = A
    return A, params


# Construct the ResNet-50 model
def ResNet50(input_images):
    params={}
    
    X_input = input_images
    X = zero_padding(X_input, (3, 3))
    
    params['input'] = X_input
    params['zero_pad'] = X

    # Stage 1
    params['stage1'] = {}
    A_1, params['stage1']['conv'] = conv2D(X, filters=64, k_size=(7, 7), strides=(2, 2), padding='VALID', name='conv1')
    A_1_bn = batch_norm(A_1, name='bn_conv1')
    A_1_act = tf.nn.relu(A_1_bn)
    A_1_pool = tf.nn.max_pool(A_1_act, ksize=(1, 3, 3, 1), strides=(1, 2, 2, 1), padding='VALID')
    params['stage1']['bn'] = A_1_bn
    params['stage1']['act'] = A_1_act
    params['stage1']['pool'] = A_1_pool

    # Stage 2
    params['stage2'] = {}
    A_2_cb, params['stage2']['cb'] = convolutional_block(A_1_pool, f=3, filters=[64, 64, 256],
                                                         stage=2, block='a', s=1)
    A_2_ib1, params['stage2']['ib1'] = identity_block(A_2_cb, f=3, filters=[64, 64, 256],
                                                      stage=2, block='b')
    A_2_ib2, params['stage2']['ib2'] = identity_block(A_2_ib1, f=3, filters=[64, 64, 256],
                                                      stage=2, block='c')

    # Stage 3
    params['stage3'] = {}
    A_3_cb, params['stage3']['cb'] = convolutional_block(A_2_ib2, 3, [128, 128, 512],
                                                         stage=3, block='a', s=2)
    A_3_ib1, params['stage3']['ib1'] = identity_block(A_3_cb, 3, [128, 128, 512],
                                                      stage=3, block='b')
    A_3_ib2, params['stage3']['ib2'] = identity_block(A_3_ib1, 3, [128, 128, 512],
                                                      stage=3, block='c')
    A_3_ib3, params['stage3']['ib3'] = identity_block(A_3_ib2, 3, [128, 128, 512],
                                                      stage=3, block='d')

    # Stage 4
    params['stage4'] = {}
    A_4_cb, params['stage4']['cb'] = convolutional_block(A_3_ib3, 3, [256, 256, 1024],
                                                         stage=4, block='a', s=2)
    A_4_ib1, params['stage4']['ib1'] = identity_block(A_4_cb, 3, [256, 256, 1024],
                                                      stage=4, block='b')
    A_4_ib2, params['stage4']['ib2'] = identity_block(A_4_ib1, 3, [256, 256, 1024],
                                                      stage=4, block='c')
    A_4_ib3, params['stage4']['ib3'] = identity_block(A_4_ib2, 3, [256, 256, 1024],
                                                      stage=4, block='d')
    A_4_ib4, params['stage4']['ib4'] = identity_block(A_4_ib3, 3, [256, 256, 1024],
                                                      stage=4, block='e')
    A_4_ib5, params['stage4']['ib5'] = identity_block(A_4_ib4, 3, [256, 256, 1024],
                                                      stage=4, block='f')

    # Stage 5
    params['stage5'] = {}
    A_5_cb, params['stage5']['cb'] = convolutional_block(A_4_ib5, 3, [512, 512, 2048],
                                                         stage=5, block='a', s=2)
    A_5_ib1, params['stage5']['ib1'] = identity_block(A_5_cb, 3, [512, 512, 2048],
                                                      stage=5, block='b')
    A_5_ib2, params['stage5']['ib2'] = identity_block(A_5_ib1, 3, [512, 512, 2048],
                                                      stage=5, block='c')


    # feature wise convolution layers, no non-linearity
    conv_depth_1 = conv2d_depth_or_not(A_5_cb, "convFeatures")
    # two layer of feature-wise convolution, a cubic feature transformation
    conv_depth   = conv2d_depth_or_not(conv_depth_1, "depth")

    # this is a replcement of last FCL layer from VGG (common in GAP & GMP models)
    # this layer does not have non-nonlinearity
    
    conv_last = conv2d_depth_or_not(conv_depth, "convReplacement")
    gap       = tf.reduce_mean(conv_last, [1,2])

    last_features = 1024

    with tf.variable_scope("GAP"):
        gap_w = tf.get_variable("convReplacement_W", shape=(last_features, n_labels),
                initializer=tf.random_normal_initializer(stddev=stddev))

    class_prob = tf.matmul(gap, gap_w)

    # print_model_params()
    return conv_last, gap, class_prob

def p(t):
    print (t.name, t.get_shape())


# Returns classmaps for the trained model
def get_classmap(class_, conv_last):
    with tf.variable_scope("GAP", reuse=True):
        class_w = tf.gather(tf.transpose(tf.get_variable("convReplacement_W")), class_)
        class_w = tf.reshape(class_w, [-1, cnn_param.last_features, 1]) 
    conv_last_ = tf.image.resize_bilinear(conv_last, [hyper.image_h, hyper.image_w])
    conv_last_ = tf.reshape(conv_last_, [-1, hyper.image_h*hyper.image_w, cnn_param.last_features]) 
    classmap   = tf.reshape(tf.matmul(conv_last_, class_w), [-1, hyper.image_h,hyper.image_w])
    return classmap

