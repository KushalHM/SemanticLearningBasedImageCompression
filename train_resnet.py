from __future__ import division
from __future__ import print_function
import math
import pandas as pd
from time import time

import tensorflow as tf
import numpy as np
import os


from resnet_model import ResNet50
from util import *
from params import TrainingParams, HyperParams
from saveDataNp import load_train_data, load_test_data

tparam = TrainingParams(verbose=True)  
hyper  = HyperParams(verbose=True)


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Read the images to be used for training and testing
data_train    = pd.read_pickle(tparam.data_train_path)
data_test     = pd.read_pickle(tparam.data_test_path)
len_train     = len(data_train)
len_test      = len(data_train)
train_b_num   = int(math.ceil(len_train/tparam.batch_size))
test_b_num    = int(math.ceil(len_train/tparam.batch_size))

# Create TensorFlow placeholders
images_tf     = tf.placeholder(tf.float32, [None, hyper.image_h, hyper.image_w, hyper.image_c], name = "images")

if hyper.sparse:
    labels_tf = tf.placeholder(tf.int64,   [None], name = 'labels')
else:
    labels_tf = tf.placeholder(tf.int64, [None, hyper.n_labels], name = 'labels')

n_labels    = 257

# Create new ResNet model
_,_,prob_tf   = ResNet50(images_tf)


if hyper.sparse:
    loss   = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=prob_tf, labels=labels_tf))
else:
    loss   = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prob_tf, labels=labels_tf))

# Initialize the optimizer
train_op = tf.train.AdamOptimizer(1e-3).minimize(loss)

train_loss    = tf.summary.scalar("training_loss", loss)
test_loss     = tf.summary.scalar("validation_loss", loss)
np_data_train = load_train_data()

np_train_csv = pd.read_csv(tparam.train_csv)

saver = tf.train.Saver()

def sparse_labels_or_not(batch):
    if hyper.sparse:
        return batch['label'].values
    else:
        labels = np.zeros((len(batch), hyper.n_labels))
        for i,j in enumerate(batch['label'].values):
            labels[i,j] = 1
        return labels


# Begin Training and Validation
with tf.Session() as sess:
    try:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        summary_writer = tf.summary.FileWriter('tensorboards', sess.graph)
        if tparam.resume_training:
        	saver.restore(sess, tparam.model_path + 'model')
        	print("model restored...")
        	
        ix = 0
        for epoch in range(tparam.num_epochs):
            start = time()
            # Training
            epoch_loss = 0
            for b, train_batch in enumerate(chunker(data_train.sample(frac=1),tparam.batch_size)):
                train_images  = np.array(list(map(lambda i: load_image(i, np_data_train, np_train_csv), train_batch['image_path'].values)))
                train_labels  = sparse_labels_or_not(train_batch)
                _, batch_loss, loss_sw = sess.run([train_op, loss, train_loss], feed_dict={images_tf: train_images, labels_tf: train_labels})
                average_batch_loss = np.average(batch_loss)

                epoch_loss += average_batch_loss
                summary_writer.add_summary(loss_sw, epoch*train_b_num+b)
                print("Train: epoch:{}, batch:{}/{}, loss:{}".format(epoch, b, train_b_num, average_batch_loss))

            print("Train: epoch:{}, total loss:{}".format(epoch, epoch_loss/train_b_num))

            # Validation
            validation_loss = 0
            for b, test_batch in enumerate(chunker(data_test,tparam.batch_size)): # no need to randomize test batch
                test_images        = np.array(list(map(lambda i: load_image_disk(i), test_batch['image_path'].values)))
                test_labels        = sparse_labels_or_not(test_batch)
                batch_loss,loss_sw = sess.run([loss, test_loss], feed_dict={images_tf: test_images, labels_tf: test_labels})
                summary_writer.add_summary(loss_sw, epoch*test_b_num+b)
            
            print("Test: epoch:{}, total loss:{}".format(epoch, validation_loss/b))
            print("Time for one epoch:{}".format(time()-start))
            with open('time.txt', 'a') as fid:
                fid.write(str(epoch)+','+str(epoch_loss/train_b_num)+','+str(validation_loss/b)+','+str(time()-start)+'\n')

            # Save the model
            saver.save(sess, tparam.model_path + '/model')


    finally:
        sess.close()


# Start training
if __name__ == '__main__':
    train()