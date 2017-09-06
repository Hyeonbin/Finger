# -*- coding: utf-8 -*-
import os
import tensorflow as tf
import numpy as np

def search(dirname):
    filenames = os.listdir(dirname)
    for filename in filenames:
        full_filename = os.path.join(dirname, filename)
        return filenames

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial);

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')

def Nextbatch(data, label, batchsize):
    idx = np.arange(0 , len(data))
    np.random.shuffle(idx)
    idx = idx[:batchsize]

    _data = [data[i] for i in idx]
    _label = [label[i] for i in idx]

    return np.asarray(_data), np.asarray(_label)

def _DropOut(model):
    keep_prob = tf.placeholder(tf.float32)
    dropout = tf.nn.dropout(model, keep_prob)

    return dropout, keep_prob


def _CNNModel(num):
    x = tf.placeholder("float", shape=[None, num])
    x_image = tf.reshape(x, [-1, 72, 40, 1])

    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    return h_pool2, x

def _FlatModel(model):
    W_fc1 = weight_variable([18 * 10 * 64, 256])
    b_fc1 = bias_variable([256])
    h_pool2_flat = tf.reshape(model, [-1, 18*10*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    return h_fc1

def _SoftMax(model):
    W_fc2 = weight_variable([256, 5])
    b_fc2 = bias_variable([5])

    y_conv = tf.matmul(model, W_fc2) + b_fc2

    return y_conv

def _SetAccuracy(model, num):
    y_ = tf.placeholder("float", shape=[None, num])

    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_,logits=model))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(model,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    return train_step, accuracy, y_, correct_prediction
