# coding=utf-8
# This is a tensorflow/keras model to classify leaves based on images and some additional data

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

from Deep_Learning.kaggle.leaves.tools import crop_to_first, load_images, batch_random_crop_and_distort, \
    weight_variable, \
    bias_variable

# settings
BATCH_SIZE = 50
N_CLASSES = 99
IMAGE_RESOLUTION = (227, 227)

path = '/home/kamil/Documents/kaggle/leaves/'
path_images = '/home/kamil/Documents/kaggle/leaves/images/'
train_path = path + 'train.csv'
test_path = path + 'test.csv'

# Load all labels
train = pd.read_csv(train_path)
labels = train['species'].tolist()

# Load all training images
all_images = load_images(path_images, ids=train.id)

# Create a dict of 'id': additional features
additional = train.drop('species', 1).set_index('id').T.to_dict('list')

# Create a dict of 'id': species
train_ids_species = train[['id', 'species']].set_index('id').T.to_dict('records')[0]

# Encode labels
label_encoder = LabelEncoder()
label_encoder.fit(labels)
labels_int = label_encoder.transform(labels)

oh_encoder = OneHotEncoder()
onehot_labels = oh_encoder.fit_transform([[x] for x in labels_int])
onehot_labels = onehot_labels.toarray()  # array of one-hot encodings for each label

# Crop all images to the first occurence of a white pixel
all_images = np.asarray([crop_to_first(img) for img in all_images])  # convert to array + crop

# Normalize all images
min_max_scaler = preprocessing.MinMaxScaler()
all_images = [min_max_scaler.fit_transform(x) for x in all_images]

# Create a dict of labels and their images: {'id': [image]}
all_images = {iden: img for iden, img in zip(list(train_ids_species.keys()), all_images)}

# Create a model
x = tf.placeholder(tf.float32, shape=[None, 227, 227, 1])
y = tf.placeholder(tf.float32, shape=[None, N_CLASSES])

# 1st conv layer
W_conv1 = weight_variable([11, 11, 1, 96])
b_conv1 = bias_variable([96])

h_conv1 = tf.nn.relu(tf.nn.conv2d(x, W_conv1, strides=[1, 4, 4, 1], padding='SAME') + b_conv1)
h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
normalization_h1 = tf.nn.local_response_normalization(h_pool1)

# 2nd conv layer
W_conv2 = weight_variable([5, 5, 96, 256])
b_conv2 = bias_variable([256])

h_conv2 = tf.nn.relu(tf.nn.conv2d(normalization_h1, W_conv2, strides=[1, 1, 1, 1], padding='SAME') + b_conv2)
h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
normalization_h2 = tf.nn.local_response_normalization(h_pool2)

# 3rd conv layer
W_conv3 = weight_variable([3, 3, 256, 384])
b_conv3 = bias_variable([384])
h_conv3 = tf.nn.relu(tf.nn.conv2d(normalization_h2, W_conv3, strides=[1, 1, 1, 1], padding='SAME') + b_conv3)

# 4th conv layer
W_conv4 = weight_variable([3, 3, 384, 384])
b_conv4 = bias_variable([384])
h_conv4 = tf.nn.relu(tf.nn.conv2d(h_conv3, W_conv4, strides=[1, 1, 1, 1], padding='SAME') + b_conv4)

# 5th conv layer
W_conv5 = weight_variable([3, 3, 384, 256])
b_conv5 = bias_variable([256])
h_conv5 = tf.nn.relu(tf.nn.conv2d(h_conv4, W_conv5, strides=[1, 1, 1, 1], padding='SAME') + b_conv5)
h_pool5 = tf.nn.max_pool(h_conv5, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
normalization_h5 = tf.nn.local_response_normalization(h_pool5)

# 1st fully connected layer
# TODO: add additional features from the csv file
W_fc1 = weight_variable([256 * 13 * 13, 4096])
b_fc1 = bias_variable([4096])

normalization_h5_flat = tf.reshape(normalization_h5, [-1, 256 * 13 * 13])
h_fc1 = tf.nn.tanh(tf.matmul(normalization_h5_flat, W_fc1) + b_fc1)
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# 2nd fully connected layer
W_fc2 = weight_variable([4096, 4096])
b_fc2 = bias_variable([4096])

h_fc2 = tf.nn.tanh(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)

# 3rd fully connected layer
W_fc3 = weight_variable([4096, N_CLASSES])
b_fc3 = bias_variable([N_CLASSES])

y_conv = tf.nn.softmax(tf.matmul(h_fc2_drop, W_fc3) + b_fc3)

# Create a model training sequence
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y * tf.log(y_conv), reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

for i in range(20000):
    batch = batch_random_crop_and_distort(BATCH_SIZE)
    if i % 100 == 0:
        train_accuracy = accuracy.eval(feed_dict={x: batch[0], y: batch[1], keep_prob: 1.})
        print('step %d, training accuracy %g' % (i, train_accuracy))
    train_step.run(feed_dict={x: batch[0], y: batch[1], keep_prob: .5})

batch = batch_random_crop_and_distort(990)
print('test accuracy %g' % accuracy.eval(feed_dict={x: batch[0], y: batch[1], keep_prob: 1.}))
