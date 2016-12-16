# coding=utf-8
# This is a tensorflow/keras model to classify leaves based on images and some additional data

import numpy as np
import pandas as pd
import tensorflow as tf
import tflearn
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from tflearn import conv_2d
from tflearn import dropout
from tflearn import fully_connected
from tflearn import input_data
from tflearn import local_response_normalization
from tflearn import max_pool_2d
from tflearn import regression

from Deep_Learning.kaggle.leaves.tools import load_images, crop_to_first, random_batch_distorted

BATCH_SIZE = 990
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
network = input_data(shape=[None, IMAGE_RESOLUTION[0], IMAGE_RESOLUTION[1], 1])
network = conv_2d(network, 96, 11, strides=4, activation='relu')
network = max_pool_2d(network, 3, strides=2)
network = local_response_normalization(network)
network = conv_2d(network, 256, 5, activation='relu')
network = max_pool_2d(network, 3, strides=2)
network = local_response_normalization(network)
network = conv_2d(network, 384, 3, activation='relu')
network = conv_2d(network, 384, 3, activation='relu')
network = conv_2d(network, 256, 3, activation='relu')
network = max_pool_2d(network, 3, strides=2)
network = local_response_normalization(network)
network = fully_connected(network, 4096, activation='tanh')
network = dropout(network, .5)
network = fully_connected(network, 4096, activation='tanh')
network = dropout(network, .5)
network = fully_connected(network, N_CLASSES, activation='softmax')
network = regression(network, optimizer='momentum', loss='categorical_crossentropy', learning_rate=.001)

col = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
for x in col:
    tf.add_to_collection(tf.GraphKeys.VARIABLES, x)

model = tflearn.DNN(network, checkpoint_path='model_alexnet', max_checkpoints=1, tensorboard_verbose=2)

# for i in range(20000):
batch = random_batch_distorted(list(all_images.values()), onehot_labels, BATCH_SIZE, IMAGE_RESOLUTION)
model.fit(batch[0], batch[1], n_epoch=1000, validation_set=.1, shuffle=True, show_metric=True,
          batch_size=BATCH_SIZE, snapshot_step=200, snapshot_epoch=False, run_id='alexnet')
