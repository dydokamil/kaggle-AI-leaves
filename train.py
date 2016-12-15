# coding=utf-8

'''
This is a tensorflow/keras model to classify leaves based on images and some additional data
'''

import tensorflow as tf
import numpy as np
import time
from PIL import Image
import matplotlib.pyplot as plt
from keras.preprocessing.image import load_img, img_to_array
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from tflearn import DNN
from tflearn import conv_2d
from tflearn import dropout
from tflearn import fully_connected
from tflearn import input_data
from tflearn import local_response_normalization
from tflearn import max_pool_2d
from tflearn import regression

from Deep_Learning.kaggle.leaves.tools import crop_to_first, load_images, batch_random_crop_and_distort, \
    random_batch_distorted

# settings
BATCH_SIZE = 128
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
# min_max_scaler = preprocessing.MinMaxScaler()
# all_images = [min_max_scaler.fit_transform(x) for x in all_images]

# Create a dict of labels and their images: {'id': [image]}
all_images = {iden: img for iden, img in zip(list(train_ids_species.keys()), all_images)}

X_batch, y_batch = random_batch_distorted(list(all_images.values()), onehot_labels, 10, IMAGE_RESOLUTION)

# TODO Create a model
