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

from Deep_Learning.kaggle.leaves.tools import crop_to_first, add_frame, distorted_input, load_images, batch_random_crop

# settings
LEAF_CROPS = 5

path = '/home/kamil/Documents/kaggle/leaves/'
path_images = '/home/kamil/Documents/kaggle/leaves/images/'
train_path = path + 'train.csv'
test_path = path + 'test.csv'

# Load all images
all_images = load_images(path_images)

# Load all labels
train = pd.read_csv(train_path)
labels = train['species'].tolist()

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
all_images = [crop_to_first(np.asarray(img)) for img in all_images]  # convert to array + crop

# Randomly crop all images using a square "filter"
# 1st dim = a leaf; 2nd dim = one of LEAF_CROPS crops of the leaf; 3rd, 4th dim = image
all_images_randomly_cropped = [y for x in all_images for y in [batch_random_crop(x, LEAF_CROPS)]]

# TODO: Randomly distort images
distorted_images = np.asarray([[distorted_input(x, 1) for x in x] for x in all_images_randomly_cropped])

# TODO: Resize all images

# TODO: Create a dict of images and their labels (w/ distorted images) 'id': [images]

# TODO: uncomment it
# Normalize all images
# min_max_scaler = preprocessing.MinMaxScaler()
# X_train_minmax = [min_max_scaler.fit_transform(x) for x in all_images]
# ####################

# TODO: create a model


# img = Image.open(path_images + '/1573.jpg')
# img_arr = np.asarray(img)
# cropped = crop_to_first(img_arr)
# Image.fromarray(cropped).show()

# framed = add_frame(cropped)
# img_framed = Image.fromarray(framed)

# img_framed.thumbnail((224, 224), Image.ANTIALIAS)
# img_framed.show()

# distorted = distorted_input(path_images + '/1583.jpg', 5)
# Image.fromarray(np.reshape(distorted[0], [234, 1295])).show()

# randomly_cropped = random_crop(img_arr)
# Image.fromarray(randomly_cropped).show()
