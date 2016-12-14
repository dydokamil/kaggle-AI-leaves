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

from Deep_Learning.kaggle.leaves.tools import crop_to_first, load_images, batch_random_crop_and_distort

# settings
BATCH_SIZE = 128

path = '/home/kamil/Documents/kaggle/leaves/'
path_images = '/home/kamil/Documents/kaggle/leaves/images/'
train_path = path + 'train.csv'
test_path = path + 'test.csv'

# Load all images
all_images = load_images(path_images)[:20]

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
all_images = np.asarray([crop_to_first(img) for img in all_images])  # convert to array + crop

# TODO: Create a dict of labels and their images: {'id': [image]}
all_images = {id: img for id, img in zip(list(train_ids_species.keys()), all_images)}

# TODO: uncomment it
# Normalize all images
# min_max_scaler = preprocessing.MinMaxScaler()
# X_train_minmax = [min_max_scaler.fit_transform(x) for x in all_images]
# ####################

# TODO: Resize all images

# TODO: create a model

# TODO: use batch_random_crop_and_distort to generate batches of distorted images

# img_framed.thumbnail((224, 224), Image.ANTIALIAS)
# img_framed.show()
