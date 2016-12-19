# coding=utf-8
# This is a tensorflow/keras model to classify leaves based on images and some additional data

import numpy as np
import pandas as pd
from keras.engine import Merge
from keras.layers import Dense, Convolution2D, MaxPooling2D, BatchNormalization, Dropout, Flatten
from keras.models import Sequential
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

from Deep_Learning.kaggle.leaves.tools import load_images, crop_to_first, random_batch_distorted, get_model, \
    IMAGE_RESOLUTION, ADDITIONAL_FEATURES_LEN, N_CLASSES

BATCH_SIZE = 130
VALID_SIZE = 990
NB_EPOCH = 35
ITERATIONS = 20000

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

# Get the model (keras)
model = get_model(IMAGE_RESOLUTION, ADDITIONAL_FEATURES_LEN, N_CLASSES)

if False:
    model.load_weights('/media/kamil/c0a6bdfe-d860-4f81-8a6f-1f1d714ac49f/keras/leaves/570v3.h5')

for e in range(ITERATIONS):
    X_train, y_train, Z_train = random_batch_distorted(list(all_images.values()), onehot_labels, BATCH_SIZE,
                                                       IMAGE_RESOLUTION, list(additional.values()), distorted=True)
    X_valid, y_valid, Z_valid = random_batch_distorted(list(all_images.values()), onehot_labels, VALID_SIZE,
                                                       IMAGE_RESOLUTION, list(additional.values()), distorted=False)
    model.fit([X_train, Z_train], y_train, batch_size=BATCH_SIZE, nb_epoch=NB_EPOCH,
              validation_data=([X_valid, Z_valid], y_valid))
    if e % 30 == 0:
        model.save_weights('/media/kamil/c0a6bdfe-d860-4f81-8a6f-1f1d714ac49f/keras/leaves/' + str(e) + 'v6.h5')
