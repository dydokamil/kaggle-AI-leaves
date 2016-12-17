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

from Deep_Learning.kaggle.leaves.tools import load_images, crop_to_first, random_batch_distorted

BATCH_SIZE = 64
NB_EPOCH = 20000
N_CLASSES = 99
IMAGE_RESOLUTION = (224, 224)
ADDITIONAL_FEATURES_LEN = 192

path = '/home/kamil/Documents/kaggle/leaves/'
path_images = '/home/kamil/Documents/kaggle/leaves/images/'
train_path = path + 'train.csv'
test_path = path + 'test.csv'

# Load all labels
train = pd.read_csv(train_path)
labels = train['species'].tolist()

# Load all training images
all_images = load_images(path_images, ids=train.id)[:64]  # TODO: remove [:64]

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

# Create a model (keras)
model = Sequential()
model.add(Convolution2D(96, 11, 11, subsample=(4, 4), activation='relu',
                        input_shape=(IMAGE_RESOLUTION[0], IMAGE_RESOLUTION[1]) + (1,)))
model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), border_mode='same'))
model.add(BatchNormalization())
model.add(Convolution2D(256, 5, 5, activation='relu', border_mode='same'))
model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), border_mode='same'))
model.add(BatchNormalization())
model.add(Convolution2D(384, 3, 3, activation='relu', border_mode='same'))
model.add(Convolution2D(384, 3, 3, activation='relu', border_mode='same'))
model.add(Convolution2D(256, 3, 3, activation='relu', border_mode='same'))
model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), border_mode='same'))
model.add(BatchNormalization())
model.add(Flatten())
model.add(Dense(4096, activation='tanh'))
model.add(Dropout(.5))
model.add(Dense(4096, activation='tanh'))
model.add(Dropout(.5))
model.add(Dense(N_CLASSES, activation='softmax'))

model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='sgd')

for e in range(NB_EPOCH):
    X, y = random_batch_distorted(list(all_images.values()), onehot_labels, BATCH_SIZE, IMAGE_RESOLUTION,
                                  distorted=True)
    X = np.array(X)
    loss = model.fit(X, y, batch_size=BATCH_SIZE, nb_epoch=40)
