# coding=utf-8
# This is a tensorflow/keras model to classify leaves based on images and some additional data

import numpy as np
import pandas as pd
from sklearn import preprocessing

from tools import load_images, crop_to_first, IMAGE_RESOLUTION, ADDITIONAL_FEATURES_LEN, N_CLASSES, get_model, \
    random_batch_distorted, NB_EPOCH
from tools import onehot_encode

BATCH_SIZE = 32
VALID_SIZE = 200
ITERATIONS = 20000

path = '/home/kamil/Documents/kaggle/leaves/'
path_images = '/home/kamil/Documents/kaggle/leaves/images/'
train_path = path + 'train.csv'

# Load all labels
train = pd.read_csv(train_path)
labels = train['species'].tolist()

# Load all training images
all_images = load_images(path_images, ids=train.id)

# Create a dict of 'id': additional features
additional = train.drop('species', 1).set_index('id').T.to_dict('list')
additional = list(additional.values())

# Encode labels
onehot_labels = onehot_encode(labels)

# Crop all images to the first occurence of a white pixel
all_images = np.asarray([crop_to_first(img) for img in all_images])  # convert to array + crop

# Normalize all images
min_max_scaler = preprocessing.MinMaxScaler()
all_images = [min_max_scaler.fit_transform(x) for x in all_images]

# Get the model (keras)
model = get_model(IMAGE_RESOLUTION, ADDITIONAL_FEATURES_LEN, N_CLASSES)

if True:
    model.load_weights('/media/kamil/c0a6bdfe-d860-4f81-8a6f-1f1d714ac49f/keras/leaves/145v14small.h5')

for e in range(ITERATIONS):
    X_train, y_train, Z_train = random_batch_distorted(all_images, onehot_labels, BATCH_SIZE,
                                                       IMAGE_RESOLUTION, additional, distorted=True)
    X_valid, y_valid, Z_valid = random_batch_distorted(all_images, onehot_labels, VALID_SIZE,
                                                       IMAGE_RESOLUTION, additional, distorted=False)
    model.fit([X_train, Z_train], y_train, batch_size=BATCH_SIZE, nb_epoch=NB_EPOCH)
    loss = model.evaluate([X_valid, Z_valid], y_valid, VALID_SIZE)

    # logging
    print('validation loss, accuracy: ', loss)

    with open("/media/kamil/c0a6bdfe-d860-4f81-8a6f-1f1d714ac49f/keras/leaves/losses5.txt", "a") as myfile:
        myfile.write(str(loss))

    if e % 5 == 0 and e != 0:
        model.save_weights('/media/kamil/c0a6bdfe-d860-4f81-8a6f-1f1d714ac49f/keras/leaves/' + str(e) + 'v14small.h5')
