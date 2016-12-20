# coding=utf-8

import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from collections import Counter

from Deep_Learning.kaggle.leaves.tools import load_images, onehot_encode, crop_to_first, IMAGE_RESOLUTION, \
    ADDITIONAL_FEATURES_LEN, N_CLASSES, get_model, random_batch_distorted, random_batch_testing, get_encoders

POOL_SIZE = 21  # number of crops

path = '/home/kamil/Documents/kaggle/leaves/'
path_images = '/home/kamil/Documents/kaggle/leaves/images/'
test_path = path + 'test.csv'
# Choose the model version
model_weights_path = '/media/kamil/c0a6bdfe-d860-4f81-8a6f-1f1d714ac49f/keras/leaves/540v8.h5'

# Load all labels
test = pd.read_csv(test_path)

# Load all training images
all_images = load_images(path_images, ids=test.id)

# Get the batch size
batch_size = len(all_images)

# Create a dict of 'id': additional features
additional = test.drop(['id'], 1).T.to_dict('list')
additional = list(additional.values())

# Crop all images to the first occurence of a white pixel
all_images = np.asarray([crop_to_first(img) for img in all_images])  # convert to array + crop

# Normalize all images
min_max_scaler = preprocessing.MinMaxScaler()
all_images = [min_max_scaler.fit_transform(x) for x in all_images]

label_encoder, oh_encoder = get_encoders()

# Get the model (keras)
model = get_model(IMAGE_RESOLUTION, ADDITIONAL_FEATURES_LEN, N_CLASSES)
model.load_weights(model_weights_path)

for i in range(len(all_images)):
    X_test, Z_test = random_batch_testing(all_images[i], POOL_SIZE, IMAGE_RESOLUTION, additional[i])
    prediction = model.predict_classes([X_test, Z_test])
    predicted_class = Counter(prediction).most_common(1)[0][0]
    predicted_class_name = label_encoder.classes_[predicted_class]
