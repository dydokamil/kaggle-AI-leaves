# coding=utf-8

import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from collections import Counter

from Deep_Learning.kaggle.leaves.tools import load_images, onehot_encode, crop_to_first, IMAGE_RESOLUTION, \
    ADDITIONAL_FEATURES_LEN, N_CLASSES, get_model, random_batch_distorted, random_batch_testing, get_encoders

POOL_SIZE = 31  # number of crops

path = '/home/kamil/Documents/kaggle/leaves/'
path_images = '/home/kamil/Documents/kaggle/leaves/images/'
test_path = path + 'test.csv'
submission_path = path + 'submission4.csv'
sample_submission_path = path + 'sample_submission.csv'
# Choose the model version
model_weights_path = '/media/kamil/c0a6bdfe-d860-4f81-8a6f-1f1d714ac49f/keras/leaves/20v13small.h5'

# Load all labels
test = pd.read_csv(test_path)

# Load all training images
all_images = load_images(path_images, ids=test.id)

ids = test.id.tolist()

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

predictions = []

for i in range(len(all_images)):
    X_test, Z_test = random_batch_testing(all_images[i], POOL_SIZE, IMAGE_RESOLUTION, additional[i])
    prediction = model.predict_proba([X_test, Z_test])
    prediction = sum(prediction) / POOL_SIZE
    predictions.append(prediction)

predictions = np.array(predictions)
ids = np.array([[x] for x in ids])

sample_submission_df = pd.read_csv(sample_submission_path)
column_names = list(sample_submission_df.columns.values)[1:]
ids = sample_submission_df.iloc[:, 0]

submission_df = pd.DataFrame(data=predictions, columns=column_names, index=ids)
submission_df.to_csv(submission_path, float_format='%.16f')
