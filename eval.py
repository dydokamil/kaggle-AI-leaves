# coding=utf-8

# Get the model (keras)
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

from Deep_Learning.kaggle.leaves.tools import get_model, IMAGE_RESOLUTION, ADDITIONAL_FEATURES_LEN, N_CLASSES, \
    random_batch_distorted, load_images

import pandas as pd
import numpy as np

path = '/home/kamil/Documents/kaggle/leaves/'
path_images = '/home/kamil/Documents/kaggle/leaves/images/'
train_path = path + 'train.csv'
test_path = path + 'test.csv'

train = pd.read_csv(train_path)
labels = train['species'].tolist()

all_images = np.array(load_images(path_images, ids=train.id))

# Encode labels
label_encoder = LabelEncoder()
label_encoder.fit(labels)
labels_int = label_encoder.transform(labels)

oh_encoder = OneHotEncoder()
onehot_labels = oh_encoder.fit_transform([[x] for x in labels_int])
onehot_labels = onehot_labels.toarray()  # array of one-hot encodings for each label

additional = train.drop('species', 1).set_index('id').T.to_dict('list')

model = get_model(IMAGE_RESOLUTION, ADDITIONAL_FEATURES_LEN, N_CLASSES)

model.load_weights('/media/kamil/c0a6bdfe-d860-4f81-8a6f-1f1d714ac49f/keras/leaves/120v4.h5')

X, y, Z = random_batch_distorted(all_images, onehot_labels, 990, IMAGE_RESOLUTION,
                                 list(additional.values()), distorted=False)
loss = model.evaluate([X, Z], y, 990)
print('loss: ', loss)
