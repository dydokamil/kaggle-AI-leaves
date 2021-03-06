# coding=utf-8
import os
import random

import numpy as np
import sys
from PIL import Image
from keras.engine import Merge
from keras.layers import Convolution2D, MaxPooling2D, BatchNormalization, Flatten, Dense, Dropout
from keras.models import Sequential
from keras.optimizers import SGD, Adam
from keras.preprocessing.image import ImageDataGenerator, img_to_array
from keras.utils.visualize_util import plot
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import pickle

N_CLASSES = 99
IMAGE_RESOLUTION = (150, 150)
ADDITIONAL_FEATURES_LEN = 192
NB_EPOCH = 35


def crop_to_first(image):
    '''
    Helper function that removes "empty lines"
    :param image: image, uint8 array
    :return: image, uint8 array
    '''
    image = image[~np.all(image == 0, axis=1)]  # rows
    image = image[:, ~np.all(image == 0, axis=0)]  # cols
    return np.asarray(image)


def add_frame(image):
    '''
    Helper function that adds a frame around the image to make it a square
    :param image: image, uint8 array
    :return: image, uint8 array
    '''
    difference = np.abs(image.shape[0] - image.shape[1])
    first_frame = int(np.floor(difference / 2.))
    second_frame = int(np.ceil(difference / 2.))

    if image.shape[0] < image.shape[1]:  # more columns than rows
        image = np.vstack((image, np.zeros((first_frame, image.shape[1]))))
        image = np.vstack((np.zeros((second_frame, image.shape[1])), image))

    elif image.shape[0] > image.shape[1]:  # more rows than columns
        image = np.hstack((image, np.zeros((image.shape[0], first_frame))))
        image = np.hstack((np.zeros((image.shape[0], second_frame)), image))

    return image


def __distort_input(img, count=1):
    '''
    Distorts images
    :param img: Image to distort
    :param count: how many images to return
    :return: array of distorted images
    '''
    assert count > 0
    datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='constant')

    x = img_to_array(img)
    x = x.reshape((1,) + x.shape)

    i = 0
    images = []
    for batch in datagen.flow(x, batch_size=1):
        images.append(batch[0])
        i += 1
        if i >= count:
            return np.array(images).squeeze()


def onehot_encode(labels):
    label_encoder = LabelEncoder()
    label_encoder.fit(labels)
    labels_int = label_encoder.transform(labels)

    oh_encoder = OneHotEncoder()
    onehot_labels = oh_encoder.fit_transform([[x] for x in labels_int])
    onehot_labels = onehot_labels.toarray()  # array of one-hot encodings for each label

    pickle.dump(label_encoder, open("label_encoder.pickle", "wb"))
    pickle.dump(oh_encoder, open("oh_encoder.pickle", "wb"))

    return onehot_labels


def get_encoders():
    try:
        label_encoder = pickle.load(open("label_encoder.pickle", "rb"))
        oh_encoder = pickle.load(open("oh_encoder.pickle", "rb"))
    except FileNotFoundError as e:
        print('Files not found. Run onehot_encode() from train.py first')
        sys.exit()

    return label_encoder, oh_encoder


def __random_crop(image):
    '''
    Randomly selects a square from an image along shorter axis
    :param image: image, uint8 array
    :return: random square crop
    '''
    if image.shape[0] == image.shape[1]:
        return image

    size = image.shape[0] if image.shape[0] < image.shape[1] else image.shape[1]

    if image.shape[0] < image.shape[1]:  # more columns
        random_start = np.random.randint(0, image.shape[1] - size)
        return np.asarray(image[:, random_start:random_start + size])

    elif image.shape[0] > image.shape[1]:  # more rows
        random_start = np.random.randint(0, image.shape[0] - size)
        return np.asarray(image[random_start:random_start + size, :])


def __batch_random_crop(image, batch_size):
    '''
    Returns a batch of square crops from /image/
    :param image: numpy array image
    :param batch_size: how many images to return
    :return: array of square images
    '''
    return np.asarray([y for x in range(batch_size) for y in [__random_crop(image)]])


def random_crop_and_distort(image, distort=True):
    return __random_crop(__distort_input(image) if distort else image)


def batch_random_crop_and_distort(image, count):
    '''
    Randomly selects a part of the image and distorts it
    :param image: image to distort
    :param count: how many images to return
    :return: batch of distorted images
    '''

    distorted_array = __distort_input(image, count)
    return np.asarray([__random_crop(distorted) for distorted in distorted_array])


def resize_image(img, shape):
    return np.asarray(Image.fromarray(img).resize(shape))


def add_dim(img):
    return np.asarray(np.reshape(img, img.shape + (1,)))


def load_images(directory, ids):
    '''
    Load all images from a specified directory
    :param directory: directory to load images from
    :return: array of images
    '''
    ids = ids.tolist()
    files = os.listdir(directory)
    path_files = [directory + str(id) + '.jpg' for id in ids]
    return np.asarray([np.asarray(Image.open(image)) for image in path_files])


def random_batch_distorted(images, labels, batch_size, shape, additional, distorted=True):
    '''
    This function takes the entire list of images & labels and returns a random batch of [images] and [labels] of /size/
    for training
    :param images: 3D list: the entire list of images
    :param labels: 2D list: the entire list of labels
    :param batch_size: how many samples to return
    :param shape: shape of the images to return
    :param additional: the additional information about the images
    :return: 4D list [images]: [batch_size, x, y, 1], 2D list [labels]: [sample number, label], 2D additional features
    '''
    assert len(images) >= batch_size

    choices = random.sample(range(len(images)), batch_size)
    samples_distorted = [add_dim(resize_image(random_crop_and_distort(image, distorted), shape)) for image in
                         np.asarray(images)[choices]]

    return np.array(samples_distorted), labels[choices], np.array(additional)[choices]


def random_batch_testing(image, num_images, shape, additional):
    '''
    This function returns a batch of /num_images/ crops of each image for testing
    :param image: the sample to crop
    :param num_images: number of images to return per sample
    :param shape: shape of each image
    :param additional: additional information from the file
    :return: batch of shape [images per sample, x, y, 1], additional information
    '''

    img_arr = []
    additional_arr = []

    for i in range(num_images):
        img_arr.append(add_dim(resize_image(random_crop_and_distort(image, distort=False), shape)))
        additional_arr.append(additional)

    return np.array(img_arr), np.array(additional_arr)


def get_model(img_res, additional_features_len, n_classes):
    '''
    Creates and returns a pre-trained keras model
    :param img_res: 2D tuple of x and y resolution
    :param additional_features_len: length of additional features (columns)
    :param n_classes: number of classes
    :return: pre-trained keras model
    '''
    model = Sequential()
    model.add(Convolution2D(64, 8, 8, subsample=(3, 3), activation='relu',
                            input_shape=(img_res[0], img_res[1]) + (1,)))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), border_mode='same'))
    model.add(Convolution2D(128, 4, 4, activation='relu', border_mode='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), border_mode='same'))
    model.add(Convolution2D(256, 3, 3, activation='relu', border_mode='same'))
    model.add(Convolution2D(256, 3, 3, activation='relu', border_mode='same'))
    model.add(Convolution2D(128, 3, 3, activation='relu', border_mode='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), border_mode='same'))
    model.add(Flatten())
    model.add(Dense(1024, activation='tanh'))
    model.add(Dropout(.5))

    additional_info_model = Sequential()
    additional_info_model.add(Dense(192, input_shape=(192,)))

    model_merged = Sequential()
    model_merged.add(Merge([model, additional_info_model], mode='concat'))
    model_merged.add(Dense(1024 + additional_features_len, activation='tanh'))
    model_merged.add(Dropout(.5))
    model_merged.add(Dense(n_classes, activation='softmax'))
    optimizer = Adam(decay=1e-6)
    model_merged.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=optimizer)

    # Plot the model to files
    plot(model, to_file='./model_left.png', show_shapes=True)
    plot(additional_info_model, to_file='./model_right.png', show_shapes=True)
    plot(model_merged, to_file='./model_merged.png', show_shapes=True)

    return model_merged
