# coding=utf-8
import os
import random

import numpy as np
from PIL import Image
from keras.engine import Merge
from keras.layers import Convolution2D, MaxPooling2D, BatchNormalization, Flatten, Dense, Dropout
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator, img_to_array

N_CLASSES = 99
IMAGE_RESOLUTION = (224, 224)
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
    path_files = [directory + file for file in files]
    path_files = [path for path in path_files if int(path.split('/')[-1].split('.')[0]) in ids]
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


def get_model(img_res, additional_features_len, n_classes):
    '''
    Creates and returns a pre-trained keras model
    :param img_res: 2D tuple of x and y resolution
    :param additional_features_len: length of additional features (columns)
    :param n_classes: number of classes
    :return: pre-trained keras model
    '''
    model = Sequential()
    model.add(Convolution2D(96, 11, 11, subsample=(4, 4), activation='relu',
                            input_shape=(img_res[0], img_res[1]) + (1,)))
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

    additional_info_model = Sequential()
    additional_info_model.add(Dense(192, input_shape=(192,)))

    model_merged = Sequential()
    model_merged.add(Merge([model, additional_info_model], mode='concat'))
    model_merged.add(Dense(4096 + additional_features_len, activation='tanh'))
    model_merged.add(Dropout(.5))
    model_merged.add(Dense(n_classes, activation='softmax'))
    model_merged.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

    return model_merged
