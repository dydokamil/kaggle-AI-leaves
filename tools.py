# coding=utf-8
import os

import numpy as np
from PIL import Image
from keras.preprocessing.image import ImageDataGenerator, img_to_array


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


def batch_random_crop_and_distort(image, count):
    '''
    Randomly selects a part of the image and distorts it
    :param image: image to distort
    :param count: how many images to return
    :return: batch of distorted images
    '''

    distorted_array = __distort_input(image, count)
    return np.asarray([__random_crop(distorted) for distorted in distorted_array])


def load_images(directory):
    '''
    Load all images from a specified directory
    :param directory: directory to load images from
    :return: array of images
    '''
    files = os.listdir(directory)
    path_files = [directory + file for file in files]
    return np.asarray([np.asarray(Image.open(image)) for image in path_files])


def my_normalize(img, min=0., max=255.):
    return np.asarray([[(x - min) / (max - min) for x in x] for x in img])
