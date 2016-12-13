# coding=utf-8
import numpy as np
import tensorflow as tf
import os

from PIL import Image
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array


def crop_to_first(image):
    '''
    Helper function that removes "empty lines"
    :param image: image, uint8 array
    :return: image, uint8 array
    '''
    image = image[~np.all(image == 0, axis=1)]  # rows
    image = image[:, ~np.all(image == 0, axis=0)]  # cols
    return image


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


def distorted_input(img, count):
    '''
    Distorts images
    :param img: Image to distort
    :param count: how many images to return
    :return: array of distorted images
    '''
    assert count > 0
    datagen = ImageDataGenerator(
        rotation_range=180,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.9,
        horizontal_flip=True,
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
        return image[:, random_start:random_start + size]

    elif image.shape[0] > image.shape[1]:  # more rows
        random_start = np.random.randint(0, image.shape[0] - size)
        return image[random_start:random_start + size, :]


def batch_random_crop(image, batch_size):
    '''
    Returns a batch of square crops from /image/
    :param image: numpy array image
    :param batch_size: how many images to return
    :return: array of square images
    '''
    return [y for x in range(batch_size) for y in [__random_crop(image)]]


def load_images(directory):
    '''
    Load all images from a specified directory
    :param directory: directory to load images from
    :return: array of images
    '''
    files = os.listdir(directory)
    path_files = [directory + file for file in files]
    return [Image.open(image) for image in path_files]


def my_normalize(img, min=0., max=255.):
    return np.asarray([[(x-min) / (max-min) for x in x] for x in img])
