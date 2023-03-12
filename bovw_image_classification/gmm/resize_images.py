import cv2 as cv


def get_average_width(images):
    width_sum = 0
    for image in images:
        width_sum += image.shape[1]
    return int(width_sum / len(images))


def resize_to_same_width(images, width):
    resized_images = []
    for image in images:
        dimension = (width, int(image.shape[0] * width / image.shape[1]))
        resized_images.append(cv.resize(image, dimension, interpolation=cv.INTER_AREA))
    return resized_images


def get_average_height(images):
    height_sum = 0
    for image in images:
        height_sum += image.shape[0]
    return int(height_sum / len(images))


def resize_to_same_height(images, height):
    resized_images = []
    for image in images:
        dimension = (int(image.shape[1] * height / image.shape[0]), height)
        resized_images.append(cv.resize(image, dimension, interpolation=cv.INTER_AREA))
    return resized_images


def resize_to(images, height, width):
    resized_images = []
    for image in images:
        dimension = (int(image.shape[1] * height / image.shape[0]), int(image.shape[0] * width / image.shape[1]))
        resized_images.append(cv.resize(image, dimension, interpolation=cv.INTER_AREA))
    return resized_images
