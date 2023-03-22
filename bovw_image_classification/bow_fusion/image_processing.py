import os
import cv2 as cv


def open_images_from_dir_of_dir(dir_of_dir):
    images = []
    for directory in os.listdir(dir_of_dir):
        for filename in os.listdir(os.path.join(dir_of_dir, directory)):
            image = cv.imread(os.path.join(dir_of_dir, directory, filename))
            images.append(image)
    return images


def open_images_from_dir(dir):
    images = []
    for filename in os.listdir(dir):
        images.append(cv.imread(os.path.join(dir, filename)))
    return images


def open_image(filepath):
    return cv.imread(filepath)


def convert_images_to(images, type):
    all_images = []
    for image in images:
        all_images.append(cv.normalize(src=image, dst=None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=type))
    return all_images



