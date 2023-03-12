import os
import cv2 as cv


def get_all_filepath_from_dir_of_dir(dir_of_dir):
    all_filepath = []
    for directory in os.listdir(dir_of_dir):
        for filename in os.listdir(os.path.join(dir_of_dir, directory)):
            all_filepath.append(os.path.join(dir_of_dir, directory, filename))
    return all_filepath


def get_all_filepath_from_dir(dir):
    all_filepath = []
    for filename in os.listdir(dir):
        all_filepath.append(os.path.join(dir, filename))
    return all_filepath


def open_images_from_dir_of_dir(dir_of_dir):
    images = []
    for directory in os.listdir(dir_of_dir):
        for filename in os.listdir(os.path.join(dir_of_dir, directory)):
            images.append(cv.imread(os.path.join(dir_of_dir, directory, filename)))
    return images


def open_images_from_dir(dir):
    images = []
    for filename in os.listdir(dir):
        images.append(cv.imread(os.path.join(dir, filename)))
    return images

