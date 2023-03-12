import numpy as np
import cv2 as cv
import os


def get_sift_descriptor(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    sift = cv.SIFT_create()
    kp, des = sift.detectAndCompute(gray, None)
    return des


def compute(images, descriptor_function):
    first = descriptor_function(images[0])
    features = first
    descriptors = [first]

    for image in images[1:]:
        descriptor = descriptor_function(image)
        features = np.vstack((features, descriptor))
        descriptors.append(descriptor)
    return features, descriptors


def save_features(filename, filepath, data):
    file = open(os.path.join(filepath, filename), 'w')
    for sub_data in data:
        for element in sub_data:
            file.write(str(element) + " ")
        file.write("\n")
    file.close()


def read_features(filepath, type):
    file = open(filepath, 'r')
    data = []
    for line in file:
        data.append([type(x) for x in line.split()])
    file.close()
    return data


def save_descriptors(filename, filepath, data):
    file = open(os.path.join(filepath, filename), 'w')
    for matrix in data:
        for elements in matrix:
            for element in elements:
                file.write(str(element) + " ")
            file.write("\n")
        file.write("\n")
    file.close()


def read_descriptors(filepath, type):
    file = open(filepath, 'r')
    data = []
    matrix = []
    for line in file:
        if line == "\n":
            data.append(matrix)
            matrix = []
        else:
            matrix.append([type(x) for x in line.split()])
    file.close()
    return data
