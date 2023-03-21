import numpy as np
import cv2 as cv
from sklearn.feature_extraction.image import extract_patches_2d
import statistics
import os


def get_sift_descriptor(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    sift = cv.SIFT_create()
    kp, des = sift.detectAndCompute(gray, None)
    return des


def get_sift_descriptor_from(patches):
    all = []
    sift = cv.SIFT_create()
    for patch in patches:
        kp, des = sift.detectAndCompute(patch, None)
        if des is not None:
            for sub_des in des:
                all.append(sub_des)
    return all


def get_color_information(patches):
    descriptors = []
    for patch in patches:
        components1 = []
        components2 = []
        components3 = []

        for row in patch:
            for pixel in row:
                sum = pixel[0] + pixel[1] + pixel[2]
                if sum == 0:
                    components1.append(0)
                    components2.append(0)
                    components3.append(0)
                else:
                    components1.append(pixel[0] / sum)
                    components2.append(pixel[1] / sum)
                    components3.append(sum / (3*255))

        mean1 = statistics.mean(components1)
        mean2 = statistics.mean(components2)
        mean3 = statistics.mean(components3)
        descriptors.append([mean1, statistics.stdev(components1, xbar=mean1), mean2, statistics.stdev(components1, xbar=mean2), mean3, statistics.stdev(components1, xbar=mean3)])
    return descriptors


def get_orb_descriptor(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    orb = cv.ORB_create()
    kp, des = orb.detectAndCompute(gray, None)
    return des


def get_surf_descriptor(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    surf = cv.SURF_create()
    kp, des = surf.detectAndCompute(gray, None)
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

def compute_patches(image, nb_patches):
    return extract_patches_2d(image, (12, 12), max_patches=nb_patches)

