import statistic

import numpy as np
import cv2 as cv

from sklearn.feature_extraction.image import extract_patches_2d


'''
Computes SIFT keypoint and descriptor for a given image. The image is converted to grayscale.
input = an image already opened as a numpy array.
output = a list of SIFT descriptor of the given image.
-- use open-cv library
'''
def get_sift_descriptor(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    sift = cv.SIFT_create()
    kp, descriptor = sift.detectAndCompute(gray, None)
    return descriptor


'''
Computes color descriptor for a given image.
input = an image already opened as a numpy array, the number of patches to be extracted.
output = a list of color descriptor of the given image.
-- use scikit-learn library
'''
def get_color_descriptor(image, nb_patches):
    patches = extract_patches_2d(image, (12, 12), max_patches=nb_patches)
    descriptor = []

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
                    components3.append(sum / (3 * 255))

        mean1, stdev1 = statistic.compute_statistics(components1)
        mean2, stdev2 = statistic.compute_statistics(components2)
        mean3, stdev3 = statistic.compute_statistics(components3)

        descriptor.append([mean1, stdev1, mean2, stdev2, mean3, stdev3])
    return descriptor


'''
Computes SIFT descriptors for a given list of images.
input = a list of images already opened as numpy array.
output = a list of all SIFT descriptors computed for all images,
         a list of each image's SIFT descriptors, thus a matrix.
-- use numpy library
'''
def compute_sift_descriptors(images):
    first = get_sift_descriptor(images[0])
    features = first
    descriptors = [first]

    for image in images[1:]:
        descriptor = get_sift_descriptor(image)
        features = np.vstack((features, descriptor))
        descriptors.append(descriptor)

    return features, descriptors


'''
Computes color descriptors for a given list of images.
input = a list of images already opened as numpy array, the number of patches to be extracted.
output = a list of all color descriptors computed for all images,
         a list of each image's color descriptors, thus a matrix.
-- use numpy library
'''
def compute_color_descriptors(images, nb_patches):
    first = get_color_descriptor(images[0], nb_patches)
    features = first
    descriptors = [first]

    for image in images[1:]:
        descriptor = get_color_descriptor(image, nb_patches)
        features = np.vstack((features, descriptor))
        descriptors.append(descriptor)

    return features, descriptors
