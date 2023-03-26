import cv2 as cv

'''
This file was authored by Chloe BUTTIGIEG.
'''

'''
Finds the minimum width of a list of images. Each image is a numpy array.
Images can be previously opened with the function imread() of open-cv library.
input = a list of images. 
output = minimum width.
'''
def get_min_width(images):
    min_width = 0
    for image in images:
        if min_width < image.shape[1]:
            min_width = image.shape[1]
    return min_width

'''
Computes average width of a list of images. Each image is a numpy array.
Images can be previously opened with the function imread() of open-cv library.
input = a list of images. 
output = average width.
'''
def get_average_width(images):
    width_sum = 0
    for image in images:
        width_sum += image.shape[1]
    return int(width_sum / len(images))


# --------------------------------------------------------------------------------- #

'''
Finds the minimum height of a list of images. Each image is a numpy array.
Images can be previously opened with the function imread() of open-cv library.
input = a list of images. 
output = minimum height.
'''
def get_min_height(images):
    min_height = 0
    for image in images:
        if min_height < image.shape[0]:
            min_height = image.shape[0]
    return min_height

'''
Computes average height of a list of images. Each image is a numpy array.
Images can be previously opened with the function imread() of open-cv library.
input = a list of images. 
output = average height.
'''
def get_average_height(images):
    height_sum = 0
    for image in images:
        height_sum += image.shape[0]
    return int(height_sum / len(images))


# --------------------------------------------------------------------------------- #

'''
Resizes a list of image to a given width. Images are not distorted. Each image is a numpy array.
Images can be previously opened with the function imread() of open-cv library.
input = a list of images, the width. 
output = the input list of images resized.
-- use open-cv library
'''
def resize_to_same_width(images, width):
    resized_images = []
    for image in images:
        dimension = (width, int(image.shape[0] * width / image.shape[1]))
        resized_images.append(cv.resize(image, dimension, interpolation=cv.INTER_AREA))
    return resized_images

'''
Resizes a list of image to a given height. Images are not distorted. Each image is a numpy array.
Images can be previously opened with the function imread() of open-cv library.
input = a list of images, the height. 
output = the input list of images resized.
-- use open-cv library
'''
def resize_to_same_height(images, height):
    resized_images = []
    for image in images:
        dimension = (int(image.shape[1] * height / image.shape[0]), height)
        resized_images.append(cv.resize(image, dimension, interpolation=cv.INTER_AREA))
    return resized_images

'''
Resizes a list of image to given width and height. Images can be distorted. Each image is a numpy array.
Images can be previously opened with the function imread() of open-cv library.
input = a list of images, the width, the height. 
output = the input list of images resized.
-- use open-cv library
'''
def resize_to(images, height, width):
    resized_images = []
    for image in images:
        dimension = (int(image.shape[1] * height / image.shape[0]), int(image.shape[0] * width / image.shape[1]))
        resized_images.append(cv.resize(image, dimension, interpolation=cv.INTER_AREA))
    return resized_images
