import os
import cv2 as cv


'''
Transforms pictures file such as png, jpg, jpeg... to a numpy array thank to open-cv imread() function.
input = path of a folder that contains folders. Pictures are extracted from each sub-folder.
output = a list of picture file opened as a numpy array.
-- use open-cv library
'''
def open_images_from_folder_of_folder(folder_of_folder):
    images = []
    for directory in os.listdir(folder_of_folder):
        for filename in os.listdir(os.path.join(folder_of_folder, directory)):
            image = cv.imread(os.path.join(folder_of_folder, directory, filename))
            images.append(image)
    return images


'''
Transforms pictures file from a folder such as png, jpg, jpeg... to a numpy array thank to open-cv imread() function.
input = path of a folder.
output = a list of picture file opened as a numpy array.
-- use open-cv library
'''
def open_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        images.append(cv.imread(os.path.join(folder, filename)))
    return images

'''
Transforms a picture file such as png, jpg, jpeg... to a numpy array thank to open-cv imread() function.
input = path of file.
output = picture file opened as a numpy array.
-- use open-cv library
'''
def open_image(filepath):
    return cv.imread(filepath)

# --------------------------------------------------------------------------------- #

'''
Transforms a list of images into a specific given type. The type must be an type from open-cv library.
input = a list of images. It can be images already opened as a numpy array, or a list of filepath,
        the type to be converted.
output = list of input images converted to given type.
-- use open-cv library
'''
def convert_images_to(images, convertType):
    all_images = []
    for image in images:
        all_images.append(cv.normalize(src=image, dst=None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=convertType))
    return all_images



