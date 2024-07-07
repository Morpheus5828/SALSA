import cv2
import numpy as np
import os
import cv2
from PIL import Image


def get_bw_image(path):
    bw_image = []
    all_label = []
    label = 0
    for folder in os.listdir(path):
        for image in os.listdir(path + "/" + folder):
            img = np.array(Image.open(path + "/" + folder + "/" + image))
            if len(img.shape) > 2:
                bw_image.append(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
            else:
                bw_image.append(img)
            all_label.append(label)
        label += 1
    return bw_image, all_label
