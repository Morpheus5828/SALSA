"""This module contains script to extract features using BRISK OpenCv algorithms
https://docs.opencv.org/4.x/de/dbf/classcv_1_1BRISK.html
..moduleauthor:: Marius Thorre
"""

import numpy as np
import cv2 as cv
from tqdm import tqdm
import sys, os

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '../../'))
if project_root not in sys.path:
    sys.path.append(project_root)


def get_kp_descriptors(
        path: str,
        draw_kp: bool = False,
        filter_color: int = cv.COLOR_RGB2GRAY
) -> tuple:
    if os.path.exists(path):
        img = cv.imread(path)
        filter = cv.cvtColor(img, filter_color)
        brisk = cv.BRISK_create()
        kp, des = brisk.detectAndCompute(filter, None)
        if draw_kp:
            img = cv.drawKeypoints(filter, kp, img)
            path = os.path.splitext(os.path.basename(path))[0]
            cv.imwrite(path + f"_BRISK_kp_{filter_color}.jpg", img)
        return kp, des
    else:
        print("Path not exist")


def extract_feature(images: np.ndarray) -> tuple:
    keypoints = []
    descriptors = []
    brisk = cv.BRISK_create()
    for img_index in tqdm(range(images.shape[0])):
        img = images[img_index]
        if np.any(img) != 0:
            kp, desc = brisk.detectAndCompute(img, None)
            if desc is not None:
                keypoints.append(kp)
                descriptors.append(desc)
    return keypoints, descriptors


def _process_descriptor(desc):
    descriptors = np.array(desc[0])
    for descriptor in desc[1:]:
        descriptors = np.vstack((descriptors, descriptor))

    return descriptors
