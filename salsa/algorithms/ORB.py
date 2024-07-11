"""This module contains script to extract features using ORB OpenCv algorithms
https://docs.opencv.org/3.4/d1/d89/tutorial_py_orb.html
..moduleauthor:: Marius Thorre
"""

import numpy as np
import cv2 as cv
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
        orb = cv.ORB_create()
        kp, des = orb.detectAndCompute(filter,None)
        if draw_kp:
            img = cv.drawKeypoints(filter, kp, img)
            path = os.path.splitext(os.path.basename(path))[0]
            cv.imwrite(path + f"_ORB_kp_{filter_color}.jpg", img)
        return kp, des
    else:
        print("Path not exist")

