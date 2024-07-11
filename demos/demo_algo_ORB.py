"""This module show an example of using ORB algorithm on image
..moduleauthor:: Marius THORRE
"""

import matplotlib.pyplot as plt
import sys, os
import cv2
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '../'))
if project_root not in sys.path:
    sys.path.append(project_root)
import salsa.algorithms.ORB as ORB

if __name__ == "__main__":
    path = os.path.join(project_root, "resources/dataset/other/ouie.jpeg")
    kp, des = ORB.get_kp_descriptors(path, draw_kp=True)