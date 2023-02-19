import math
import os

import cv2
import numpy as np
from matplotlib import pyplot as plt

sea_ocean = os.listdir("../dataset/sea_ocean/")
img = sea_ocean[0]  # 838s.jpg

img = cv2.imread("../dataset/sea_ocean/" + img)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
gray = np.float32(gray)


#############################

def get_keypoint(kp):
    return np.asarray([key_point.pt for key_point in kp])


def detect_keypoint(n):
    return cv2.FastFeatureDetector_create(n)


def get_distance(x1, y1, x2, y2):
    return int(math.sqrt(math.pow(x2-x1,2) + math.pow(y2-y1,2)))


def main(img):
    detector = detect_keypoint(50)  # by default
    kp_array = get_keypoint(detector.detect(img, None))
    print(kp_array)


print(get_distance(0, 0, 1, 0))

# img2 = cv2.drawKeypoints(img, kp, None, flags=0)

# cv2.imshow('corne', img2)
# cv2.waitKey(0)
