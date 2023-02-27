import sys

import numpy as np
import cv2
from matplotlib import pyplot as plt

img1 = cv2.imread("za7huw.jpeg")
img2 = cv2.imread("z3tt.png")

sift = cv2.SIFT_create()
kp1, desc1 = sift.detectAndCompute(img1, None)
kp2, desc2 = sift.detectAndCompute(img2, None)

index_paras = dict(algorithm=0, trees=5)
search_params = dict()
flann = cv2.FlannBasedMatcher(index_paras, search_params)

matches = flann.knnMatch(desc1, desc2, k=2)

good_point = []
for m, n in matches:
    if m.distance < 0.2*n.distance:
        good_point.append(m)

print("taille de kp1: ", len(kp1))
print("taille de kp2: ", len(kp2))
print(len(good_point))

result = cv2.drawMatches(img1, kp1, img2, kp2, good_point, None)
cv2.imshow("result", cv2.resize(result, None, fx=0.4, fy=0.4))
cv2.waitKey(0)

