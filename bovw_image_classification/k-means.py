import math
import os

import cv2
import joblib
import numpy as np
from scipy.cluster.vq import kmeans, vq
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
from sklearn.svm import LinearSVC

from sift_detection.sift_detection import extract_SIFT_descriptors

sea_ocean = os.listdir("../dataset/sea_ocean")

image_training = []
for image in sea_ocean:
    image_training.append(np.array(cv2.imread("../dataset/sea_ocean/" + image)))

# add grayscale
image_gray = []
for image in image_training:
    if len(image.shape) > 2:
        image_gray.append(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY))
    else:
        image_gray.append(image)

# extract features
extractor = cv2.SIFT_create()

keypoint = []
descriptor = []

for i in image_gray:
    kp, desc = extractor.detectAndCompute(i, None)
    keypoint.append(kp)
    descriptor.append(desc)

to_drop = []
for i, img_desc in enumerate(descriptor):
    if img_desc is not None:
        to_drop.append(i)

for i in sorted(to_drop, reverse=True):
    del descriptor[i], keypoint[i]


descriptor = np.stack(descriptor)  # add all vector in a single array

k = int(math.log(len(sea_ocean)))
iters = 1
codebook, variance = kmeans(descriptor, 200, iters)

joblib.dump((k,codebook), "bovw-codebook.pkl", compress=3)

# vector

visual_word = []
for img in descriptor:
    img_visual_words, distance = vq(img, codebook)
    visual_word.append(img_visual_words)






















