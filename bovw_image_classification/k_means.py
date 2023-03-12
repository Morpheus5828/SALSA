import math
import os
import time

import cv2
import joblib
import numpy as np
from scipy.cluster.vq import kmeans, vq
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
from sklearn.svm import LinearSVC
from numpy.linalg import norm
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

from sift_detection.sift_detection import extract_SIFT_descriptors

sea_ocean = os.listdir("../dataset/sea_ocean")
other = os.listdir("../dataset/other")
all_dataset = os.listdir("../dataset/all")

start = time.time()

# 0°) Create class which represent all image with label

class ImageTraining:
    img_array = ""
    index = ""
    label = ""

    def __init__(self, array, path_img, index, label):
        self.array = array
        self.path_img = path_img
        self.index = index
        self.label = label


training_img = []
index_img = 0

for img in sea_ocean:
    training_img.append(
        ImageTraining(
            np.array(cv2.imread("../dataset/sea_ocean/" + img)),
            "../dataset/sea_ocean/" + img,
            index_img,
            1
        )
    )  # 1 because it's sea
    index_img += 1

for img in other:
    training_img.append(
        ImageTraining(
            np.array(cv2.imread("../dataset/other/" + img)),
            "../dataset/other/" + img,
            index_img,
            0
        )
    )  # 0 because sea not found
    index_img += 1

all_label = []
for i in training_img:
    all_label.append(i.label)
all_label = np.stack(all_label)

# 1°) Extract feature for all dataset


#  Add gray filter on all images
black_white_img = []
for img in training_img:
    if len(img.array.shape) > 2:
        black_white_img.append(cv2.cvtColor(img.array, cv2.COLOR_RGB2GRAY))
    else:
        black_white_img.append(img.array)

# Extract feature (keypoint and descriptor)
extractor = cv2.SIFT_create()

keypoints = []
descriptors = []

for img in black_white_img:
    img_kp, img_desc = extractor.detectAndCompute(img, None)  # no mask to add
    keypoints.append(img_kp)
    descriptors.append(img_desc)

drop_img = []
for index, img_desc in enumerate(descriptors):
    if img_desc is None:  # if no feature in image, we don't need it
        all_label = np.delete(all_label, index)
        drop_img.append(index)


for index in sorted(drop_img, reverse=True):
    del descriptors[index], keypoints[index]

all_desc = []
for img_desc in descriptors:
    for desc in img_desc:
        all_desc.append(desc)


all_desc = np.stack(all_desc)  # shape: (873842, 128)


# 2°) Set up codebook with K_means algo for the sample

k = len(descriptors)
iters = 1
codebook, variance = kmeans(all_desc, k, iters)


# 3°) Vectorisation

visual_words = []
for img_desc in descriptors:
    img_visual_words, distance = vq(img_desc, codebook)
    visual_words.append(img_visual_words)


# 4°) Frequency count

frequency_vectors = []
for img_visual_words in visual_words:
    img_frequency_vector = np.zeros(k)
    for word in img_visual_words:
        img_frequency_vector[word] += 1
    frequency_vectors.append(img_frequency_vector)
frequency_vectors = np.stack(frequency_vectors)
print("frequency_vectors:", frequency_vectors.shape)
print("label:", all_label.shape)

# 5°) Gaussian

X_train, X_test, y_train, y_test = train_test_split(frequency_vectors, all_label, test_size=0.2)
gnb = GaussianNB()
y_pred = gnb.fit(X_train, y_train).predict(X_test)


# 6°) Score
gaussian_score = accuracy_score(y_test, y_pred)
print(gaussian_score)
end = time.time()
print("\n Execution time:", end - start)


