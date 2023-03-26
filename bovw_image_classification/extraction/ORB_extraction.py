import math
import os
import time

import cv2
import numpy as np
from scipy.cluster.vq import kmeans, vq
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sift_detection.sift_detection import extract_SIFT_descriptors

import os

#by Marius THORRE

sea_ocean = os.listdir("../../dataset/sea_ocean/without_changes")
other = os.listdir("../../dataset/other/without_changes")


def extract_features(sea_ocean, other, repo):
    dataset_with_grey_filter = []
    dataset_no_filter = []
    dataset_labels = []

    for img in sea_ocean:
        dataset_with_grey_filter.append(cv2.imread("../dataset/sea_ocean/" + repo + "/" + img, cv2.IMREAD_GRAYSCALE))
        dataset_no_filter.append(cv2.imread("../dataset/sea_ocean/" + repo + "/" + img))
        dataset_labels.append(1)
    for img in other:
        dataset_with_grey_filter.append(cv2.imread("../dataset/other/" + repo + "/" + img, cv2.IMREAD_GRAYSCALE))
        dataset_no_filter.append(cv2.imread("../dataset/other/" + repo + "/" + img))
        dataset_labels.append(-1)

    ORB = cv2.ORB_create()

    keypoints = []
    descriptors = []

    # extraction of descriptors
    for index in range(len(dataset_with_grey_filter)):
        gray_img = dataset_with_grey_filter[index]
        normal_img = dataset_no_filter[index]

        kp, desc = ORB.detectAndCompute(gray_img, None)
        img_1 = cv2.drawKeypoints(gray_img, kp, normal_img, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv2.imwrite("ORB_kp_drawing/" + str(index) + ".jpg", img_1)

        keypoints.append(kp)
        descriptors.append(desc)

    return descriptors, dataset_labels


def K_means(descriptors):
    all_descriptors = []
    for img_desc in descriptors:
        for desc in img_desc:
            all_descriptors.append(desc.astype('float'))

    all_descriptors = np.stack(all_descriptors)
    cluster_nb = int(math.log(414))
    iters = 1

    codebook, variance = kmeans(all_descriptors, cluster_nb, iters)
    return codebook, cluster_nb


def calculate_frequencies(descriptors, codebook, cluster_nb):
    visual_words = []
    for img_desc in descriptors:
        img_visual_words, distance = vq(img_desc, codebook)
        visual_words.append(img_visual_words)
    frequency_vectors = []
    for img_visual_words in visual_words:
        img_frequency_vector = np.zeros(cluster_nb)
        for word in img_visual_words:
            img_frequency_vector[word] += 1
        frequency_vectors.append(img_frequency_vector)
    frequency_vectors = np.stack(frequency_vectors)
    return frequency_vectors



