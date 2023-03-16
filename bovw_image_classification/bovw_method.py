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

sea_ocean = os.listdir("../dataset/sea_ocean/without_changes")
other = os.listdir("../dataset/other/without_changes")


def prepare_data(sea_ocean, other, repo):
    class ImageTraining:
        img_array = ""
        index = ""
        label = ""

        def __init__(self, array, path_img, index, label):
            self.array = array
            self.path_img = path_img
            self.index = index
            self.label = label


    dataset = []
    index_img = 0

    for img in sea_ocean:
        dataset.append(
            ImageTraining(
                np.array(cv2.imread("../dataset/sea_ocean/" + repo + "/" + img, cv2.IMREAD_GRAYSCALE)),
                "../dataset/sea_ocean/" + repo + "/" + img,
                index_img,
                1
            )
        )  # 1 because it's sea
        index_img += 1

    for img in other:
        dataset.append(
            ImageTraining(
                np.array(cv2.imread("../dataset/other/" + repo + "/" + img, cv2.IMREAD_GRAYSCALE)),
                "../dataset/other/" + repo + "/" + img,
                index_img,
                -1
            )
        )  # 0 because sea not found
        index_img += 1

    all_label = []
    for i in dataset:
        all_label.append(i.label)
    all_label = np.stack(all_label)
    return all_label, dataset


def extract_feature(dataset, all_label):
    #  Add gray filter on all images
    black_white_img = []
    for img in dataset:
        if len(img.array.shape) > 2:
            black_white_img.append(cv2.cvtColor(img.array, cv2.COLOR_RGB2GRAY))
        else:
            black_white_img.append(img.array)

    # Extract feature (keypoint and descriptor)
    extractor = cv2.xfeatures2D.SURF_create()
    keypoints = []
    descriptors = []

    for img in black_white_img:
        img_kp, img_desc = extractor.detectAndCompute(img, None)  # no mask to add
        keypoints.append(img_kp)
        for i in img_desc:
            for j in i:
                print(type(j))
        descriptors.append(float_desc)


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

    return all_desc, descriptors, all_label

def extract_fuature_FREAK(dataset, all_label):
    color_img = []
    for img in dataset:
        color_img.append()





def create_codebook(all_desc):
    print(all_desc)
    # 2°) Set up codebook with K_means algo for the sample
    cluster_nb = 5
    iters = 1

    codebook, variance = kmeans(all_desc, cluster_nb, iters)
    return codebook, cluster_nb


def generate_visual_words(descriptors, codebook):
    visual_words = []
    for img_desc in descriptors:
        img_visual_words, distance = vq(img_desc, codebook)
        visual_words.append(img_visual_words)
    return visual_words


def save_codebook(codebook):
    return codebook


def calculate_frequency(visual_words, cluster_nb):
    frequency_vectors = []
    for img_visual_words in visual_words:
        img_frequency_vector = np.zeros(cluster_nb)
        for word in img_visual_words:
            img_frequency_vector[word] += 1
        frequency_vectors.append(img_frequency_vector)
    frequency_vectors = np.stack(frequency_vectors)
    return frequency_vectors


all_label, training_img = prepare_data(sea_ocean, other, "without_changes")
all_desc, descriptors, all_label = extract_feature(training_img, all_label)
codebook, cluster_nb = create_codebook(all_desc)
visual_words = generate_visual_words(descriptors, codebook)
frequency_vectors = calculate_frequency(visual_words, cluster_nb)