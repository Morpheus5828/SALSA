"""This module contains Bag-of-Visual-Words, find in this paper:
Karim, A. A. A., & Sameer, R. A. (2018).
Image classification using bag of visual words (bovw).
Al-Nahrain Journal of Science, 21(4), 76-82.

..moduleauthor:: Marius THORRE
"""

import sys, os
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from scipy.cluster.vq import kmeans
from scipy.cluster.vq import vq
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from salsa.tools.image_info import img_folder_sizes_infos, data_augmentation
from sklearn.metrics import accuracy_score
import salsa.algorithms.SIFT as SIFT
from salsa.tools.image_tools import get_codebook

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '../../'))
if project_root not in sys.path:
    sys.path.append(project_root)


def Bag_Of_Visual_Words(Xdata: np.ndarray, ydata: np.ndarray, k_cluster: int = 30):
    X_train, X_test, y_train, y_test = train_test_split(Xdata, ydata, test_size=0.30)
    _, data_desc = SIFT.extract_feature(Xdata)
    _, train_desc = SIFT.extract_feature(X_train)

    all_descriptors = []
    for img_descriptors in train_desc:
        for descriptor in img_descriptors:
            all_descriptors.append(descriptor)
    all_descriptors = np.stack(all_descriptors)

    codebook = get_codebook(training_data=all_descriptors, k_cluster=k_cluster)

    visual_words = []
    for img_descriptors in data_desc:
        img_visual_words, distance = vq(img_descriptors, codebook)
        visual_words.append(img_visual_words)

    frequency_vectors = []
    for img_visual_words in visual_words:
        img_frequency_vector = np.zeros(k_cluster)
        for word in img_visual_words:
            img_frequency_vector[word] += 1
        frequency_vectors.append(img_frequency_vector)
    frequency_vectors = np.stack(frequency_vectors)

    df = np.sum(frequency_vectors > 0, axis=0)
    idf = np.log(len(Xdata) / df)
    tfidf = frequency_vectors * idf
    return tfidf










