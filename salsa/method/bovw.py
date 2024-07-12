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
from salsa.tools.image_info import img_folder_sizes_infos
from sklearn.metrics import accuracy_score
import salsa.algorithms.SIFT as SIFT
import salsa.algorithms.BRISK as BRISK
from salsa.tools.image_tools import get_codebook

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '../../'))
if project_root not in sys.path:
    sys.path.append(project_root)


def _stack_descriptors(train_desc) -> np.ndarray:
    all_descriptors = []
    for img_descriptors in train_desc:
        for descriptor in img_descriptors:
            all_descriptors.append(descriptor)
    return np.stack(all_descriptors)


def _get_visual_words(data_desc: np.ndarray, codebook: np.ndarray) -> np.ndarray:
    visual_words = []
    for img_descriptors in data_desc:
        img_visual_words, distance = vq(img_descriptors, codebook)
        visual_words.append(img_visual_words)
    return visual_words


def _get_frequencies(visual_words: np.ndarray, k_cluster: int, dataset_size: int) -> np.ndarray:
    frequency_vectors = []
    for img_visual_words in visual_words:
        img_frequency_vector = np.zeros(k_cluster)
        for word in img_visual_words:
            img_frequency_vector[word] += 1
        frequency_vectors.append(img_frequency_vector)
    frequency_vectors = np.stack(frequency_vectors)
    df = np.sum(frequency_vectors > 0, axis=0)
    idf = np.log(dataset_size / df)
    return frequency_vectors * idf


def Bag_Of_Visual_Words(
        X_data: np.ndarray,
        y_data: np.ndarray,
        extract_method: str = "SIFT",
        k_cluster: int = 30
) -> tuple:
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.30)
    data_desc = None
    train_desc = None
    if extract_method == "SIFT":
        _, data_desc, y_data = SIFT.extract_feature(X_data, y_data)
        _, train_desc, y_train = SIFT.extract_feature(X_train, y_train)
    elif extract_method == "BRISK":
        _, data_desc, y_data = BRISK.extract_feature(X_data, y_data)
        _, train_desc, y_train = BRISK.extract_feature(X_train, y_train)
    else:
        print("Error: Extract method not recognized")

    all_descriptors = _stack_descriptors(train_desc)

    codebook = get_codebook(
        training_data=all_descriptors,
        k_cluster=k_cluster
    )

    visual_words = _get_visual_words(
        data_desc=data_desc,
        codebook=codebook
    )
    frequency_vectors = _get_frequencies(
        visual_words=visual_words,
        k_cluster=k_cluster,
        dataset_size=len(X_data)
    )
    return frequency_vectors, y_data
