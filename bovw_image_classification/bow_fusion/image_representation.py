import resize
import clustering_algorithm as clustering
import image_processing as processing
import features_extraction as extraction

import cv2 as cv
import numpy as np
from sklearn.preprocessing import StandardScaler


def get_occurrences_histogram(clustering_model, descriptor, nb_clusters):
    histogram = np.zeros(nb_clusters)
    clusters = clustering_model.predict(descriptor)
    for cluster in clusters:
        histogram[cluster] += 1
    return histogram

# --------------------------------------------------------------------------------- #


def normalize_representation(data):
    stdslr = StandardScaler().fit(data["representations"])
    data["representations"] = stdslr.transform(data["representations"])

# --------------------------------------------------------------------------------- #


def compute_fusion_representation(train_data, first_bow, second_bow):
    train_data["representations"] = []

    for index in range(len(train_data["images"])):
        first_occurrences_histogram = get_occurrences_histogram(first_bow.clustering_model,
                                                                first_bow.descriptors[index], first_bow.nb_clusters)
        second_occurrences_histogram = get_occurrences_histogram(second_bow.clustering_model,
                                                                 second_bow.descriptors[index], second_bow.nb_clusters)
        train_data["representations"].append(
            np.concatenate((first_occurrences_histogram, second_occurrences_histogram)))
