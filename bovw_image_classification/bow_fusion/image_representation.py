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


def compute_fusion_representation(train_data, nb_clusters):
    average_height = resize.get_average_height(train_data["images"])
    average_width = resize.get_average_width(train_data["images"])

    resized_images = resize.resize_to(train_data["images"], average_height, average_width)
    resized_images_UINT16 = processing.convert_images_to(resized_images, cv.CV_16U)

    color_features, color_descriptors = extraction.compute_color_descriptors(resized_images_UINT16, 500)
    sift_features, sift_descriptors = extraction.compute_sift_descriptors(resized_images)

    color_clustering = clustering.get_kmeans_model(color_features, nb_clusters)
    sift_clustering = clustering.get_kmeans_model(sift_features, nb_clusters)

    train_data["representations"] = []
    for index in range(len(train_data["images"])):
        color_occurrences_histogram = get_occurrences_histogram(color_clustering, color_descriptors[index], nb_clusters)
        sift_occurrences_histogram = get_occurrences_histogram(sift_clustering, sift_descriptors[index], nb_clusters)
        train_data["representations"].append(np.concatenate((color_occurrences_histogram, sift_occurrences_histogram)))

    stdslr = StandardScaler().fit(train_data["representations"])
    train_data["representations"] = stdslr.transform(train_data["representations"])
