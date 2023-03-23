import features_extraction as extraction
import clustering_algorithm as clustering
import image_processing as processing

import cv2 as cv


class SiftBow:

    def __init__(self, images):
        features, descriptors = extraction.compute_sift_descriptors(images)
        self.features = features
        self.descriptors = descriptors

        self.clustering_model = None
        self.nb_clusters = None

    def learn_clustering_model(self, nb_clusters):
        self.nb_clusters = nb_clusters
        self.clustering_model = clustering.get_kmeans_model(self.features, nb_clusters)

    def assign_clustering_model(self, clustering_model, nb_clusters):
        self.nb_clusters = nb_clusters
        self.clustering_model = clustering_model


class ColorBow:

    def __init__(self, images, nb_patches):
        images_UINT16 = processing.convert_images_to(images, cv.CV_16U)
        features, descriptors = extraction.compute_color_descriptors(images_UINT16, nb_patches)
        self.features = features
        self.descriptors = descriptors

        self.clustering_model = None
        self.nb_clusters = None

    def learn_clustering_model(self, nb_clusters):
        self.nb_clusters = nb_clusters
        self.clustering_model = clustering.get_kmeans_model(self.features, nb_clusters)

    def assign_clustering_model(self, clustering_model, nb_clusters):
        self.nb_clusters = nb_clusters
        self.clustering_model = clustering_model
