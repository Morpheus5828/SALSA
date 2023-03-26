import features_extraction as extraction
import clustering_algorithm as clustering
import image_processing as processing

import cv2 as cv


'''
This class represents a Bag of Words that use the SIFT algorithm descriptor.
'''
class SiftBow:

    '''
    Initiates a bag of word and compute SIFT descriptors.
    output = a SiftBoW
    '''
    def __init__(self, images):
        features, descriptors = extraction.compute_sift_descriptors(images)
        self.features = features
        self.descriptors = descriptors

        self.clustering_model = None
        self.nb_clusters = None

    '''
    Computes a K-Means clustering model on the descriptors previously computed.
    input = the number of clusters.
    '''
    def learn_clustering_model(self, nb_clusters):
        self.nb_clusters = nb_clusters
        self.clustering_model = clustering.get_kmeans_model(self.features, nb_clusters)

    '''
    Assign a existing clustering trained model to this BoW.
    input = the clustering model to assign, the number of clusters.
    '''
    def assign_clustering_model(self, clustering_model, nb_clusters):
        self.nb_clusters = nb_clusters
        self.clustering_model = clustering_model


'''
This class represents a Bag of Words that use a color descriptor.
'''
class ColorBow:

    '''
    Initiates a bag of word and compute color descriptors.
    output = a ColorBoW
    '''
    def __init__(self, images, nb_patches):
        images_UINT16 = processing.convert_images_to(images, cv.CV_16U)
        features, descriptors = extraction.compute_color_descriptors(images_UINT16, nb_patches)
        self.features = features
        self.descriptors = descriptors

        self.clustering_model = None
        self.nb_clusters = None

    '''
    Computes a K-Means clustering model on the descriptors previously computed.
    input = the number of clusters.
    '''
    def learn_clustering_model(self, nb_clusters):
        self.nb_clusters = nb_clusters
        self.clustering_model = clustering.get_kmeans_model(self.features, nb_clusters)

    '''
    Assign a existing clustering trained model to this BoW.
    input = the clustering model to assign, the number of clusters.
    '''
    def assign_clustering_model(self, clustering_model, nb_clusters):
        self.nb_clusters = nb_clusters
        self.clustering_model = clustering_model
