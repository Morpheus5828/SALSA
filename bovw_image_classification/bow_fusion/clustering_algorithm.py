import joblib
import os

from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans


def get_gaussian_mixture_model(data, nb_clusters):
    gmm = GaussianMixture(n_components=nb_clusters)
    gmm.fit(data)
    return gmm


def get_kmeans_model(data, nb_clusters):
    kmeans = KMeans(n_clusters=nb_clusters)
    kmeans.fit(data)
    return kmeans
