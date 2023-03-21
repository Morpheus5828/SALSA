import joblib
from joblib import dump, load
import os

from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import MeanShift


def get_gaussian_mixture_model(data, nb_clusters):
    gmm = GaussianMixture(n_components=nb_clusters)
    gmm.fit(data)
    return gmm


def get_kmeans_model(data, nb_clusters):
    kmeans = KMeans(n_clusters=nb_clusters)
    kmeans.fit(data)
    return kmeans


def get_affinity_propagation(data):
    aff = AffinityPropagation()
    aff.fit(data)
    return aff


def get_mean_shift(data):
    mean_shift = MeanShift()
    mean_shift.fit(data)
    return mean_shift


def save_model(clustering_algo, filename, filepath):
    joblib.dump(clustering_algo, os.path.join(filepath, filename))


def load_model(filepath):
    return joblib.load(filepath)
