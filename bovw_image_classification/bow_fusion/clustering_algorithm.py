from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans

'''
This file was authored by Chloe BUTTIGIEG.
'''


'''
Computes Gaussian Mixture model from scikit-learn library on given data.
input = data to be trained on. It must be a matrix,
        the number of mixtures.
output = a Gaussian Mixture model trained
-- use scikit-learn library
'''
def get_gaussian_mixture_model(data, nb_mixtures):
    gmm = GaussianMixture(n_components=nb_mixtures)
    gmm.fit(data)
    return gmm

# --------------------------------------------------------------------------------- #


'''
Computes K-Means model from scikit-learn library on given data.
input = data to be trained on. It must be a matrix,
        the number of clusters.
output = a K-Means model trained
-- use scikit-learn library
'''
def get_kmeans_model(data, nb_clusters):
    kmeans = KMeans(n_clusters=nb_clusters)
    kmeans.fit(data)
    return kmeans
