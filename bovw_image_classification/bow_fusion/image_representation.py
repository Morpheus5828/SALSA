import numpy as np

from sklearn.preprocessing import StandardScaler

'''
This file was authored by Chloe BUTTIGIEG.
'''


'''
Computes occurrences histogram for a list of descriptor. Clusters are predicted from a given clustering model.
input = a clustering model from scikit-learn trained,
        a list of descriptor,
        number of clusters.
output = histogram of occurrences.
'''
def get_occurrences_histogram(clustering_model, descriptor, nb_clusters):
    histogram = np.zeros(nb_clusters)
    clusters = clustering_model.predict(descriptor)
    for cluster in clusters:
        histogram[cluster] += 1
    return histogram

# --------------------------------------------------------------------------------- #


'''
Applies normal standardisation to a list of numeric value. The function modify the data itself.
The data provided must be a dictionary with one required field:
- representations: it's a list that contains all images' representation. Each representation is also a list.
Thus, representations is a matrix.
input = data as a dictionary of list.
-- use scikit-learn library
'''
def normalize_representation(data):
    stdslr = StandardScaler().fit(data["representations"])
    data["representations"] = stdslr.transform(data["representations"])

# --------------------------------------------------------------------------------- #

'''
Computes histogram occurrences of given data from a given bag of words.
The data provided must be a dictionary with one required field:
- images: it a list of all images, as numpy array.
The function modify the data itself and add a new field, 'representations'. It's a list that contains all
images' occurrences histogram. Thus, representations is a matrix.
input = data as a dictionary of list,
        a bag of words from bag_of_words.py file.
'''
def compute_bow_representation(train_data, bow):
    train_data["representations"] = []
    for index in range(len(train_data["images"])):
        occurrences_histogram = get_occurrences_histogram(bow.clustering_model, bow.descriptors[index], bow.nb_clusters)
        train_data["representations"].append(occurrences_histogram)


'''
Computes histogram occurrences of given data from two given bags of words. Each occurrences histogram
is computed separately, then concatenate to form a representation.
The data provided must be a dictionary with one required field:
- images: it a list of all images, as numpy array.
The function modify the data itself and add a new field, 'representations'. It's a list that contains all
images' representation. Thus, representations is a matrix.
input = data as a dictionary of list,
        a first bag of words from bag_of_words.py file,
        a second bag of words from bag_of_words.py file.
'''
def compute_bow_fusion_representation(train_data, first_bow, second_bow):
    train_data["representations"] = []

    for index in range(len(train_data["images"])):
        first_occurrences_histogram = get_occurrences_histogram(first_bow.clustering_model,
                                                                first_bow.descriptors[index], first_bow.nb_clusters)
        second_occurrences_histogram = get_occurrences_histogram(second_bow.clustering_model,
                                                                 second_bow.descriptors[index], second_bow.nb_clusters)
        train_data["representations"].append(
            np.concatenate((first_occurrences_histogram, second_occurrences_histogram)))
