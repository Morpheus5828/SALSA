import joblib

import file_management as management
import resize_images as resize
import features_extraction as extract
import clustering_algorithm as clustering
import model
import model2

import numpy as np
import cv2 as cv

from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import StandardScaler


# Open train images
all_images = management.open_images_from_dir_of_dir("data")


all_filepath = management.get_all_filepath_from_dir_of_dir("data")

# Resize images
average_height = resize.get_average_height(all_images)
average_width = resize.get_average_width(all_images)

all_images = resize.resize_to(all_images, average_height, average_width)

all_images_UINT16 = management.convert_images_to(all_images, cv.CV_16U)

all_features_sift, all_descriptors_sift = extract.compute(all_images, extract.get_sift_descriptor)

average_size = 0
for descriptor in all_descriptors_sift:
    average_size += len(descriptor)
average_size = average_size / len(all_descriptors_sift)


# Extract patches
all_patches = []
for image in all_images_UINT16:
    patches = extract.compute_patches(image, 500)
    all_patches.append(patches)

all_descriptors = []
count = 0
for patches in all_patches:
    descriptor = extract.get_color_information(patches)
    all_descriptors.append(descriptor)
    count += 1

all_features = all_descriptors[0]
for descriptor in all_descriptors:
    all_features = np.vstack((all_features, descriptor))


'''nb_clusters = 20
clustering_model_mm = clustering.get_kmeans_model(all_features, nb_clusters)

all_words = []
for descriptor in all_descriptors:
    all_words.append(clustering_model_mm.predict(descriptor))

occurrence_all_images = [0 for x in range(nb_clusters)]
for words in all_words:
    for cluster in range(nb_clusters):
        if cluster in words:
            occurrence_all_images[cluster] += 1


clustering_model_sm = clustering.get_kmeans_model(all_features_sift, nb_clusters)

train_data = model.load_transform_label_train_data3("data", all_descriptors, all_descriptors_sift, clustering_model_mm, clustering_model_sm, nb_clusters)

# Normalize
stdslr = StandardScaler().fit(train_data["representations"])
train_data["representations"] = stdslr.transform(train_data["representations"])

score = model.estimate_model_score_average(50, train_data, LinearSVC(penalty='l2'), 0.2)
print(score)'''

def compute_score(filepath, min_cluster, max_cluster):
    file = open(filepath, "a")
    file.write("\n")
    for nb_clusters in range(min_cluster, max_cluster):
        print(nb_clusters)
        clustering_model = clustering.get_kmeans_model(all_features, nb_clusters)

        all_words = []
        for descriptor in all_descriptors:
            all_words.append(clustering_model.predict(descriptor))

        occurrence_all_images = [0 for x in range(nb_clusters)]
        for words in all_words:
            for cluster in range(nb_clusters):
                if cluster in words:
                    occurrence_all_images[cluster] += 1

        train_data = model.load_transform_label_train_data("data", all_descriptors, clustering_model, nb_clusters,
                                                           occurrence_all_images, len(all_images))

        # Normalize
        stdslr = StandardScaler().fit(train_data["representations"])
        train_data["representations"] = stdslr.transform(train_data["representations"])

        # Train model
        salsa = LinearSVC(penalty='l2')
        score = model.estimate_model_score_average(50, train_data, salsa, 0.2)
        file.write(str(nb_clusters) + " " + str(score) + "\n")
    file.close()

def compute_score_both(filepath, min_cluster, max_cluster):
    file = open(filepath, "a")
    file.write("\n")
    for nb_clusters in range(min_cluster, max_cluster):
        print(nb_clusters)
        clustering_model_mm = clustering.get_kmeans_model(all_features, nb_clusters)

        all_words = []
        for descriptor in all_descriptors:
            all_words.append(clustering_model_mm.predict(descriptor))

        clustering_model_sm = clustering.get_kmeans_model(all_features_sift, nb_clusters)



        train_data = model.load_transform_label_train_data3("data", all_descriptors, all_descriptors_sift,
                                                            clustering_model_mm, clustering_model_sm, nb_clusters)

        # Normalize
        stdslr = StandardScaler().fit(train_data["representations"])
        train_data["representations"] = stdslr.transform(train_data["representations"])

        # Train model
        salsa = LinearSVC(penalty='l2')
        score = model.estimate_model_score_average(50, train_data, salsa, 0.2)
        file.write(str(nb_clusters) + " " + str(score) + "\n")
    file.close()

def compute_score_both_okapi(filepath, min_cluster, max_cluster):
    file = open(filepath, "a")
    file.write("\n")
    for nb_clusters in range(min_cluster, max_cluster):
        print(nb_clusters)
        clustering_model_mm = clustering.get_kmeans_model(all_features, nb_clusters)

        all_words = []
        for descriptor in all_descriptors:
            all_words.append(clustering_model_mm.predict(descriptor))

        clustering_model_sm = clustering.get_kmeans_model(all_features_sift, nb_clusters)

        occurrence_all_images = [0 for x in range(nb_clusters)]
        for words in all_words:
            for cluster in range(nb_clusters):
                if cluster in words:
                    occurrence_all_images[cluster] += 1

        all_words_sift = []
        for descriptor in all_descriptors_sift:
            all_words_sift.append(clustering_model_sm.predict(descriptor))

        occurrence_all_images_sift = [0 for x in range(nb_clusters)]
        for words in all_words_sift:
            for cluster in range(nb_clusters):
                if cluster in words:
                    occurrence_all_images_sift[cluster] += 1

        train_data = model.load_transform_label_train_data4("data", all_descriptors, all_descriptors_sift,
                                                            clustering_model_mm, clustering_model_sm, nb_clusters, occurrence_all_images, occurrence_all_images_sift, len(all_images), average_size)

        # Normalize
        stdslr = StandardScaler().fit(train_data["representations"])
        train_data["representations"] = stdslr.transform(train_data["representations"])

        # Train model
        salsa = LinearSVC(penalty='l2')
        score = model.estimate_model_score_average(50, train_data, salsa, 0.2)
        file.write(str(nb_clusters) + " " + str(score) + "\n")
    file.close()

compute_score_both_okapi("result/result-both.txt", 18, 19)



