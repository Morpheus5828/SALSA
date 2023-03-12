import joblib

import file_management as management
import resize_images as resize
import features_extraction as extract
import clustering_algorithm as clustering
import model

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

'''# Open images
all_images = management.open_images_from_dir_of_dir("Data")
images = management.open_images_from_dir("Data/Mer")


# Resize images
average_height = resize.get_average_height(all_images)
images = resize.resize_to_same_height(images, average_height)


# Save features
features, descriptors = extract.compute(images, extract.get_sift_descriptor)

extract.save_features("resized_h-sift-sea-features.txt", "save/features/sift/sea/features", features)
extract.save_descriptors("resized_h-sift-sea-descriptors.txt", "save/features/sift/sea/descriptors", descriptors)'''

'''# Load features
features = extract.read_features("save/features/sift/sea/features/resized_h-sift-sea-features.txt", float)
descriptors = extract.read_descriptors("save/features/sift/sea/descriptors/resized_h-sift-sea-descriptors.txt", float)
print(int(np.log(len(descriptors))))

# Save clustering algorithm
clustering_model = clustering.get_gaussian_mixture_model(features, int(np.log(len(descriptors))))
clustering.save_model(clustering_model, "resized_h-gm-sift-sea-5.joblib", "save/clustering/gaussian-mixture/sift")'''


'''# Load clustering algorithm
clustering_model = clustering.load_model("save/clustering/gaussian-mixture/sift/resized_h-gm-sift-sea-5.joblib")
nb_clusters = 5


# Compute histogram
all_descriptors = extract.read_descriptors("save/features/sift/all/descriptors/resized_h-sift-all-descriptors.txt", float)
data = model.load_transform_label_train_data("Data", all_descriptors, clustering_model, nb_clusters)'''

'''# Normalize
stdslr = StandardScaler().fit(data["representations"])
data["representations"] = stdslr.transform(data["representations"])'''


'''# Create model
file = open("result.txt", 'w')

file.write("GaussianNB : " + str(model.estimate_model_score_average(50, data, GaussianNB(), 0.2)) + "\n")
file.write("LinearSVC : " + str(model.estimate_model_score_average(50, data, LinearSVC(), 0.2)) + "\n")
file.write("Perceptron : " + str(model.estimate_model_score_average(50, data, Perceptron(), 0.2)) + "\n")
file.write("Perceptron multicouche : " + str(model.estimate_model_score_average(50, data, MLPClassifier(), 0.2)) + "\n")
file.write("Logistic Regression : " + str(model.estimate_model_score_average(50, data, LogisticRegression(), 0.2)) + "\n")
file.write("K Neighbors : " + str(model.estimate_model_score_average(50, data, KNeighborsClassifier(), 0.2)) + "\n")
file.write("Bagging arbre : " + str(model.estimate_model_score_average(50, data, BaggingClassifier(), 0.2)) + "\n")
file.write("Bagging perceptron : " + str(model.estimate_model_score_average(50, data, BaggingClassifier(estimator=Perceptron()), 0.2)) + "\n")
file.write("Bagging perceptron multicouche : " + str(model.estimate_model_score_average(50, data, BaggingClassifier(estimator=MLPClassifier()), 0.2)) + "\n")
file.write("Random Forest : " + str(model.estimate_model_score_average(50, data, RandomForestClassifier(), 0.2)))

file.close()'''