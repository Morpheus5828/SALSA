import threading

import os
import cv2
from sklearn.linear_model import Perceptron
from sklearn.ensemble import BaggingClassifier
from sklearn.svm import SVC
import numpy as np
from scipy.cluster.vq import kmeans, vq
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from bovw_image_classification.extraction import BRISK_extraction, SIFT_extraction
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sift_detection.sift_detection import extract_SIFT_descriptors


sea_ocean = os.listdir("../dataset/sea_ocean/without_changes")
other = os.listdir("../dataset/other/without_changes")

sea_ocean_average = os.listdir("../dataset/sea_ocean/average_resize")
other_average = os.listdir("../dataset/other/average_resize")

'''
By Marius THORRE
This file is used for get classifier score 
Run evaluate() function and then getScore()
'''

classifier_result = open("classifier_result.txt.txt", "w")

def gaussian_img_not_resize(frequency_vectors, all_label):
    X_train, X_test, y_train, y_test = train_test_split(frequency_vectors, all_label, test_size=0.2)
    gnb = GaussianNB()
    y_pred = gnb.fit(X_train, y_train).predict(X_test)
    score = accuracy_score(y_pred, y_test)
    answer = "without_changes g " + str(score)
    classifier_result.write(answer)
    print(answer)


def gaussian_with_average_resize(frequency_vectors, all_label):
    X_train, X_test, y_train, y_test = train_test_split(frequency_vectors, all_label, test_size=0.2)
    gnb = GaussianNB()
    y_pred = gnb.fit(X_train, y_train).predict(X_test)
    score = accuracy_score(y_pred, y_test)
    answer = "average_resize g " + str(score)
    classifier_result.write(answer)
    print("average_resize g ", score)


def k_nn_img_not_resize(frequency_vectors, all_label):
    X_train, X_test, y_train, y_test = train_test_split(frequency_vectors, all_label, test_size=0.2)
    neigh = KNeighborsClassifier(n_neighbors=3)
    y_pred = neigh.fit(X_train, y_train).predict(X_test)
    score = accuracy_score(y_pred, y_test)
    answer = "without_changes knn " + str(score)
    classifier_result.write(answer)
    print(answer)


def k_nn_with_average_resize(frequency_vectors, all_label):
    X_train, X_test, y_train, y_test = train_test_split(frequency_vectors, all_label, test_size=0.2)
    neigh = KNeighborsClassifier(n_neighbors=3)
    y_pred = neigh.fit(X_train, y_train).predict(X_test)
    score = accuracy_score(y_pred, y_test)
    answer = "average_resize knn" + str(score)
    classifier_result.write(answer)
    print(answer)


def perceptron_img_not_resize(frequency_vectors, all_label):
    X_train, X_test, y_train, y_test = train_test_split(frequency_vectors, all_label, test_size=0.2)
    pct = Perceptron(tol=1e-3, random_state=0)
    y_pred = pct.fit(X_train, y_train).predict(X_test)
    score = accuracy_score(y_pred, y_test)
    answer = "without_changes p" + str(score)
    classifier_result.write(answer)
    print(answer)


def perceptron_average_resize(frequency_vectors, all_label):
    X_train, X_test, y_train, y_test = train_test_split(frequency_vectors, all_label, test_size=0.2)
    pct = Perceptron(tol=1e-3, random_state=0)
    y_pred = pct.fit(X_train, y_train).predict(X_test)
    score = accuracy_score(y_pred, y_test)
    answer = "average_resize p" + str(score)
    classifier_result.write(answer)
    print(answer)


def bagging_img_not_resize(frequency_vectors, all_label):
    X_train, X_test, y_train, y_test = train_test_split(frequency_vectors, all_label, test_size=0.2)
    bgg = BaggingClassifier(estimator=SVC(), n_estimators=10, random_state=0)
    y_pred = bgg.fit(X_train, y_train).predict(X_test)
    score = accuracy_score(y_pred, y_test)
    answer = "without_changes bg" + str(score)
    classifier_result.write(answer)
    print(answer)


def bagging_img_average_resize(frequency_vectors, all_label):
    X_train, X_test, y_train, y_test = train_test_split(frequency_vectors, all_label, test_size=0.2)
    bgg = BaggingClassifier(estimator=SVC(),n_estimators=10, random_state=0)
    y_pred = bgg.fit(X_train, y_train).predict(X_test)
    score = accuracy_score(y_pred, y_test)
    answer = "average_resize bg" + str(score)
    classifier_result.write(answer)
    print(answer)


def random_forest_class_average_resize(frequency_vectors, all_label):
    X_train, X_test, y_train, y_test = train_test_split(frequency_vectors, all_label, test_size=0.2)
    clf = RandomForestClassifier(max_depth=2, random_state=0)
    y_pred = clf.fit(X_train, y_train).predict(X_test)
    score = accuracy_score(y_pred, y_test)
    answer = "average_resize rfc" + str(score)
    classifier_result.write(answer)
    print(answer)


def random_forest_class_img_not_resize(frequency_vectors, all_label):
    X_train, X_test, y_train, y_test = train_test_split(frequency_vectors, all_label, test_size=0.2)
    clf = RandomForestClassifier(max_depth=2, random_state=0)
    y_pred = clf.fit(X_train, y_train).predict(X_test)
    score = accuracy_score(y_pred, y_test)
    answer = "without_changes rfc" + str(score)
    classifier_result.write(answer)
    print(answer)


def ada_boost_img_not_resize(frequency_vectors, all_label):
    X_train, X_test, y_train, y_test = train_test_split(frequency_vectors, all_label, test_size=0.2)
    clf = AdaBoostClassifier(n_estimators=100, random_state=0)
    y_pred = clf.fit(X_train, y_train).predict(X_test)
    score = accuracy_score(y_pred, y_test)
    answer = "without_changes ada" + str(score)
    classifier_result.write(answer)
    print(answer)


def ada_boost_average_resize(frequency_vectors, all_label):
    X_train, X_test, y_train, y_test = train_test_split(frequency_vectors, all_label, test_size=0.2)
    clf = AdaBoostClassifier(n_estimators=100, random_state=0)
    y_pred = clf.fit(X_train, y_train).predict(X_test)
    score = accuracy_score(y_pred, y_test)
    answer = "average_resize ada" + str(score)
    classifier_result.write(answer)
    print(answer)

# By default in evaluate function, we used SIFT_extraction but you can change by BRISK_extraction or ORB_extraction
def evaluate():

    # get score for image which size not change
    for i in range(50):
        descriptors, labels = SIFT_extraction.extract_features(sea_ocean, other, "without_changes")
        codebook, cluster_nb = SIFT_extraction.K_means(descriptors)
        frequency_vectors = SIFT_extraction.calculate_frequencies(descriptors, codebook, cluster_nb)

        t = threading.Thread(target=gaussian_img_not_resize(frequency_vectors, labels))
        t1 = threading.Thread(target=k_nn_img_not_resize(frequency_vectors, labels))
        t2 = threading.Thread(target=perceptron_img_not_resize(frequency_vectors, labels))
        t3 = threading.Thread(target=bagging_img_not_resize(frequency_vectors, labels))
        t8 = threading.Thread(target=random_forest_class_img_not_resize(frequency_vectors, labels))
        t10 = threading.Thread(target=ada_boost_img_not_resize(frequency_vectors, labels))

        t.start()
        t1.start()
        t2.start()
        t3.start()
        t8.start()
        t10.start()

    # get score for image which size change
    for i in range(50):
        descriptors, labels = SIFT_extraction.extract_features(sea_ocean, other, "average_resize")
        codebook, cluster_nb = SIFT_extraction.K_means(descriptors)
        frequency_vectors = SIFT_extraction.calculate_frequencies(descriptors, codebook, cluster_nb)

        t4 = threading.Thread(target=gaussian_with_average_resize(frequency_vectors, labels))
        t5 = threading.Thread(target=k_nn_with_average_resize(frequency_vectors, labels))
        t6 = threading.Thread(target=perceptron_average_resize(frequency_vectors, labels))
        t7 = threading.Thread(target=bagging_img_average_resize(frequency_vectors, labels))
        t9 = threading.Thread(target=random_forest_class_average_resize(frequency_vectors, labels))
        t11 = threading.Thread(target=ada_boost_average_resize(frequency_vectors, labels))

        t4.start()
        t5.start()
        t6.start()
        t9.start()
        t7.start()
        t11.start()


def getScore():
    file_result = open("classifier_result.txt.txt")

    file_result = file_result.read().split()

    z = 0
    a_g = []
    a_knn = []
    a_p = []
    a_bg = []
    a_ada = []
    a_rfc = []
    wc_g = []
    wc_knn = []
    wc_p = []
    wc_bg = []
    wc_ada = []
    wc_rfc = []

    for i in range(len(file_result)):
        if file_result[i] == "average_resize":
            if file_result[i + 1] == "g":
                a_g.append(float(file_result[i + 2]))
            if file_result[i + 1] == "knn":
                a_knn.append(float(file_result[i + 2]))
            if file_result[i + 1] == "p":
                a_p.append(float(file_result[i + 2]))
            if file_result[i + 1] == "bg":
                a_bg.append(float(file_result[i + 2]))
            if file_result[i + 1] == "ada":
                a_ada.append(float(file_result[i + 2]))
            if file_result[i + 1] == "rfc":
                a_rfc.append(float(file_result[i + 2]))

        if file_result[i] == "without_changes":
            if file_result[i + 1] == "g":
                wc_g.append(float(file_result[i + 2]))
            if file_result[i + 1] == "knn":
                wc_knn.append(float(file_result[i + 2]))
            if file_result[i + 1] == "p":
                wc_p.append(float(file_result[i + 2]))
            if file_result[i + 1] == "bg":
                wc_bg.append(float(file_result[i + 2]))
            if file_result[i + 1] == "ada":
                wc_ada.append(float(file_result[i + 2]))
            if file_result[i + 1] == "rfc":
                wc_rfc.append(float(file_result[i + 2]))
                z += 1
    print("image resize")
    print("\tgaussian: ", sum(a_g) / len(a_g))
    print("\tknn: ", sum(a_knn) / len(a_knn))
    print("\tperceptron: ", sum(a_p) / len(a_p))
    print("\tbagging: ", sum(a_bg) / len(a_bg))
    print("\tada boost: ", sum(a_ada) / len(a_ada))
    print("\trandom forest : ", sum(a_rfc) / len(a_rfc))
    print("image not resize")
    print("\tgaussian: ", sum(wc_g) / len(wc_g))
    print("\tknn: ", sum(wc_knn) / len(wc_knn))
    print("\tperceptron: ", sum(wc_p) / len(wc_p))
    print("\tbagging: ", sum(wc_bg) / len(wc_bg))
    print("\tada boost: ", sum(wc_ada) / len(wc_ada))
    print("\trandom forest : ", sum(wc_rfc) / len(wc_rfc))


evaluate()
classifier_result.close()