import threading

import cv2
from sklearn.linear_model import Perceptron
from sklearn.ensemble import BaggingClassifier
from sklearn.svm import SVC
from bovw_image_classification.bovw_method import *
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from bovw_image_classification.extraction import BRISK_extraction, SIFT_extraction

sea_ocean = os.listdir("../dataset/sea_ocean/without_changes")
other = os.listdir("../dataset/other/without_changes")

sea_ocean_average = os.listdir("../dataset/sea_ocean/average_resize")
other_average = os.listdir("../dataset/other/average_resize")


def gaussian_img_not_resize(frequency_vectors, all_label):
    X_train, X_test, y_train, y_test = train_test_split(frequency_vectors, all_label, test_size=0.2)
    gnb = GaussianNB()
    y_pred = gnb.fit(X_train, y_train).predict(X_test)
    score = accuracy_score(y_pred, y_test)
    print("without_changes g ", score)


def gaussian_with_average_resize(frequency_vectors, all_label):
    X_train, X_test, y_train, y_test = train_test_split(frequency_vectors, all_label, test_size=0.2)
    gnb = GaussianNB()
    y_pred = gnb.fit(X_train, y_train).predict(X_test)
    score = accuracy_score(y_pred, y_test)
    print("average_resize g ", score)


def k_nn_img_not_resize(frequency_vectors, all_label):
    X_train, X_test, y_train, y_test = train_test_split(frequency_vectors, all_label, test_size=0.2)
    neigh = KNeighborsClassifier(n_neighbors=3)
    y_pred = neigh.fit(X_train, y_train).predict(X_test)
    score = accuracy_score(y_pred, y_test)
    print("without_changes knn", score)


def k_nn_with_average_resize(frequency_vectors, all_label):
    X_train, X_test, y_train, y_test = train_test_split(frequency_vectors, all_label, test_size=0.2)
    neigh = KNeighborsClassifier(n_neighbors=3)
    y_pred = neigh.fit(X_train, y_train).predict(X_test)
    score = accuracy_score(y_pred, y_test)
    print("average_resize knn", score)


def perceptron_img_not_resize(frequency_vectors, all_label):
    X_train, X_test, y_train, y_test = train_test_split(frequency_vectors, all_label, test_size=0.2)
    pct = Perceptron(tol=1e-3, random_state=0)
    y_pred = pct.fit(X_train, y_train).predict(X_test)
    score = accuracy_score(y_pred, y_test)
    print("without_changes p", score)


def perceptron_average_resize(frequency_vectors, all_label):
    X_train, X_test, y_train, y_test = train_test_split(frequency_vectors, all_label, test_size=0.2)
    pct = Perceptron(tol=1e-3, random_state=0)
    y_pred = pct.fit(X_train, y_train).predict(X_test)
    score = accuracy_score(y_pred, y_test)
    print("average_resize p", score)


def bagging_img_not_resize(frequency_vectors, all_label):
    X_train, X_test, y_train, y_test = train_test_split(frequency_vectors, all_label, test_size=0.2)
    bgg = BaggingClassifier(estimator=SVC(), n_estimators=10, random_state=0)
    y_pred = bgg.fit(X_train, y_train).predict(X_test)
    score = accuracy_score(y_pred, y_test)
    print("without_changes bg", score)


def bagging_img_average_resize(frequency_vectors, all_label):
    X_train, X_test, y_train, y_test = train_test_split(frequency_vectors, all_label, test_size=0.2)
    bgg = BaggingClassifier(estimator=SVC(),n_estimators=10, random_state=0)
    y_pred = bgg.fit(X_train, y_train).predict(X_test)
    score = accuracy_score(y_pred, y_test)
    print("average_resize bg", score)


def random_forest_class_average_resize(frequency_vectors, all_label):
    X_train, X_test, y_train, y_test = train_test_split(frequency_vectors, all_label, test_size=0.2)
    clf = RandomForestClassifier(max_depth=2, random_state=0)
    y_pred = clf.fit(X_train, y_train).predict(X_test)
    score = accuracy_score(y_pred, y_test)
    print("average_resize rfc", score)


def random_forest_class_img_not_resize(frequency_vectors, all_label):
    X_train, X_test, y_train, y_test = train_test_split(frequency_vectors, all_label, test_size=0.2)
    clf = RandomForestClassifier(max_depth=2, random_state=0)
    y_pred = clf.fit(X_train, y_train).predict(X_test)
    score = accuracy_score(y_pred, y_test)
    print("without_changes rfc", score)


def ada_boost_img_not_resize(frequency_vectors, all_label):
    X_train, X_test, y_train, y_test = train_test_split(frequency_vectors, all_label, test_size=0.2)
    clf = AdaBoostClassifier(max_depth=2, random_state=0)
    y_pred = clf.fit(X_train, y_train).predict(X_test)
    score = accuracy_score(y_pred, y_test)
    print("without_changes ada", score)


def ada_boost_average_resize(frequency_vectors, all_label):
    X_train, X_test, y_train, y_test = train_test_split(frequency_vectors, all_label, test_size=0.2)
    clf = AdaBoostClassifier(max_depth=2, random_state=0)
    y_pred = clf.fit(X_train, y_train).predict(X_test)
    score = accuracy_score(y_pred, y_test)
    print("average_resize ada", score)


def evaluate():
    '''for i in range(50):
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
        t10.start()'''

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



evaluate()