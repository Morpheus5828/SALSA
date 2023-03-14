import threading

from bovw_image_classification import bovw_method
from sklearn.linear_model import Perceptron
from sklearn.ensemble import BaggingClassifier
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from bovw_image_classification.bovw_method import *
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

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
    print("average_resize bg", score)


def random_forest_class_img_not_resize(frequency_vectors, all_label):
    X_train, X_test, y_train, y_test = train_test_split(frequency_vectors, all_label, test_size=0.2)
    clf = RandomForestClassifier(max_depth=2, random_state=0)
    y_pred = clf.fit(X_train, y_train).predict(X_test)
    score = accuracy_score(y_pred, y_test)
    print("without_changes", score)


def ada_boost_img_not_resize(frequency_vectors, all_label):
    X_train, X_test, y_train, y_test = train_test_split(frequency_vectors, all_label, test_size=0.2)
    clf = RandomForestClassifier(max_depth=2, random_state=0)
    y_pred = clf.fit(X_train, y_train).predict(X_test)
    score = accuracy_score(y_pred, y_test)
    print("without_changes", score)


def ada_boost_average_resize(frequency_vectors, all_label):
    X_train, X_test, y_train, y_test = train_test_split(frequency_vectors, all_label, test_size=0.2)
    clf = RandomForestClassifier(max_depth=2, random_state=0)
    y_pred = clf.fit(X_train, y_train).predict(X_test)
    score = accuracy_score(y_pred, y_test)
    print("average_resize", score)


def evaluate():
    for i in range(50):
        all_label, training_img = prepare_data(sea_ocean, other, "without_changes")
        all_desc, descriptors, all_label = extract_feature(training_img, all_label)
        codebook, cluster_nb = create_codebook(all_desc)
        visual_words = generate_visual_words(descriptors, codebook)
        frequency_vectors = calculate_frequency(visual_words, cluster_nb)

        t = threading.Thread(target=gaussian_img_not_resize(frequency_vectors, all_label))
        t1 = threading.Thread(target=k_nn_img_not_resize(frequency_vectors, all_label))
        t2 = threading.Thread(target=perceptron_img_not_resize(frequency_vectors, all_label))
        t3 = threading.Thread(target=bagging_img_not_resize(frequency_vectors, all_label))

        t.start()
        t1.start()
        t2.start()
        t3.start()

    for i in range(50):
        all_label, training_img = prepare_data(sea_ocean_average, other_average, "average_resize")
        all_desc, descriptors, all_label = extract_feature(training_img, all_label)
        codebook, cluster_nb = create_codebook(all_desc)
        visual_words = generate_visual_words(descriptors, codebook)
        frequency_vectors = calculate_frequency(visual_words, cluster_nb)

        t4 = threading.Thread(target=gaussian_with_average_resize(frequency_vectors, all_label))
        t5 = threading.Thread(target=k_nn_with_average_resize(frequency_vectors, all_label))
        t6 = threading.Thread(target=perceptron_average_resize(frequency_vectors, all_label))
        t7 = threading.Thread(target=bagging_img_average_resize(frequency_vectors, all_label))

        t4.start()
        t5.start()
        t6.start()
        t7.start()


evaluate()