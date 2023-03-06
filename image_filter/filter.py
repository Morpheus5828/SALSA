import cv2
import os
import model
import numpy as np

from sklearn.ensemble import BaggingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


def get_sift(source, sift):
    source = cv2.imread(source)
    gray1 = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
    return sift.detectAndCompute(gray1, None)
def sift_from(source, dest, sift):
    if source == dest:
        return 0

    kp1, des1 = get_sift(source, sift)
    kp2, des2 = get_sift(dest, sift)
    return compute_score(kp1, des1, kp2, des2)


def compute_score(kp1, des1, kp2, des2):
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    return len(matches) / (len(kp2) + len(kp1))

def write(directory, filename, data):
    try:
        file = open(os.path.join(directory, filename), 'w')
    except FileExistsError:
        return "not OK"
    for index in range(0, len(data)):
        file.write(str(data[index]) + "\n")
    return "OK"


def get_label(category):
    if category == "Mer":
        return 1
    else:
        return -1


def load_transform_label_train_data(source, directory):
    data = {"scores": [], "labels": [], "filenames": []}
    for folder in os.listdir(directory):
        for filename in os.listdir(os.path.join(directory, folder)):
            data["labels"].append(get_label(folder))
            data["filenames"].append(filename)
    return data


def compare_sea(source, directory):
    sift = cv2.SIFT_create()
    kp1, des1 = get_sift(source, sift)
    scores = []
    for filename in os.listdir(directory):
        kp2, des2 = get_sift(os.path.join(directory, filename), sift)
        scores.append(compute_score(kp1, des1, kp2, des2))
    return scores

names = ["ar54ff.png", "ioio.jpeg", "kljui87cx4.jpg", "lmpp.jpeg", "mari5s.jpeg", "njhuy.jpeg", "TCHOME.jpg", "tr50.jpg", "wsmmff.jpeg", "xwvrr.jpeg"]
data = model.load_transform_label_train_data("data", "k", 10, names)
print("load data")

file = open("result.txt", 'w')

file.write("Perceptron lbfgs ")
score = model.estimate_model_score_average(50, data, [], 0.2, MLPClassifier(solver='lbfgs', max_iter=1000))
file.write(str(score) + "\n")

file.write("Bagging avec perceptron lbfgs ")
score = model.estimate_model_score_average(50, data, [], 0.2, BaggingClassifier(estimator=MLPClassifier(solver='lbfgs', max_iter=1000)))
file.write(str(score) + "\n")

print("END")
file.close()



