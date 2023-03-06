import os
import numpy as np
import cv2
import random

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier
from sklearn.neural_network import MLPClassifier


def get_sift(source, sift):
    source = cv2.imread(source)
    gray1 = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
    return sift.detectAndCompute(gray1, None)

def get_sift_from_filter(source, sift):
    image = cv2.imread(source)
    image = cv2.normalize(src=image, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_bound = np.array([80, 50, 50])
    upper_bound = np.array([130, 255, 255])
    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    image2 = cv2.bitwise_and(image, image, mask=mask)

    gray1 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    return sift.detectAndCompute(gray1, None)


def get_sift_from(data, sift):
    result = []
    for label, filename, path in data:
        result.append(get_sift(path, sift))
    return result


def compute_score(kp1, des1, kp2, des2):
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    if des2 is not None and des1 is not None:
        if len(des1) > 1 and len(des2) > 1:
            matches = flann.knnMatch(des1, des2, k=2)
            return len(matches) / (len(kp2) + len(kp1))
        else:
            return 0
    else:
        return 0

def filter_blue_value(image):
    image = cv2.imread(image)
    image = cv2.normalize(src=image, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_bound = np.array([80, 50, 50])
    upper_bound = np.array([130, 255, 255])
    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, hierarchy = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    real_contours = []
    for cnt in contours:
        if cv2.contourArea(cnt) > 100.0:
            real_contours.append(cnt)

    average_size = 0
    for cnt in real_contours:
        average_size += cv2.contourArea(cnt)
    if average_size != 0:
        average_size = int(average_size / len(real_contours))

    return average_size, len(real_contours)
def rep(source, destinations, sift):
    scores = []
    for kp1, des1 in destinations:
        kp2, des2 = get_sift(source, sift)
        scores.append(compute_score(kp1, des1, kp2, des2))

    return np.sort(scores)

def rep3(source, destinations, sift):
    result = []
    scores = rep(source, destinations, sift)
    average = 0
    for i in range(1, 6):
        average += scores[-i]
    result.append(average / 5)
    av, size = filter_blue_value(source)
    result.append(av)
    result.append(size)
    return result

def rep2(source, destinations, sift):
    scores = []
    kp2, des2 = get_sift_from_filter(source, sift)
    for kp1, des1 in destinations:
        scores.append(compute_score(kp1, des1, kp2, des2))

    return np.sort(scores)



def raw_image_to_representation(image, representation):
    if representation == "KP":
        image = cv2.imread(image)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        sift = cv2.SIFT_create()
        kp = sift.detect(gray, None)
        image = cv2.drawKeypoints(image, kp, image)
        return image

    if representation == "KP-BLUE":
        image = cv2.imread(image)
        image = cv2.normalize(src=image, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_bound = np.array([80, 50, 50])
        upper_bound = np.array([130, 255, 255])
        mask = cv2.inRange(hsv, lower_bound, upper_bound)
        kernel = np.ones((7, 7), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        image2 = cv2.bitwise_and(image, image, mask=mask)

        gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
        sift = cv2.SIFT_create()
        kp = sift.detect(gray, None)
        image = cv2.drawKeypoints(image, kp, image2)
        return image

    if representation == "TEST2":
        image = cv2.imread(image)
        image = cv2.normalize(src=image, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        lower_bound = np.array([80, 50, 50])
        upper_bound = np.array([130, 255, 255])
        mask = cv2.inRange(hsv, lower_bound, upper_bound)
        kernel = np.ones((7, 7), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        contours, hierarchy = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        real_contours = []
        for cnt in contours:
            if cv2.contourArea(cnt) > 100.0:
                real_contours.append(cnt)

        average_size = 0
        for cnt in real_contours:
            average_size += cv2.contourArea(cnt)
        average_size = int(average_size / len(real_contours))

        return [average_size, len(real_contours)]


def get_label(folder):
    if folder == "Mer":
        return 1
    else:
        return -1


def split_random(data, k):
    result = []
    index = []
    for i in range(k):
        j = random.randint(0, len(data)-1-i)
        while j in index:
            j = random.randint(0, len(data)-1-i)
        index.append(j)
    for i in index:
        result.append(data.pop(i))
    return result


def load_split_data(directory, k_split):
    data = {"sea": [], "other": []}
    for folder in os.listdir(directory):
        for filename in os.listdir(os.path.join(directory, folder)):
            if folder == "Mer":
                data["sea"].append((1, filename, os.path.join(directory, folder, filename)))
            else:
                data["other"].append((-1, filename, os.path.join(directory, folder, filename)))

    result = {"extracted": split_random(data["sea"], k_split), "images": data["sea"] + data["other"]}
    return result

def load_split_data2(directory, names_extract):
    data = {"extracted": [], "images": []}
    for folder in os.listdir(directory):
        for filename in os.listdir(os.path.join(directory, folder)):
            if folder == "Mer":
                if filename in names_extract:
                    data["extracted"].append((1, filename, os.path.join(directory, folder, filename)))
                else:
                    data["images"].append((1, filename, os.path.join(directory, folder, filename)))
            else:
                if filename in names_extract:
                    data["extracted"].append((-1, filename, os.path.join(directory, folder, filename)))
                else:
                    data["images"].append((-1, filename, os.path.join(directory, folder, filename)))

    return data


'''def load_transform_label_train_data(directory, representation):
    data = {"representations": [], "labels": [], "filenames": []}
    for folder in os.listdir(directory):
        for filename in os.listdir(os.path.join(directory, folder)):
            data["representations"].append(
                raw_image_to_representation(os.path.join(directory, folder, filename), representation))
            data["labels"].append(get_label(folder))
            data["filenames"].append(filename)
    return data'''


def load_transform_label_train_data(directory, representation, k_split, names):
    sift = cv2.SIFT_create()
    load = load_split_data2(directory, names)
    dest = get_sift_from(load["extracted"], sift)

    data = {"representations": [], "labels": [], "filenames": []}
    for label, filename, path in load["images"]:

        data["representations"].append(rep3(path, dest, sift))
        data["labels"].append(label)
        data["filenames"].append(filename)
    return data


def load_transform_label_train_data2(directory, representation, k_split):
    sift = cv2.SIFT_create()
    load = load_split_data(directory, k_split)
    dest = get_sift_from(load["extracted"], sift)

    data = {"representations": [], "labels": [], "filenames": []}
    for label, filename, path in load["images"]:

        data["representations"].append(rep2(path, dest, sift))
        data["labels"].append(label)
        data["filenames"].append(filename)
    return data


def load_transform_test_data(directory, representation):
    data = {"representations": [], "labels": [], "filenames": []}
    for folder in os.listdir(directory):
        for filename in os.listdir(directory):
            data["representations"].append(
                raw_image_to_representation(os.path.join(directory, filename), representation))
            data["labels"].append(filename)
    return data


def learn_model_from_data(train_data, algo_dico, model):
    data = train_data["representations"]
    target = train_data["labels"]
    model.fit(data, target)
    return model


def predict_example_label(example, model):
    label = model.predict([example])[0]
    return label


def predict_sample_label(data, model):
    predictions = model.predict(data)
    return predictions


def write_predictions(directory, filename, data, model):
    try:
        file = open(os.path.join(directory, filename), 'x')
    except FileExistsError:
        return "not OK"
    predictions = predict_sample_label(data["representations"], model)
    for index in range(0, len(data)):
        file.write(data["filenames"][index] + " label " + str(predictions[index]) + "\n")
    return "OK"


def estimate_model_score(train_data, algo_dico, k, solver):
    X_train, X_test, y_train, y_test = train_test_split(train_data["representations"], train_data["labels"],
                                                        test_size=k)
    model = learn_model_from_data({"representations": X_train, "labels": y_train}, algo_dico, solver)
    y_predicts = predict_sample_label(X_test, model)
    return accuracy_score(y_test, y_predicts)


def estimate_model_score_average(trial, train_data, algo_dico, k, solver):
    average = 0
    for index in range(trial):
        score = estimate_model_score(train_data, algo_dico, k, solver)
        average += score
    return average / trial

