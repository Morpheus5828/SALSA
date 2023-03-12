import os
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def get_representation(clustering_model, descriptor, nb_clusters):
    histogram = np.zeros(nb_clusters)
    clusters = clustering_model.predict(descriptor)
    for cluster in clusters:
        histogram[cluster] += 1
    return histogram

# --------------------------------------------------------------------------------- #


def get_label(category):
    if category == "Mer":
        return 1
    else:
        return -1


def load_transform_label_train_data(dir_of_dir, descriptors, clustering_model, nb_clusters):
    data = {"representations": [], "labels": [], "filenames": []}
    index = 0
    for dir in os.listdir(dir_of_dir):
        for filename in os.listdir(os.path.join(dir_of_dir, dir)):
            data["representations"].append(get_representation(clustering_model, descriptors[index], nb_clusters))
            data["labels"].append(get_label(dir))
            data["filenames"].append(filename)
            index += 1
    return data


def load_transform_test_data(directory):
    data = {"representations": [], "labels": [], "filenames": []}
    for folder in os.listdir(directory):
        for filename in os.listdir(directory):
            data["representations"].append(get_representation(os.path.join(directory, filename)))
            data["labels"].append(filename)
    return data

# --------------------------------------------------------------------------------- #


def learn_model_from_data(train_data, model):
    data = train_data["representations"]
    target = train_data["labels"]
    model.fit(data, target)
    return model

# --------------------------------------------------------------------------------- #


def predict_example_label(example, model):
    label = model.predict([example])[0]
    return label


def predict_sample_label(data, model):
    predictions = model.predict(data)
    return predictions

# --------------------------------------------------------------------------------- #


def estimate_model_score(train_data, model, k):
    X_train, X_test, y_train, y_test = train_test_split(train_data["representations"], train_data["labels"], test_size=k)
    model = learn_model_from_data({"representations":X_train, "labels":y_train}, model)
    y_predits = predict_sample_label(X_test, model)
    return accuracy_score(y_test, y_predits)


def estimate_model_score_average(trial, train_data, model, k):
    average = 0
    for index in range(trial):
        score = estimate_model_score(train_data, model, k)
        average += score
    return average / trial
