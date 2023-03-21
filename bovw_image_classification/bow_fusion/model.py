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

def get_color_representation(clustering_model, descriptor, nb_clusters, idfs, nb_images):
    occurences_histogram = get_representation(clustering_model, descriptor, nb_clusters)
    representation = np.empty(nb_clusters)
    for cluster in range(nb_clusters):
        tf = occurences_histogram[cluster] / (occurences_histogram[cluster] + 1)
        idf = np.log((nb_images - idfs[cluster] + 0.5) / (idfs[cluster] + 0.5))
        if idf < 0:
            idf = 0
        representation[cluster] = tf * idf
    return representation

def get_sift_representation(clustering_model, descriptor, nb_clusters, idfs, nb_images, average):
    occurences_histogram = get_representation(clustering_model, descriptor, nb_clusters)
    representation = np.empty(nb_clusters)
    b = 0.75
    for cluster in range(nb_clusters):
        tf = occurences_histogram[cluster] / (occurences_histogram[cluster] + (1-b+b*(len(descriptor) / average)))
        idf = np.log((nb_images - idfs[cluster] + 0.5) / (idfs[cluster] + 0.5))
        if idf < 0:
            idf = 0
        representation[cluster] = tf * idf
    return representation

# --------------------------------------------------------------------------------- #


def get_label(category):
    if category == "sea":
        return 1
    else:
        return -1


def load_transform_label_train_data(dir_of_dir, descriptors, clustering_model, nb_clusters, idfs, nb_images):
    data = {"representations": [], "labels": [], "filenames": []}
    index = 0
    for dir in os.listdir(dir_of_dir):
        for filename in os.listdir(os.path.join(dir_of_dir, dir)):
            data["representations"].append(get_color_representation(clustering_model, descriptors[index], nb_clusters, idfs, nb_images))
            data["labels"].append(get_label(dir))
            data["filenames"].append(filename)
            index += 1
    return data

def load_transform_label_train_data2(dir_of_dir, descriptors, sift_descriptors, clustering_model, clustering_model_sm, nb_clusters, idfs, nb_images):
    data = {"representations": [], "labels": [], "filenames": []}
    index = 0
    for dir in os.listdir(dir_of_dir):
        for filename in os.listdir(os.path.join(dir_of_dir, dir)):
            mm = get_color_representation(clustering_model, descriptors[index], nb_clusters, idfs, nb_images)
            sm = get_representation(clustering_model_sm, sift_descriptors[index], nb_clusters)
            data["representations"].append(np.concatenate((mm, sm)))
            data["labels"].append(get_label(dir))
            data["filenames"].append(filename)
            index += 1
    return data

def load_transform_label_train_data3(dir_of_dir, descriptors, sift_descriptors, clustering_model, clustering_model_sm, nb_clusters):
    data = {"representations": [], "labels": [], "filenames": []}
    index = 0
    for dir in os.listdir(dir_of_dir):
        for filename in os.listdir(os.path.join(dir_of_dir, dir)):
            mm = get_representation(clustering_model, descriptors[index], nb_clusters)
            sm = get_representation(clustering_model_sm, sift_descriptors[index], nb_clusters)
            data["representations"].append(np.concatenate((mm, sm)))
            data["labels"].append(get_label(dir))
            data["filenames"].append(filename)
            index += 1
    return data

def load_transform_label_train_data4(dir_of_dir, descriptors, sift_descriptors, clustering_model, clustering_model_sm, nb_clusters, idfs, idfs_sift, nb_images, average):
    data = {"representations": [], "labels": [], "filenames": []}
    index = 0
    for dir in os.listdir(dir_of_dir):
        for filename in os.listdir(os.path.join(dir_of_dir, dir)):
            mm = get_color_representation(clustering_model, descriptors[index], nb_clusters, idfs, nb_images)
            sm = get_sift_representation(clustering_model_sm, sift_descriptors[index], nb_clusters, idfs_sift, nb_images, average)
            data["representations"].append(np.concatenate((mm, sm)))
            data["labels"].append(get_label(dir))
            data["filenames"].append(filename)
            index += 1
    return data


def load_transform_test_data(directory, clustering_model, descriptors, nb_clusters):
    data = {"representations": [], "filenames": []}
    index = 0
    for filename in os.listdir(directory):
        data["representations"].append(get_representation(clustering_model, descriptors[index], nb_clusters))
        data["filenames"].append(filename)
        index += 1
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
    return accuracy_score(y_predits, y_test)


def estimate_model_score_average(trial, train_data, model, k):
    average = 0
    for index in range(trial):
        score = estimate_model_score(train_data, model, k)
        average += score
    return average / trial

# --------------------------------------------------------------------------------- #


def write_predictions(directory, filename, data, model):
    try:
        file = open(os.path.join(directory, filename), 'w')
    except FileExistsError:
        return "not OK"
    predictions = predict_sample_label(data["representations"], model)
    print(predictions)
    for index in range(0, len(data["representations"])):
        file.write(data["filenames"][index] + " " + str(predictions[index]) + "\n")
    return "OK"