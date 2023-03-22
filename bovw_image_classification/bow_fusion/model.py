import os
import joblib

import image_representation as representation
import image_processing as processing

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def get_label(category):
    if category == "sea":
        return 1
    else:
        return -1


def load_labeled_train_data(folder_of_folder):
    train_data = {"images": [], "labels": [], "filenames": []}
    for folder in os.listdir(folder_of_folder):
        for filename in os.listdir(os.path.join(folder_of_folder, folder)):
            train_data["images"].append(processing.open_image(os.path.join(folder_of_folder, folder, filename)))
            train_data["labels"].append(get_label(folder))
            train_data["filenames"].append(filename)
    return train_data


def load_test_data(folder):
    test_data = {"images": [], "filenames": []}
    for filename in os.listdir(folder):
        test_data["images"].append(processing.open_image(os.path.join(folder, filename)))
        test_data["filenames"].append(filename)
    return test_data


def compute_representation(data, nb_clusters):
    representation.compute_fusion_representation(data, nb_clusters)

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
    X_train, X_test, y_train, y_test = train_test_split(train_data["representations"], train_data["labels"],
                                                        test_size=k)
    model = learn_model_from_data({"representations": X_train, "labels": y_train}, model)
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


# --------------------------------------------------------------------------------- #


def save_model(model, filename, filepath):
    joblib.dump(model, os.path.join(filepath, filename))


def load_model(filepath):
    return joblib.load(filepath)
