import image_processing as processing

import os
import joblib

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

'''
This file was authored by Chloe BUTTIGIEG.
'''

'''
Determines the label according the category given.
If the category is sea, the label is 1.
Otherwise, the label is -1.
input = a category name. It can be a folder name. 
output = the label.
'''
def get_label(category):
    if category == "sea":
        return 1
    else:
        return -1


'''
Loads train data. Train data must be classed in two folders. The folder name is used to determine the label.
Train data are represented as a dictionary with 3 fields:
- images: contains a list of all images loaded.
- labels: contains a list of images' label
- filenames: contains a list of all images' filename
Elements of each list are in the same order. It means that the image images[0] has the label labels[0] and
his filename is filenames[0].
input = folder path. It must be a folder of folder.
output = train data. It's a dictionary of list.
-- use open-cv and os libraries
'''
def load_labeled_train_data(folder_of_folder):
    train_data = {"images": [], "labels": [], "filenames": []}
    for folder in os.listdir(folder_of_folder):
        for filename in os.listdir(os.path.join(folder_of_folder, folder)):
            train_data["images"].append(processing.open_image(os.path.join(folder_of_folder, folder, filename)))
            train_data["labels"].append(get_label(folder))
            train_data["filenames"].append(filename)
    return train_data


'''
Loads test data. Test data are represented as a dictionary with 2 fields:
- images: contains a list of all images loaded.
- filenames: contains a list of all images' filename
Elements of each list are in the same order. It means that the filename of image images[0] is filenames[0].
input = folder path. It must be a folder that contains all the test images.
output = test data. It's a dictionary of list.
-- use open-cv and os libraries
'''
def load_test_data(folder_path):
    test_data = {"images": [], "filenames": []}
    for filename in os.listdir(folder_path):
        test_data["images"].append(processing.open_image(os.path.join(folder_path, filename)))
        test_data["filenames"].append(filename)
    return test_data

# --------------------------------------------------------------------------------- #


'''
Learns given model from train data. Train data must be a dictionary with 2 required fields:
- representations: it's a list that contains all images' representation. Each representation is also a list.
Thus, representations is a matrix.
- labels: a list that contains all images' labels.
Each list are in the same order. It means that the label of representation representations[0] is labels[0].
input = train data as a dictionary of list, a model from scikit-learn library.
output = the model.
'''
def learn_model_from_data(train_data, model):
    data = train_data["representations"]
    target = train_data["labels"]
    model.fit(data, target)
    return model


# --------------------------------------------------------------------------------- #

'''
Predicts label of a given representation from a given model. The representation must be a list.
input = a representation, a model from scikit-learn library.
output = the label predicted.
'''
def predict_example_label(example, model):
    label = model.predict([example])[0]
    return label


'''
Predicts labels of a given list of representation from a given model. Each representation must be a list.
input = a list of representation (thus a matrix), a model from scikit-learn library.
output = a list of labels predicted.
'''
def predict_sample_label(data, model):
    predictions = model.predict(data)
    return predictions


# --------------------------------------------------------------------------------- #

'''
Estimates the accuracy score of a given model. The model is trained from the given train data.
Train data must be a dictionary with 2 required fields:
- representations: it's a list that contains all images' representation. Each representation is also a list.
Thus, representations is a matrix.
- labels: a list that contains all images' labels.
Each list are in the same order. It means that the label of representation representations[0] is labels[0].
input = train data as a dictionary, a model from scikit-learn library, the proportion of train data to be included in the test split.
output = accuracy score of the model trained on the given data.
-- use scikit-learn library
'''
def estimate_model_score(train_data, model, k):
    X_train, X_test, y_train, y_test = train_test_split(train_data["representations"], train_data["labels"],
                                                        test_size=k)
    model = learn_model_from_data({"representations": X_train, "labels": y_train}, model)
    y_predits = predict_sample_label(X_test, model)
    return accuracy_score(y_predits, y_test)


'''
Estimates the average accuracy score of a given model. The model is trained from the given train data.
Train data must be a dictionary with 2 required fields:
- representations: it's a list that contains all images' representation. Each representation is also a list.
Thus, representations is a matrix.
- labels: a list that contains all images' labels.
Each list are in the same order. It means that the label of representation representations[0] is labels[0].
input = the number of iteration, train data as a dictionary,
    a model from scikit-learn library, the proportion of train data to be included in the test split.
output = accuracy score of the model trained on the given data.
-- use scikit-learn library
'''
def estimate_model_score_average(trial, train_data, model, k):
    average = 0
    for index in range(trial):
        score = estimate_model_score(train_data, model, k)
        average += score
    return average / trial


# --------------------------------------------------------------------------------- #

'''
Writes a file that contains labels predicted of given data from a trained model.
Data must be a dictionary with 2 required fields:
- representations: it's a list that contains all images' representation. Each representation is also a list.
Thus, representations is a matrix.
- filenames: it's a list of all images' filename
Elements of each list are in the same order. It means that the filename of image images[0] is filenames[0].
input = filepath of the prediction file,
        filename of the prediction file. If a file as the same name, the function will NOT overwrite it,
        data to be predicted, it's a dictionary,
        a model from scikit-learn library that has been previously trained.        
output = 'not OK' if the file already exist, 'OK' if the file was successfully computed.
'''
def write_predictions(filepath, filename, data, model):
    try:
        file = open(os.path.join(filepath, filename), 'x')
    except FileExistsError:
        return "not OK"
    predictions = predict_sample_label(data["representations"], model)
    for index in range(0, len(data["representations"])):
        if predictions[index] == 1:
            file.write(data["filenames"][index] + " " + "+1" + "\n")
        if predictions[index] == -1:
            file.write(data["filenames"][index] + " " + "-1" + "\n")
    return "OK"


# --------------------------------------------------------------------------------- #


'''
Saves a model from scikit-learn.
input = model to be saved, filename of the model without extension, filepath of the model.
-- use joblib library
'''
def save_model(model, filename, filepath):
    joblib.dump(model, os.path.join(filepath, filename+".joblib"))
    print("Model " + filename + " saved.")


'''
Load a model from a joblib file as a scikit-learn model.
input = filepath of the model.
output = the model.
-- use joblib library
'''
def load_model(filepath):
    return joblib.load(filepath)
