import model
import resize
import bag_of_word as bow
import image_representation as representation

import os

'''
Computes classifier from train data and save it. It also save clustering model computed. The classifier
is saved under the directory 'save/model/classification' and both clustering model are saved in the
directory : 'save/model/clustering'.
input = path to find train data. It must be a folder of folder,
        filename of the sift clustering model,
        filename of the color clustering model,
        filename of the classifier,
        number of clusters,
        model of scikit-learn library.
'''
def compute_and_save_classifier(train_data_path, sift_clustering_filename, color_clustering_filename,
                              classifier_filename, nb_clusters, classifier):
    # Load train data
    train_data = model.load_labeled_train_data(train_data_path)

    # Resize images
    average_height = resize.get_average_height(train_data["images"])
    average_width = resize.get_average_width(train_data["images"])

    resized_images = resize.resize_to(train_data["images"], average_height, average_width)

    # Compute BOW
    sift_bow = bow.SiftBow(resized_images)
    color_bow = bow.ColorBow(resized_images, 500)

    sift_bow.learn_clustering_model(nb_clusters)
    color_bow.learn_clustering_model(nb_clusters)

    # Compute image's representation
    representation.compute_bow_fusion_representation(train_data, sift_bow, color_bow)
    representation.normalize_representation(train_data)

    # Compute model
    model.learn_model_from_data(train_data, classifier)

    model.save_model(classifier, classifier_filename, "save/model/classification")
    model.save_model(sift_bow.clustering_model, sift_clustering_filename, "save/model/clustering")
    model.save_model(color_bow.clustering_model, color_clustering_filename, "save/model/clustering")

'''
Predicts and writes in a file predicted labels from given data. The prediction file is saved
under the directory 'save/prediction'.
input = filename of the prediction file,
        filepath of the data,
        path of the saved sift clustering algorithm as joblib file,
        path of the saved color clustering algorithm as joblib file,
        path of the saved classifier as joblib file,
        number of clusters used for both clustering algorithm.
'''
def predict_test_data(prediction_filename, test_data_path, sift_clustering_path, color_clustering_path,
                      classifier_path, nb_clusters):
    # Load test data
    test_data = model.load_test_data(test_data_path)

    # Resize images
    average_height = resize.get_average_height(test_data["images"])
    average_width = resize.get_average_width(test_data["images"])

    resized_images = resize.resize_to(test_data["images"], average_height, average_width)

    # Load BOW
    sift_bow = bow.SiftBow(resized_images)
    color_bow = bow.ColorBow(resized_images, 500)

    sift_bow.assign_clustering_model(model.load_model(sift_clustering_path), nb_clusters)
    color_bow.assign_clustering_model(model.load_model(color_clustering_path), nb_clusters)

    # Compute image's representation
    representation.compute_bow_fusion_representation(test_data, sift_bow, color_bow)
    representation.normalize_representation(test_data)

    # Load model
    salsa = model.load_model(classifier_path)

    # Predict
    model.write_predictions("save/prediction", prediction_filename, test_data, salsa)


'''
Evaluate matching score between predictions and expected labels.
input = path of the prediction path. The file format is: <filename> <label>
        path of labeled data from which the prediction was extracted. The path must be a folder of folder.
output = matching score.
- use os library
'''
def evaluate_prediction(prediction_path, test_labeled_data_path):
    # Get predicted labels
    file = open(prediction_path, "r")
    labels_predicted = {}
    for line in file:
        data = line.split()
        labels_predicted[data[0]] = int(data[1])

    # Get expected labels
    labels = {}
    for folder in os.listdir(test_labeled_data_path):
        for filename in os.listdir(os.path.join(test_labeled_data_path, folder)):
            if folder == "sea":
                labels[filename] = 1
            else:
                labels[filename] = -1

    # Evaluate
    if labels.keys() != labels_predicted.keys():
        return "ERROR DATA DOESN'T MATCH"

    nb_correct = 0
    for filename in labels.keys():
        if labels.get(filename) == labels_predicted.get(filename):
            nb_correct += 1

    return nb_correct / len(labels)


'''
Compute average accuracy score for a given classifier of a BoW representation. The function varies the number
of clusters used for the K-Means algorithm, from min_clusters to max_clusters-1. All  the result are saved
in a file under the directory : 'save/result'. Train data must be a dictionary with 2 required fields:
- images: it's a list that contains all images opened as numpy array.
- labels: a list that contains all images' labels.
Each list are in the same order. It means that the label of image images[0] is labels[0].
input = filename of the result file,
        number of clusters to start,
        max number of clusters,
        train data, as a dictionary of list,
        a bag of words from bag_of_words.py file,
        a classifier from scikit-learn library,
        a boolean. True is data must be normalized, False otherwise.
'''
def compute_all_clusters(result_filename, min_cluster, max_cluster, train_data, bow, classifier, normalized):
    file = open(os.path.join("save/result", result_filename), 'w')
    for nb_clusters in range(min_cluster, max_cluster):
        print(nb_clusters)
        bow.learn_clustering_model(nb_clusters)

        representation.compute_bow_representation(train_data, bow)
        if normalized:
            representation.normalize_representation(train_data)
        score = model.estimate_model_score_average(50, train_data, classifier, 0.2)
        file.write(str(nb_clusters) + " " + str(score) + "\n")
    file.close()


'''
Compute average accuracy score for a given classifier of BoW fusion representation. The function varies the number
of clusters used for the K-Means algorithm, from min_clusters to max_clusters-1. All  the result are saved
in a file under the directory : 'save/result'. Train data must be a dictionary with 2 required fields:
- images: it's a list that contains all images opened as numpy array.
- labels: a list that contains all images' labels.
Each list are in the same order. It means that the label of image images[0] is labels[0].
input = filename of the result file,
        number of clusters to start,
        max number of clusters,
        train data, as a dictionary of list,
        a first bag of words from bag_of_words.py file,
        a second bag of words from bag_of_words.py file,
        a classifier from scikit-learn library,
        a boolean. True is data must be normalized, False otherwise.
'''
def compute_all_clusters_fusion(result_filename, min_cluster, max_cluster, train_data, bow1, bow2, classifier,
                                normalized):
    file = open(os.path.join("save/result", result_filename), 'w')
    for nb_clusters in range(min_cluster, max_cluster):
        print(nb_clusters)
        bow1.learn_clustering_model(nb_clusters)
        bow2.learn_clustering_model(nb_clusters)

        representation.compute_bow_fusion_representation(train_data, bow1, bow2)
        if normalized:
            representation.normalize_representation(train_data)

        score = model.estimate_model_score_average(50, train_data, classifier, 0.2)
        file.write(str(nb_clusters) + " " + str(score) + "\n")
    file.close()
