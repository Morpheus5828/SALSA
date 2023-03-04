import os

import matplotlib.pyplot as plt
from skimage import exposure
from skimage.color import rgb2gray
from PIL import Image
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier
from skimage.color import rgb2hsv
import cv2

"""
Computes a representation of an image from the (gif, png, jpg...) file 
representation can be (to extend) 
'HC': color histogram
'PX': tensor of pixels
'GC': matrix of gray pixels 
other to be defined
input = an image (jpg, png, gif)
output = a new representation of the image
"""   

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

    if representation == "HC":
        opened_image = plt.imread(image)
        colors = ['red', 'green', 'blue']
        result = []
        for index, color in enumerate(colors):
            hist, bins = exposure.histogram(opened_image[:, :, index], nbins=256)
            if (index == 0):
                result.append(bins)
            result.append(hist)
        return result

    if representation == "BLUE_HIST":
        image = plt.imread(image)
        hist, bins = exposure.histogram(image[:, :, 2], nbins=256)

        for index in range(int(bins[0])):
            bins = np.insert(bins, index, index)
            hist = np.insert(hist, index, 0)

        for index in range(len(bins), 256):
            bins = np.append(bins, index)
            hist = np.append(hist, 0)
        return hist

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

    if representation == "BLUE":
        image = plt.imread(image)
        if len(image[0][0]) == 4:
            image = image[:, :, :-1]
        image = cv2.normalize(src=image, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        image_hsv = rgb2hsv(image)

        lower_mask_1 = image_hsv[:, :, 0] > 0.45
        upper_mask_1 = image_hsv[:, :, 0] < 0.75
        saturation_1 = image_hsv[:, :, 1] > 0.15
        mask = lower_mask_1 * upper_mask_1 * saturation_1

        image = np.dstack((image[:, :, 0] * mask, image[:, :, 1] * mask, image[:, :, 2] * mask))
        hist, bins = exposure.histogram(image[:, :, 2], nbins=256)

        for index in range(int(bins[0])):
            bins = np.insert(bins, index, index)
            hist = np.insert(hist, index, 0)

        for index in range(len(bins), 256):
            bins = np.append(bins, index)
            hist = np.append(hist, 0)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray, 125, 255, 0)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        max_cnt = 0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > max_cnt:
                max_cnt = area


        return hist

    
    if representation == "PX":
        return np.asarray(Image.open(image))

    if representation == "GC":
        image = plt.imread(image)
        return rgb2gray(image[:, :, :3])

"""
Returns a data structure embedding train images described according to the 
specified representation and associate each image to its label.
-> Representation can be (to extend) 
'HC': color histogram
'PX': tensor of pixels 
'GC': matrix of gray pixels
other to be defined
input = where are the data, which represenation of the data must be produced ? 
output = a structure (dictionnary ? Matrix ? File ?) where the images of the
directory have been transformed and labelled according to the directory they are
stored in.
-- uses function raw_image_to_representation
"""

def get_label(category):
    if (category == "Mer"):
        return 1
    else:
        return -1

def load_transform_label_train_data(directory, representation):
    data = {"representations":[], "labels":[], "filenames":[]}
    for folder in os.listdir(directory):
        for filename in os.listdir(os.path.join(directory, folder)):
            data["representations"].append(raw_image_to_representation(os.path.join(directory, folder, filename), representation))
            data["labels"].append(get_label(folder))
            data["filenames"].append(filename)
    return data
    
    
"""
Returns a data structure embedding test images described according to the 
specified representation.
-> Representation can be (to extend) 
'HC': color histogram
'PX': tensor of pixels 
'GC': matrix of gray pixels 
other to be defined
input = where are the data, which represenation of the data must be produced ? 
output = a structure (dictionnary ? Matrix ? File ?) where the images of the
directory have been transformed (but not labelled)
-- uses function raw_image_to_representation
"""

def load_transform_test_data(directory, representation):
    data = {"representations":[], "labels":[], "filenames":[]}
    for folder in os.listdir(directory):
        for filename in os.listdir(directory):
            data["representations"].append(raw_image_to_representation(os.path.join(directory, filename), representation))
            data["labels"].append(filename)
    return data

"""
Learn a model (function) from a representation of data, using the algorithm 
and its hyper-parameters described in algo_dico
Here data has been previously transformed to the representation used to learn
the model
input = transformed labelled data, the used learning algo and its hyper-parameters (a dico ?)
output =  a model fit with data
"""

def learn_model_from_data(train_data, algo_dico):
    data = train_data["representations"]
    target = train_data["labels"]
    model = BaggingClassifier()
    model.fit(data, target)
    return model

"""
Given one example (representation of an image as used to compute the model),
computes its class according to a previously learned model.
Here data has been previously transformed to the representation used to learn
the model
input = representation of one data, the learned model
output = the label of that one data (+1 or -1)
-- uses the model learned by function learn_model_from_data
"""

def predict_example_label(example, model):
    label = model.predict([example])[0]
    return label


"""
Computes an array (or list or dico or whatever) that associates a prediction 
to each example (image) of the data, using a previously learned model. 
Here data has been previously transformed to the representation used to learn
the model
input = a structure (dico, matrix, ...) embedding all transformed data to a representation, and a model
output =  a structure that associates a label to each data (image) of the input sample
"""

def predict_sample_label(data, model):
    predictions = model.predict(data)
    return predictions

"""
Save the predictions on data to a text file with syntax:
filename <space> label (either -1 or 1)  
NO ACCENT  
Here data has been previously transformed to the representation used to learn
the model
input = where to save the predictions, structure embedding the data, the model used
for predictions
output =  OK if the file has been saved, not OK if not
"""

def write_predictions(directory, filename, data, model):
    try:
        file = open(os.path.join(directory, filename), 'x')
    except FileExistsError:
        return "not OK"
    predictions = predict_sample_label(data["representations"], model)
    for index in range(0, len(data)):
        file.write(data["filenames"][index] + " label " + str(predictions[index]) + "\n")
    return "OK"

"""
Estimates the accuracy of a previously learned model using train data, 
either through CV or mean hold-out, with k folds.
Here data has been previously transformed to the representation used to learn
the model
input = the train labelled data as previously structured, the learned model, and
the number of split to be used either in a hold-out or by cross-validation  
output =  The score of success (betwwen 0 and 1, the higher the better, scores under 0.5
are worst than random
"""

def estimate_model_score(train_data, algo_dico, k):
    X_train, X_test, y_train, y_test = train_test_split(train_data["representations"], train_data["labels"], test_size=k)
    model = learn_model_from_data({"representations":X_train, "labels":y_train}, algo_dico)
    y_predits = predict_sample_label(X_test, model)
    return accuracy_score(y_test, y_predits)
