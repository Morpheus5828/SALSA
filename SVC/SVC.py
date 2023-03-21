import glob

from PIL import Image
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_val_score

def image_representation(image):
    img = Image.open(image)
    image_resize = np.resize(img, (500, 500))
    return np.ravel(image_resize).tolist()


def label_data():
    label = []
    data = []

    img_mer = glob.glob("../dataset/sea_ocean/*")
    img_other = glob.glob("../dataset/other/*")

    for file_name in img_mer:
        label.append(1)
        data.append(image_representation(file_name))

    for file_name in img_other:
        label.append(-1)
        data.append(image_representation(file_name))

    return label, data

def test_data():
    test_data =[]
    img_test = glob.glob("../dataset/testdata/*")

    for file_name in img_test:
        test_data.append(image_representation(file_name))

    return test_data

def search_best_parameters(train_data, algo_dico) :
    X_train = train_data[1]
    Y_train = train_data[0]

    svc = SVC()
    grid = GridSearchCV(svc, algo_dico['param'], cv=5)
    grid.fit(X_train,Y_train)



    return grid.best_params_

algo_dico = {
        'algorithm_name': 'SVC',
        'param': {
            'C': [0.1, 1, 10, 100],
            'kernel': ['linear', 'rbf', 'sigmoid'],
            'gamma': ['scale', 0.1, 1],
            'verbose': [False]
        }
    }

traindata = label_data()


#{'C': 10, 'gamma': 'scale', 'kernel': 'rbf', 'verbose': False}

algo_dico_best_param = {
    'algorithm_name': 'SVC',
        'param': {
            'C': 10,
            'kernel': 'rbf',
            'gamma': 'scale',
            'verbose': False
        }
    }

def fit_model(train_data, algo_dico):
    X_train = train_data[1]
    Y_train = train_data[0]

    model = SVC(**algo_dico['param'])
    model.fit(X_train,Y_train)
    return model

def pred_label(exemple, model):

    pred_label = model.predict([exemple])
    return pred_label[0]

def pred_data (data, model):
    pred_label_data = []
    for image in data :
        pred = pred_label(image, model)
        pred_label_data.append(pred)

    return pred_label_data

def estimate_score(train_data, model, k) :
    X_train = train_data[1]
    Y_train = train_data[0]

    scores =  cross_val_score(model, X_train, Y_train, cv = k)
    score_moyen = scores.mean()

    return score_moyen

model = fit_model(label_data(),algo_dico_best_param)

def writter(filename, data, model):
    pred = pred_data(data, model)
    with open("prediction.txt","w") as file :
        for i in range(len(filename)):
            file.write(filename[i] + " " + str(pred[i]) + "\n")


data_test = test_data()
filename = glob.glob("../dataset/testdata/*")

writter(filename,data_test,model)