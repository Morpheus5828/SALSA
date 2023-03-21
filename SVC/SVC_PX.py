import glob

from PIL import Image
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_val_score


def average_width_height() :

    width_size = float(0)
    height_size = float(0)
    counter = 0
    img_mer = glob.glob("../dataset/sea_ocean/*")
    img_other = glob.glob("../dataset/other/*")

    for file_name in img_mer:
        img = Image.open(file_name)
        width_size = width_size + img.width
        height_size = height_size + img.height
        counter = counter + 1

    for file_name in img_other:
        img = Image.open(file_name)
        width_size = width_size + img.width
        height_size = height_size + img.height
        counter = counter + 1

    return width_size/counter, height_size/counter

def image_representation(image):
    img = Image.open(image)
    image_resize = np.resize(img, (643, 405))
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

    for i in range(5, 11):
        grid = GridSearchCV(svc, algo_dico['param'], cv=i)
        grid.fit(X_train, Y_train)
        print(i, grid.best_params_)

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


# 5 {'C': 1, 'gamma': 'scale', 'kernel': 'rbf', 'verbose': False}
# 6 {'C': 1, 'gamma': 'scale', 'kernel': 'rbf', 'verbose': False}
# 7 {'C': 1, 'gamma': 'scale', 'kernel': 'rbf', 'verbose': False}
# 8 {'C': 100, 'gamma': 'scale', 'kernel': 'rbf', 'verbose': False}
# 9 {'C': 100, 'gamma': 'scale', 'kernel': 'rbf', 'verbose': False}
# 10 {'C': 100, 'gamma': 'scale', 'kernel': 'rbf', 'verbose': False}


algo_dico_best_param = {
    'algorithm_name': 'SVC',
        'param': {
            'C': 1,
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

def estimate_score(train_data, model):
    X_train = train_data[1]
    Y_train = train_data[0]

    scores =  cross_val_score(model, X_train, Y_train, cv=10)
    score_moyen = scores.mean()
    print(score_moyen)

model = fit_model(label_data(),algo_dico_best_param)

def writter(filename, data, model):
    pred = pred_data(data, model)
    with open("prediction.txt","w") as file :
        for i in range(len(filename)):
            file.write(filename[i] + " " + str(pred[i]) + "\n")


data_test = test_data()
filename = glob.glob("../dataset/testdata/*")

