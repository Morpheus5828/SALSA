import glob

from PIL import Image
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_val_score

def image_representation(image):
    img = Image.open(image)
    grey_img = img.convert('L')
    image_resize = np.resize(grey_img, (643,405))
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

    for i in range(5,11):
        grid = GridSearchCV(svc, algo_dico['param'], cv=i)
        grid.fit(X_train,Y_train)
        print( i, grid.best_params_)



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
# 6 {'C': 10, 'gamma': 'scale', 'kernel': 'rbf', 'verbose': False}
# 7 {'C': 100, 'gamma': 'scale', 'kernel': 'sigmoid', 'verbose': False}
# 8 {'C': 100, 'gamma': 'scale', 'kernel': 'sigmoid', 'verbose': False}
# 9 {'C': 100, 'gamma': 'scale', 'kernel': 'sigmoid', 'verbose': False}
# 10 {'C': 100, 'gamma': 'scale', 'kernel': 'sigmoid', 'verbose': False}


algo_dico_best_param = {
    'algorithm_name': 'SVC',
        'param': {
            'C': 1,
            'kernel': 'rbf',
            'gamma': 'scale',
            'verbose': False
        }
    }
algo_dico_best_param2 = {
    'algorithm_name': 'SVC',
        'param': {
            'C': 10,
            'kernel': 'rbf',
            'gamma': 'scale',
            'verbose': False
        }
    }

algo_dico_best_param3 = {
    'algorithm_name': 'SVC',
        'param': {
            'C': 1,
            'kernel': 'sigmoid',
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

def estimate_score(train_data, model) :
    X_train = train_data[1]
    Y_train = train_data[0]

    for i in range(5, 11):
        if i == 5:
            scores = cross_val_score(model_1, X_train, Y_train, cv=i)
            score_moyen = scores.mean()
            print(i, score_moyen)
        elif i == 6:
            scores = cross_val_score(model_2, X_train, Y_train, cv=i)
            score_moyen = scores.mean()
            print(i, score_moyen)
        else:
            scores = cross_val_score(model_3, X_train, Y_train, cv=i)
            score_moyen = scores.mean()
            print(i, score_moyen)

model_1 = fit_model(label_data(),algo_dico_best_param)
model_2 = fit_model(label_data(), algo_dico_best_param2)
model_3 = fit_model(label_data(), algo_dico_best_param3)

def writter(filename, data, model):
    pred = pred_data(data, model)
    with open("prediction.txt","w") as file :
        for i in range(len(filename)):
            file.write(filename[i] + " " + str(pred[i]) + "\n")



data_test = test_data()
filename = glob.glob("../dataset/testdata/*")

estimate_score(traindata,model_1)