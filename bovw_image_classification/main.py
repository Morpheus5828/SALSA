from bovw_image_classification import k_means
from bovw_image_classification.k_means import *


def get_score(y_test, y_pred):
    return accuracy_score(y_test, y_pred)

def evaluate():
    training_img, all_label = init_training_data()
    all_desc, descriptors = extract_feature(training_img, all_label)
    frequency_vectors = set_up_BOVW(descriptors, all_desc)
    X_train, X_test, y_train, y_test = train_test_split(frequency_vectors, all_label, test_size=0.2)

    #  1°) Gaussian
    result = []
    for i in range(50):
        gnb = GaussianNB()
        y_pred = gnb.fit(X_train, y_train).predict(X_test)
        print(get_score(y_test, y_pred))
        result.append(get_score(y_test, y_pred))



evaluate()