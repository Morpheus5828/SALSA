import model

from sklearn.svm import LinearSVC

# TODO : complete test image evaluation


train_data = model.load_labeled_train_data("data/train")
model.compute_representation(train_data, 17)

salsa = LinearSVC(penalty="l2")
model.learn_model_from_data(train_data, salsa)

test_data = model.load_test_data("data/test/CC2/all")
model.compute_representation(test_data)
