import model

from sklearn.svm import LinearSVC

train_data = model.load_labeled_train_data("data/train")
model.compute_representation(train_data, 17)

salsa = LinearSVC(penalty="l2")
score = model.estimate_model_score_average(50, train_data, salsa, 0.2)
print(score)

