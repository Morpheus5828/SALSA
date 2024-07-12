from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_val_score, GridSearchCV
import pickle

classifiers = {
    'LogisticRegression': LogisticRegression(),
    'DecisionTree': DecisionTreeClassifier(),
    'RandomForest': RandomForestClassifier(),
    'SVM': SVC(),
    'AdaBoost': AdaBoostClassifier(algorithm="SAMME"),
    'GradientBoosting': GradientBoostingClassifier(),
}

param_grids = {
    'LogisticRegression': {
        'C': [0.1, 1, 10],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear']
    },
    'DecisionTree': {
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10]
    },
    'RandomForest': {
        'n_estimators': [10, 50, 100],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10]
    },
    'SVM': {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf', 'poly']
    },
    'AdaBoost': {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 1]
    },
    'GradientBoosting': {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 4, 5]
    },
}


class ClassifierRunning:
    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def run(self):
        best_estimators = {}
        for name, clf in classifiers.items():
            print(f"Training {name}...")
            grid_search = GridSearchCV(clf, param_grids[name], cv=5, scoring='accuracy')
            grid_search.fit(self.X_train, self.y_train)
            best_estimators[name] = grid_search.best_estimator_
            print(f"Best parameters for {name}: {grid_search.best_params_}")
            print(f"Ccore for {name}: {grid_search.best_score_}")

        for name, estimator in best_estimators.items():
            print(f"\nEvaluating {name} on test data...")
            if name == "GradientBoosting":
                with open("addD_1652_gradientBoosting.pkl", 'wb') as file:
                    pickle.dump(estimator, file)

            y_pred = estimator.predict(self.X_test)
            print(classification_report(self.y_test, y_pred))
