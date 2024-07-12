"""This module contains code to check our bovw model on real data
..moduleauthor:: Marius THORRE
"""

import os, sys

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '../'))
if project_root not in sys.path:
    sys.path.append(project_root)
import pickle
from salsa.tools.image_tools import normalize_img
from salsa.method.bovw import Bag_Of_Visual_Words
from sklearn.metrics import classification_report, accuracy_score

if __name__ == "__main__":
    width, height = 600, 600
    dataset_path = os.path.join(project_root, "resources", "real_image")
    models_path = [
        "without_addD_gradientBoosting.pkl",
        "addD_1239_gradientBoosting.pkl",
        "addD_1652_gradientBoosting.pkl",
    ]
    Xdata, ydata = normalize_img(
        path=dataset_path,
        height=height,
        width=width
    )
    for model_file_path in models_path:
        print(f"Model used: {model_file_path}")
        with open(os.path.join(project_root, "resources", "model", model_file_path), 'rb') as file:
            model = pickle.load(file)
        print(Xdata.shape, ydata.shape)
        tfidf, ydata_update = Bag_Of_Visual_Words(
            X_data=Xdata,
            y_data=ydata,
            extract_method="SIFT"
        )

        y_pred = model.predict(tfidf)

        print(classification_report(y_pred, ydata_update))




