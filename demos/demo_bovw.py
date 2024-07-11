"""This module contains an example of Bag-of-Visual-Words method
..moduleauthor::Marius THORRE
"""
import sys, os

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '../'))
if project_root not in sys.path:
    sys.path.append(project_root)

import os
import numpy as np
from scipy.cluster.vq import vq
from sklearn.model_selection import train_test_split
from salsa.tools.image_info import img_folder_sizes_infos
from salsa.tools.image_tools import normalize_img, data_augmentation
from salsa.method.bovw import Bag_Of_Visual_Words
from salsa.learning.classifiers import ClassifierRunning
import salsa.algorithms.SIFT as SIFT

if __name__ == "__main__":
    dataset_path = os.path.join(project_root, "resources", "../resources/dataset")

    infos = img_folder_sizes_infos(
        path=dataset_path,
        metric=["average"]
    )
    width, height = infos["average"]

    print("Normalization ..")
    Xdata, ydata = normalize_img(path=dataset_path, height=int(height), width=int(width))
    print("Data augmentation ..")
    Xdata, ydata = data_augmentation(Xdata, ydata, int(width), int(height))

    tfidf = Bag_Of_Visual_Words(Xdata=Xdata, ydata=ydata)

    X_train, X_test, y_train, y_test = train_test_split(
        tfidf,
        ydata,
        test_size=0.30,
        random_state=42
    )

    ClassifierRunning(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test
    ).run()
