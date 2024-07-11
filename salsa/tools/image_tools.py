"""This module contains toolkit code for image processing
..moduleauthor:: Marius THORRE
"""

import cv2
import numpy as np
import os
import cv2
from PIL import Image
import random
from sklearn.cluster import KMeans

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '../../'))
if project_root not in sys.path:
    sys.path.append(project_root)

def normalize_img(
        path: str,
        width: int,
        height: int
):
    img_normalise = []
    labels = []
    for l, folder in enumerate(os.listdir(path)):
        for image in os.listdir(os.path.join(path, folder)):
            img = cv2.imread(os.path.join(path, folder, image))
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            img = cv2.resize(img, dsize=(width, height))
            img_normalise.append(img)
            labels.append(l)
    return np.array(img_normalise), np.array(labels)


def data_augmentation(
        all_images: np.ndarray,
        all_labels: np.ndarray,
) -> tuple:
    """
    Data augmentation with adding image rotation
    :param all_images: images dataset must be same width and height
    :param all_labels: images labels
    :return: all_images and their rotation
    """
    images = []
    labels = []

    for l, img in enumerate(all_images):
        images.append(cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE))
        labels.append(all_labels[l])
        images.append(cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE))
        labels.append(all_labels[l])
        images.append(cv2.rotate(img, cv2.ROTATE_180))
        labels.append(all_labels[l])

    return np.array(images), np.array(labels)


def check_KMeans_inertia(training_data: np.ndarray) -> None:
    inertia = []
    K_range = np.arange(1, 200, 10)
    for n in tqdm(K_range):
        model = KMeans(n_clusters=n, n_init=1)
        model.fit(training_data)
        inertia.append(model.inertia_)

    plt.plot(K_range, inertia)
    plt.show()


def get_codebook(training_data: np.ndarray, k_cluster: int) -> np.ndarray:
    k = k_cluster
    kmeans = KMeans(n_clusters=k).fit(training_data)
    codebook = kmeans.cluster_centers_
    return codebook