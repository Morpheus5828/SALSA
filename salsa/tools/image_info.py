"""This module contains tools to edit image from dataset
..moduleauthor:: Marius Thorre
"""

import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def img_folder_sizes_infos(
        path: str,
        metric: list,
        plot_img_size: bool = False,
        nb_img_max: int = sys.maxsize
) -> dict:
    """ Return statistic img size
    :param path: path of the dataset
    :param metric: meaning, Q1 Q3, min, max
    :param plot_img_size: display or not img size with metrics
    :param nb_img_max: maximum number of image dataset
    :return: metric information as dict
    """
    width = []
    height = []
    c = 0
    for folder in os.listdir(path):
        for image in os.listdir(path + "/" + folder):
            if c <= nb_img_max:
                img = np.array(Image.open(path + "/" + folder + "/" + image))
                width.append(img.shape[0])
                height.append(img.shape[1])
                if plot_img_size:
                    plt.scatter(img.shape[0], img.shape[1], c="#3e5af4")
                    plt.title("Display image dimension")
                    plt.xlabel("Width")
                    plt.ylabel("Height")
                c += 1
    result = {}
    for m in metric:
        if m == "average":
            plt.scatter(np.mean(width), np.mean(height), c="black", label="Average")
            result["average"] = (np.mean(width), np.mean(height))
        elif m == "Q1":
            plt.scatter(np.percentile(width, 25), np.percentile(height, 25), c="#df397f", label="Q1")
            result["Q1"] = (np.percentile(width, 25), np.percentile(height, 25))
        elif m == "Q3":
            plt.scatter(np.percentile(width, 75), np.percentile(height, 75), c="#df3939", label="Q3")
            result["Q3"] = (np.percentile(width, 75), np.percentile(height, 75))
        elif m == "min":
            result["min"] = (np.min(width), np.min(height))
        elif m == "max":
            result["max"] = (np.max(width), np.max(height))
        else:
            print("Metric not recognized, please read the doc")
    if plot_img_size:
        plt.legend()
        plt.show()
    return result


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
            cX, cY = width // 2, height // 2
            M = cv2.getRotationMatrix2D((cX, cY), angle=45, scale=1.0)
            rotated = cv2.warpAffine(img, M, (width, height))
            img_normalise.append(rotated)
            labels.append(l)
            cX, cY = width // 2, height // 2
            M = cv2.getRotationMatrix2D((cX, cY), angle=90, scale=1.0)
            rotated = cv2.warpAffine(img, M, (width, height))
            img_normalise.append(rotated)
            labels.append(l)
            cX, cY = width // 2, height // 2
            M = cv2.getRotationMatrix2D((cX, cY), angle=135, scale=1.0)
            rotated = cv2.warpAffine(img, M, (width, height))
            img_normalise.append(rotated)
            labels.append(l)
            cX, cY = width // 2, height // 2
            M = cv2.getRotationMatrix2D((cX, cY), angle=180, scale=1.0)
            rotated = cv2.warpAffine(img, M, (width, height))
            img_normalise.append(rotated)
            labels.append(l)
    return np.array(img_normalise), np.array(labels)


