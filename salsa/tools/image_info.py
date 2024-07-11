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


def get_frequency(path):
    img = np.array(Image.open(path))
    rgb_values = np.arange(256)
    red_occurencies, blue_occurencies, green_occurencies = np.zeros(256), np.zeros(256), np.zeros(256)
    for k in rgb_values:
        red_occurencies[k] = np.count_nonzero(img[:, :, 0] == k)
        blue_occurencies[k] = np.count_nonzero(img[:, :, 1] == k)
        green_occurencies[k] = np.count_nonzero(img[:, :, 2] == k)
    plt.bar(rgb_values, red_occurencies, color="red")
    plt.bar(rgb_values, blue_occurencies, color="blue")
    plt.bar(rgb_values, green_occurencies, color="green")
    plt.xlabel("RGB 0 to 255")
    plt.ylabel("Frequencies")
    plt.title(f"Frequencies of {os.path.basename(path)} img")
    plt.show()




