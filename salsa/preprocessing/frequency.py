"""This module contains tools to get image color frequencies
..moduleauthor:: Marius Thorre
"""
import os.path

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def get_frequency(path):
    img = np.array(Image.open(path))
    rgb_values = np.arange(256)
    red_occurencies, blue_occurencies, green_occurencies = np.zeros(256), np.zeros(256), np.zeros(256)
    for k in rgb_values:
        red_occurencies[k] = np.count_nonzero(img[:, :, 0]==k)
        blue_occurencies[k] = np.count_nonzero(img[:, :, 1] == k)
        green_occurencies[k] = np.count_nonzero(img[:, :, 2] == k)
    plt.bar(rgb_values, red_occurencies, color="red")
    plt.bar(rgb_values, blue_occurencies, color="blue")
    plt.bar(rgb_values, green_occurencies, color="green")
    plt.xlabel("RGB 0 to 255")
    plt.ylabel("Frequencies")
    plt.title(f"Frequencies of {os.path.basename(path)} img")
    plt.show()


#get_frequency("C:/Users/thorr/PycharmProjects/DeepSalsa/resources/other/biv8q1.jpeg")