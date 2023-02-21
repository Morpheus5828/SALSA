import math
import os
import threading
import time

import cv2
import numpy as np
from threading import Thread
from matplotlib import pyplot as plt

sea_ocean = os.listdir("../dataset/sea_ocean/")
img = sea_ocean[0]  # 838s.jpg

img = cv2.imread("../dataset/sea_ocean/" + img)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
gray = np.float32(gray)

list_distance = []


############################# class

class Vector:
    def __int__(self, source, destination, normal):
        self.source = source
        self.destination = destination
        self.normal = normal

    def get_normal(self):
        return self.normal

    def display(self):
        return "source: ", self.source, " to: ", self.destination, " size: ", self.normal

############################# function


def get_keypoint(kp):
    return np.asarray([key_point.pt for key_point in kp])


def detect_keypoint(n, image):
    return cv2.FastFeatureDetector_create(n).detect(image, None)


def get_distance(x1, y1, x2, y2):
    return int(math.sqrt(math.pow(x2 - x1, 2) + math.pow(y2 - y1, 2)))


def calculate_kp(image):
    image = cv2.imread("../dataset/sea_ocean/" + image)
    kp = detect_keypoint(50, image)
    return get_keypoint(kp)


def get_kp_distance(dico_a, dico_b):
    distance_list = []
    for source in dico_a:
        for destination in dico_b:
            v = Vector()
            v.source = source
            v.destination = destination
            v.normal = get_distance(source[0], source[1], destination[0], destination[1])
            distance_list.append(v)

    return distance_list


def write_result_into_file(filename, body):
    file = open(filename, "w")
    for element in body: file.write(str(element) + "\n")
    file.close()


def display_images_with_score(img1, img2, score):
    _, axis = plt.subplots(ncols=2, figsize=(12, 3))
    axis[0].imshow(img1)
    axis[1].imshow(img2)
    # axis[1].set_title("Score: ", score)
    plt.show()


def get_cv2_img(path):
    return cv2.imread(path)

def evaluate(kp1, kp2):
    counter = 0
    for i in kp1:
        if i in kp2:
            counter += 1
    print("size: ", len(kp1))
    print("size: ", len(kp2))
    print("counter:", counter)
    '''common = np.intersect1d(kp1, kp2).size
    print("common: ", common)'''
    return (100 * counter) // len(kp2)


def compare(img_source, img_dest):
    start = time.time()

    # 0. convert image to cv2 image
    img_source = get_cv2_img(img_source)
    img_dest = get_cv2_img(img_dest)

    # 1. extract kp from img
    img_source_kp = get_keypoint(detect_keypoint(50, img_source))

    # 2. extract kp from other random image in repo
    img_dest_kp = get_keypoint(detect_keypoint(50, img_dest))

    # 3. evaluation
    evaluation = evaluate(img_source_kp, img_dest_kp)

    print("Evaluation: ", evaluation, "%")

    display_images_with_score(img, img, evaluation)

    end = time.time()
    print("\n Execution time:", end - start)


compare("../dataset/sea_ocean/838s.jpg", "../dataset/sea_ocean/838s.jpg")





# img2 = cv2.drawKeypoints(img, kp, None, flags=0)

# cv2.imshow('corne', img2)
# cv2.waitKey(0)
