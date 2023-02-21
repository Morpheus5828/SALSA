import math
import os
import threading
import time

import cv2
import numpy as np
from threading import Thread
from matplotlib import pyplot as plt

sea_ocean = os.listdir("../dataset/sea_ocean/")
img = sea_ocean[1]  # 838s.jpg

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

class Distance(Thread):
    def __int__(self, image_kp_dico, keypoint):
        self.image_kp_dico = image_kp_dico
        self.keypoint = keypoint
        self.dico = 0

        Thread.__init__(self)

    def get_dico(self):
        return self.dico

    def set_dico(self, value):
        self.dico = value

    def run(self):
        self.set_dico(get_kp_distance(self.image_kp_dico, self.keypoint))


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


def evaluate(kp1, kp2):
    counter = 0
    for i in kp1:
        if i in kp2:
            counter +=1
    print("size: ", len(kp1))
    print("size: ", len(kp2))
    print("counter:", counter)
    '''common = np.intersect1d(kp1, kp2).size
    print("common: ", common)'''
    return (100 * counter) // len(kp2)


def main(img):
    start = time.time()
    # 1. extract kp from img
    init_kp_array = get_keypoint(detect_keypoint(50, img))

    # 2. extract kp from each image in repo
    image_kp_dico = {}
    for index in range(1):
        arr = calculate_kp(sea_ocean[index])
        image_kp_dico[sea_ocean[index]] = arr

    # write_result_into_file("image_kp_array.txt", image_kp_dico[sea_ocean[0]])

    # 3. init len(img) threads
    '''t = Distance()
    t.image_kp_dico = image_kp_dico.get("838s.jpg")
    t.keypoint = init_kp_array
    t.start()
    t.join()'''
    #for v in t.get_dico():

    print("Evaluation: ", evaluate(init_kp_array, image_kp_dico.get("838s.jpg")), "%")
    #print("Evaluation: ", evaluate(image_kp_dico.get("838s.jpg"), image_kp_dico.get("838s.jpg")), "%")

    end = time.time()
    print("\n Execution time:", end - start)


main(img)
# img2 = cv2.drawKeypoints(img, kp, None, flags=0)

# cv2.imshow('corne', img2)
# cv2.waitKey(0)
