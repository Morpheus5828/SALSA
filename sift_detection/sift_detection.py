import os
import sys
import time

import numpy as np
import cv2
from resizer import resize_images as ri
from matplotlib import pyplot as plt

sea_ocean = os.listdir("../dataset/sea_ocean")
other = os.listdir("../dataset/other")

def compare(img1, img2):
    img1 = cv2.imread(img1)
    img2 = cv2.imread(img2)

    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()

    keypoints_1, descriptors_1 = sift.detectAndCompute(img1,None)
    keypoints_2, descriptors_2 = sift.detectAndCompute(img2,None)

    bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)

    matches = bf.match(descriptors_1,descriptors_2)
    matches = sorted(matches, key=lambda x:x.distance)

    keypoints_1 = np.asarray([key_point.pt for key_point in keypoints_1])
    keypoints_2 = np.asarray([key_point.pt for key_point in keypoints_2])

    '''print("kp1: ", len(descriptors_1))
    print("kp2: ", len(descriptors_2))'''

    omega = np.concatenate((descriptors_1, descriptors_2))
    omega = np.unique(omega, axis=0)

    '''print("omega: ", len(omega))
    print("matches: ", len(matches))

    print((100*len(matches)) / len(omega))'''
    return (100*len(matches)) // len(omega)


#compare("../dataset/sea_ocean/838s.jpg", "../dataset/sea_ocean/838s.jpg")

def main():
    start = time.time()
    for source in sea_ocean:
        score = {}
        file_name = source
        source = "../dataset/sea_ocean/" + str(source)
        for image in sea_ocean:
            destination = "../dataset/sea_ocean/" + str(image)
            value = compare(source, destination)
            if value is not None: score[value] = image + " from sea_ocean"
        for image in other:
            destination = "../dataset/other/" + str(image)
            value = compare(source, destination)
            if value is not None: score[value] = image + " from other"
        score = dict(sorted(score.items(), reverse=True))
        file = open("../evaluation/sift_compare/" + file_name + ".txt", "a")

        c = 0
        for i in score.keys():
            if c == 30:
                break
            line = str(i) + "% for " + str(score.get(i))
            file.write(line + "\n")
            c += 1
        file.close()

    end = time.time()
    print("\n Execution time:", end - start)


main()
#img3 = cv2.drawMatches(img1, keypoints_1, img2, keypoints_2, matches[:50], img2, flags=2)
#plt.imshow(img3),plt.show()