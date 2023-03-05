import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import random
from sift_detection.sift_detection import extractSIFTInformations as siftInfo
import pylab as pl
from sklearn.metrics import confusion_matrix, accuracy_score

sea_ocean = os.listdir("../dataset/sea_ocean")

list_kp_desc = []

for image in sea_ocean:
    source = cv2.imread("../dataset/other/" + str(image))
    kp, desc = siftInfo(source)
    list_kp_desc.append((kp, desc))

descriptors = list_kp_desc[0][1]
for image, descriptor in list_kp_desc[:1]:
    descriptors = np.vstack((descriptors,descriptor))

print(descriptors.shape)