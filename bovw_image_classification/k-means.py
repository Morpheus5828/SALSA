import math
import os

import numpy as np
from scipy.cluster.vq import kmeans, vq
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
from sklearn.svm import LinearSVC

from sift_detection.sift_detection import extract_SIFT_descriptors

sea_ocean = os.listdir("../dataset/sea_ocean")

# extract SIFT descriptor

des_list = []

for image in sea_ocean:
    descriptor, img = extract_SIFT_descriptors("../dataset/sea_ocean/" + image)
    des_list.append(("../dataset/sea_ocean/" + image, descriptor))

# stock all descriptors in numpy array

descriptors = des_list[0][1]
for image, desc in des_list[1:]:
    descriptors = np.vstack((descriptors, desc))

# k-means part
descriptors = descriptors.astype(float)  # convert all descriptors to float type

k = int(math.log(len(sea_ocean)))
voc, variance = kmeans(descriptors, k, 1)

# Calculate the histogram of features and represent them as vector

im_features = np.zeros((len(sea_ocean), k), "float32")
for i in range(len(sea_ocean)):
    words, distance = vq(des_list[i][1], voc)
    for w in words:
        im_features[i][w] += 1

# k-means part

kmeans_obj = KMeans(n_clusters=int(math.log(len(sea_ocean))))
label = kmeans_obj.fit_predict(descriptors)  # do prediction on learning dataset
u_labels = np.unique(label)

# plotting the results:

for i in u_labels:
    print(descriptors)
    print(descriptors[label == i, 0], descriptors[label == i, 1])
    plt.scatter(descriptors[i, 0], descriptors[i, 1])
plt.legend()
#plt.show()