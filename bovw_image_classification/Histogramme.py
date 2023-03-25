import cv2
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as pltfrom
from scipy.cluster.vq import kmeans, vq


# take the aba13.jpeg image example in report

image = cv2.imread("../dataset/sea_ocean/without_changes/aba13.jpeg", cv2.COLOR_BGR2GRAY)

descriptor = []

SIFT = cv2.SIFT_create()

kp, desc = SIFT.detectAndCompute(image, None)
descriptor.append(desc)

all_desc = []
for img_desc in descriptor:
    for desc in img_desc:
        all_desc.append(desc.astype('float'))

all_desc = np.stack(all_desc)

codebook, variance = kmeans(all_desc, 4, 1)  # 4 clusters

visual_words = []
for img_desc in descriptor:
    img_visual_words, distance = vq(img_desc, codebook)
    visual_words.append(img_visual_words)
frequency_vectors = []
for img_visual_words in visual_words:
    img_frequency_vector = np.zeros(4)
    for word in img_visual_words:
        img_frequency_vector[word] += 1
    frequency_vectors.append(img_frequency_vector)
frequency_vectors = np.stack(frequency_vectors)
print(all_desc.shape)
print(frequency_vectors)

plt.bar(list(range(4)), frequency_vectors[0])
plt.show()












