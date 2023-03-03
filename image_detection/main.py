import os
import time

from image_detection import image_compare as ic

sea_ocean = os.listdir("../dataset/sea_ocean")
other = os.listdir("../dataset/other")


def main():
    start = time.time()
    score = {}
    for source in sea_ocean:
        file_name = source
        source = "../dataset/sea_ocean/" + str(source)
        for image in sea_ocean:
            destination = "../dataset/sea_ocean/" + str(image)
            value = ic.compare(source, destination)
            if value is not None: score[value] = image
        for image in other:
            destination = "../dataset/other/" + str(image)
            value = ic.compare(source, destination)
            if value is not None: score[value] = image
        score = dict(sorted(score.items()))
        file = open("../evaluation/keypoint_compare/" + file_name + ".txt", "a")
        for i in score.keys():
            line = str(i) + "% for " + str(score.get(i))
            file.write("\n" + line)
        file.close()

    end = time.time()
    print("\n Execution time:", end - start)


main()

