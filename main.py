import os
import time

from image_detection import image_compare as ic
#from image_detection import create_evaluate_file as cev

sea_ocean = os.listdir("dataset/sea_ocean")


def main(source):
    start = time.time()
    score = {}
    for image in sea_ocean:
        destination = "dataset/sea_ocean/" + str(image)
        value = ic.compare(source, destination)
        if value is not None: score[value] = image

    score = dict(sorted(score.items()))
    body = ""
    for i in score.keys():
        #body += "key: ", i, " value: ", score.get(i), "\n"
        print("key: ", i, " value: ", score.get(i))


    end = time.time()
    print("\n Execution time:", end - start)


main("dataset/sea_ocean/image_web.jpg")

