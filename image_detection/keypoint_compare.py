import os
import time
import image_compare as ic

sea_ocean = os.listdir("../dataset/sea_ocean")
other = os.listdir("../dataset/other")

# by Marius THORRE

'''
This file is used to compare each keypoint
'''


def main():
    start = time.time()
    for source in other:
        score = {}
        file_name = source
        source = "../dataset/other/" + str(source)
        for image in sea_ocean:
            destination = "../dataset/sea_ocean_to_all/" + str(image)
            value = ic.compare(source, destination)
            if value is not None: score[value] = image + " from sea_ocean_to_all"
        for image in other:
            destination = "../dataset/other/" + str(image)
            value = ic.compare(source, destination)
            if value is not None: score[value] = image + " from other"
        score = dict(sorted(score.items(), reverse=True))
        file = open("../evaluation/keypoint_compare/other/" + file_name + ".txt", "a")

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
