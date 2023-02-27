import os


evaluation = os.listdir("../dataset/evaluation")


def create_txt(title, body):
    file = open("../dataset/evaluation/", title)
    file.write(body)
    file.close()


