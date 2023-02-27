import os
from PIL import Image


def resize_images_by(folder_path, method, value):
    resize_folder_path = os.path.join(folder_path,"resize "+method +" "+str(value))

    if method == "multiply":
        if not os.path.exists(resize_folder_path):
            os.makedirs(resize_folder_path)
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)
            if file_name.endswith(".jpeg") or file_name.endswith(".png") or file_name.endswith(".jpg"):
                try:
                    with Image.open(file_path) as image:
                        resized_image = image.resize((image.width * value, image.height * value))
                        resized_image_path = os.path.join(resize_folder_path, file_name)
                        resized_image.save(resized_image_path)
                except OSError:
                    pass

    if method == "divide":
        if not os.path.exists(resize_folder_path):
            os.makedirs(resize_folder_path)
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)
            if file_name.endswith(".jpeg") or file_name.endswith(".png") or file_name.endswith(".jpg"):
                try:
                    with Image.open(file_path) as image:
                        resized_image = image.resize((int(image.width / value), int(image.height / value)))
                        resized_image_path = os.path.join(resize_folder_path, file_name)
                        resized_image.save(resized_image_path)
                except OSError:
                    pass


resize_images_by("../dataset/other", "divide", 3)
resize_images_by("../dataset/sea_ocean", "multiply", 3)
