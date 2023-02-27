import os
from PIL import Image


def resize_images(folder_path):
    resize_folder_path = os.path.join(folder_path, "average_resize")

    width_size = float(0)
    height_size = float(0)

    # Trouver la taille moyenne
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if file_name.endswith(".jpeg") or file_name.endswith(".png") or file_name.endswith(".jpg"):  # Seulement pour les fichiers image
            try:
                with Image.open(file_path) as image:
                    width_size = width_size + image.width
                    height_size = height_size + image.height
            except OSError:
                pass

    width_avg = width_size/len(os.listdir(folder_path))
    height_avg = height_size / len(os.listdir(folder_path))

    # Redimensionner toutes les autres images à la taille moyenne et les enregistrer dans le nouveau dossier
    if not os.path.exists(resize_folder_path):
        os.makedirs(resize_folder_path)
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if file_name.endswith(".jpeg") or file_name.endswith(".png") or file_name.endswith(".jpg"):
            try:
                with Image.open(file_path) as image:
                    resized_image = image.resize((int(width_avg), int(height_avg)))
                    resized_image_path = os.path.join(resize_folder_path, file_name)
                    resized_image.save(resized_image_path)
            except OSError:
                pass



