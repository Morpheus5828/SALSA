import os
from PIL import Image

def resize_images(folder_path):
    resize_folder_path = os.path.join(folder_path, "resize")

    smallest_size = float('inf')
    smallest_image_path = None

    # Trouver la plus petite image
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if file_name.endswith(".jpeg") or file_name.endswith(".png") or file_name.endswith(".jpg"):  # Seulement pour les fichiers image
            try:
                with Image.open(file_path) as image:
                    size = image.width * image.height
                    if size < smallest_size:
                        smallest_size = size
                        smallest_image_path = file_path
            except OSError:
                pass

    # Redimensionner toutes les autres images à la taille de la plus petite image et les enregistrer dans le nouveau dossier
    if smallest_image_path is not None:
        if not os.path.exists(resize_folder_path):
            os.makedirs(resize_folder_path)
        smallest_image = Image.open(smallest_image_path)
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)
            if file_name.endswith(".jpeg") or file_name.endswith(".png") or file_name.endswith(".jpg"):
                try:
                    with Image.open(file_path) as image:
                        resized_image = image.resize((smallest_image.width, smallest_image.height))
                        resized_image_path = os.path.join(resize_folder_path, file_name)
                        resized_image.save(resized_image_path)
                except OSError:
                    pass
    else:
        print("Aucune image trouvée.")


resize_images("dataset/other")
