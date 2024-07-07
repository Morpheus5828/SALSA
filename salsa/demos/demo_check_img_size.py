"""This module contains an example to see all images size in dataset folder
..moduleauthor::Marius THORRE
"""
import sys, os
from salsa.preprocessing.image_info import img_folder_sizes_infos
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '../../'))
if project_root not in sys.path:
    sys.path.append(project_root)


if __name__ == "__main__":
    dataset_path = os.path.join(project_root, "resources", "dataset")

    infos = img_folder_sizes_infos(
        path=dataset_path,
        metric=["average", "Q1", "Q3", "min", "max"],
        plot_img_size=True
    )
    print(infos)