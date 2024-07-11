"""This module contains demo to show example of an image frequency graph
..moduleauthor:: Marius THORRE
"""
import sys, os

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '../'))
if project_root not in sys.path:
    sys.path.append(project_root)
from salsa.tools.image_info import get_frequency

if __name__ == "__main__":
    img_path = os.path.join(project_root, "resources/dataset/ocean/838s.jpg")
    get_frequency(path=img_path)
