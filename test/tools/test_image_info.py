from unittest import TestCase

import sys, os
import numpy as np
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '../../'))
if project_root not in sys.path:
    sys.path.append(project_root)
import salsa.tools.image_tools as image_tools

dataset_path = os.path.join(project_root, "resources/dataset")


class testImage_info(TestCase):
    def test_normalize_img(self):
        data, label = image_tools.normalize_img(
            path=dataset_path,
            width=100,
            height=100
        )
        self.assertTrue(data.shape == (413, 100, 100))
        self.assertTrue(label.shape == (413,))

    def test_data_augmentation(self):
        Xdata, ydata = image_tools.normalize_img(
            path=dataset_path,
            height=100,
            width=100
        )
        data, label = image_tools.data_augmentation(
            all_images=Xdata,
            all_labels=ydata,
        )

        self.assertTrue(data.shape == (1239, 100, 100))
        self.assertTrue(label.shape == (1239,))
