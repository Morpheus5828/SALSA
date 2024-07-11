from unittest import TestCase

import matplotlib.pyplot as plt

import salsa.algorithms.SIFT as SIFT


class testSift(TestCase):
    def test_get_fp_descriptors(self):
        path = "../../resources/dataset/other/ouie.jpeg"
        kp, des = SIFT.get_kp_descriptors(path, draw_kp=True)
        self.assertTrue(len(kp) == 311)
        self.assertTrue(len(des) == 311)
        # _, _ = SIFT.get_kp_descriptors(path, draw_kp=True, filter_color=0)
        # _, _ = SIFT.get_kp_descriptors(path, draw_kp=True, filter_color=1)
        # _, _ = SIFT.get_kp_descriptors(path, draw_kp=True, filter_color=2)
        # _, _ = SIFT.get_kp_descriptors(path, draw_kp=True, filter_color=3)
        # _, _ = SIFT.get_kp_descriptors(path, draw_kp=True, filter_color=4)
