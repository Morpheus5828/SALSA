from unittest import TestCase

import matplotlib.pyplot as plt

import salsa.algorithms.ORB as ORB


class testBrisk(TestCase):
    def test_get_fp_descriptors(self):
        path = "../../resources/other/ouie.jpeg"
        kp, des = ORB.get_kp_descriptors(path, draw_kp=True)
        self.assertTrue(len(kp) == 388)
        self.assertTrue(len(des) == 388)