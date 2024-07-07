from unittest import TestCase

import matplotlib.pyplot as plt

import salsa.algorithms.BRISK as BRISK


class testBrisk(TestCase):
    def test_get_fp_descriptors(self):
        path = "../../resources/other/ouie.jpeg"
        kp, des = BRISK.get_kp_descriptors(path, draw_kp=True)
        self.assertTrue(len(kp) == 370)
        self.assertTrue(len(des) == 370)