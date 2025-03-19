# -*- coding:utf-8 -*-
# @Time:    2025/3/17 20:37
# @Author:  RichardoGu
import unittest

import numpy as np

from ground_motion_tools import read_from_kik, down_sample
from spectrum import get_spectrum, SPECTRUM_PERIOD


class SpectrumTest(unittest.TestCase):
    ACC_FILE_KIK = "ABSH010011140057.EW2"
    DATA_DIR = "./data/"
    file_path = DATA_DIR + ACC_FILE_KIK
    gm_data, time_step = read_from_kik(file_path)
    gm_data = down_sample(gm_data, time_step, 0.02)
    time_step = 0.02

    def test_get_spectrum_success(self):
        acc_spectrum, vel_spectrum, disp_spectrum, _, _ = get_spectrum(SpectrumTest.gm_data, SpectrumTest.time_step)
        self.assertEqual(acc_spectrum.shape[0], len(SPECTRUM_PERIOD))

        gm_data_many = np.zeros((100, SpectrumTest.gm_data.shape[0]))
        for i in range(100):
            gm_data_many[i, :] = SpectrumTest.gm_data
        acc_spectrum, vel_spectrum, disp_spectrum, _, _ = get_spectrum(gm_data_many, SpectrumTest.time_step)
        self.assertEqual(acc_spectrum.shape[0], 100)
        self.assertEqual(acc_spectrum.shape[1], len(SPECTRUM_PERIOD))
