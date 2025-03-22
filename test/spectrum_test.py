# -*- coding:utf-8 -*-
# @Time:    2025/3/17 20:37
# @Author:  RichardoGu
import unittest

import matplotlib.pyplot as plt
import numpy as np

from ground_motion_tools import read_from_kik
from ground_motion_tools.process import down_sample, pga_adjust
from ground_motion_tools.spectrum import (
    get_spectrum, SPECTRUM_PERIOD, design_spectrum_building,
    match_discrete_periodic_point
)


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

    def test_design_spectrum_building_success(self):
        design_spectrum = []
        for ts in np.arange(0, 6, 0.01):
            design_spectrum.append(design_spectrum_building(float(ts), damping_ratio=0.05))
        design_spectrum = np.array(design_spectrum)
        self.assertEqual(design_spectrum.shape[0], len(np.arange(0, 6, 0.01)))
        # plt.plot(design_spectrum)
        # plt.show()

    def test_match_discrete_periodic_point_success(self):
        gm_data_many = np.zeros((100, SpectrumTest.gm_data.shape[0]))
        for i in range(100):
            gm_data_many[i, :] = SpectrumTest.gm_data
        gm_data_many = pga_adjust(gm_data_many, 0.35)
        match_discrete_periodic_point(
            gm_data_many,
            design_spectrum_building,
            {
                "damping_ratio": 0.05,
                "t_g": 0.35,
                "alpha_max": 0.08 * 9.8
            },
            [0.6, 0.5],
            tremble=[0.2, 0.4]
        )
        self.assertEqual(0, 0)
