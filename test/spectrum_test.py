# -*- coding:utf-8 -*-
# @Time:    2025/3/17 20:37
# @Author:  RichardoGu
import time
import unittest
import numpy as np
from ground_motion_tools import read_from_kik
from ground_motion_tools.process import down_sample, pga_adjust
from ground_motion_tools.spectrum import (
    get_spectrum,
    SPECTRUM_PERIOD,
    design_spectrum_building,
    design_spectrum_bridge,
)


class SpectrumTest(unittest.TestCase):
    ACC_FILE_KIK = "ABSH010011140057.EW2"
    DATA_DIR = "./resource/"
    file_path = DATA_DIR + ACC_FILE_KIK
    gm_data, time_step = read_from_kik(file_path)
    gm_data = down_sample(gm_data, time_step, 0.02)
    time_step = 0.02

    def test_get_spectrum_success(self):
        gm_data = pga_adjust(SpectrumTest.gm_data, 0.35)
        _, _, _, pse_acc_spectrum, _ = get_spectrum(
            gm_data, SpectrumTest.time_step, calc_opt=0
        )
        # plt.plot(SPECTRUM_PERIOD, pse_acc_spectrum)
        # plt.show()
        self.assertEqual(pse_acc_spectrum.shape[0], len(SPECTRUM_PERIOD))

        batch_size = 1000
        gm_data_many = np.zeros((batch_size, SpectrumTest.gm_data.shape[0]))
        for i in range(batch_size):
            gm_data_many[i, :] = gm_data

        time0 = time.time()
        _, _, _, pse_acc_spectrum_many, _ = get_spectrum(
            gm_data_many, SpectrumTest.time_step, calc_opt=0
        )
        time1 = time.time()
        _, _, _, pse_acc_spectrum_many, _ = get_spectrum(
            gm_data_many, SpectrumTest.time_step, calc_opt=1
        )
        time2 = time.time()
        print(f"batch_size: {batch_size}, cost time: {time1 - time0}")
        print(f"batch_size: {batch_size}, cost time: {time2 - time1}")
        print(f"Speedup: {(time1 - time0) / (time2 - time1):.2f}x")
        # plt.plot(SPECTRUM_PERIOD, pse_acc_spectrum_many[0, :])
        # plt.plot(SPECTRUM_PERIOD, pse_acc_spectrum)
        # plt.show()
        self.assertEqual(time1 - time0 > time2 - time1, True)
        self.assertEqual(
            np.allclose(pse_acc_spectrum_many[0, :], pse_acc_spectrum), True
        )
        self.assertEqual(pse_acc_spectrum_many.shape[0], batch_size)
        self.assertEqual(pse_acc_spectrum_many.shape[1], len(SPECTRUM_PERIOD))

    def test_design_spectrum_building_success(self):
        design_spectrum = []
        for ts in np.arange(0, 6, 0.01):
            design_spectrum.append(
                design_spectrum_building(float(ts), damping_ratio=0.05)
            )
        design_spectrum = np.array(design_spectrum)
        self.assertEqual(design_spectrum.shape[0], len(np.arange(0, 6, 0.01)))
        # plt.plot(design_spectrum)
        # plt.show()

    def test_design_spectrum_bridge_success(self):
        design_spectrum = []
        for ts in np.arange(0, 6, 0.01):
            design_spectrum.append(
                design_spectrum_bridge(float(ts), damping_ratio=0.05)
            )
        design_spectrum = np.array(design_spectrum)
        self.assertEqual(design_spectrum.shape[0], len(np.arange(0, 6, 0.01)))
        # plt.plot(design_spectrum)
        # plt.show()
