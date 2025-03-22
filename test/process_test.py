# -*- coding:utf-8 -*-
# @Time:    2025/3/17 20:52
# @Author:  RichardoGu
import unittest

import numpy as np
from ground_motion_tools import read_from_kik
from ground_motion_tools.process import (
    fourier, gm_data_fill, butter_worth_filter, down_sample, length_normalize,
    pga_adjust
)


class ProcessTest(unittest.TestCase):
    ACC_FILE_KIK = "ABSH010011140057.EW2"
    DATA_DIR = "./data/"
    file_path = DATA_DIR + ACC_FILE_KIK
    gm_data, time_step = read_from_kik(file_path)

    def test_gm_data_fill_success(self):
        acc, vel, disp = gm_data_fill(ProcessTest.gm_data)
        self.assertEqual(acc.shape[0], vel.shape[0])
        self.assertEqual(acc.shape[0], disp.shape[0])

    def test_fourier_success(self):
        x, y1, y2 = fourier(ProcessTest.gm_data, ProcessTest.time_step)
        self.assertEqual(x.shape[0], 11900)
        self.assertEqual(y1.shape[0], 11900)
        self.assertEqual(y2.shape[0], 11900)

    def test_butter_worth_bandpath_filter_success(self):
        filtered_gm_data = butter_worth_filter(ProcessTest.gm_data, ProcessTest.time_step, 4, 0.1, 25)
        self.assertEqual(self.gm_data.shape, filtered_gm_data.shape)

    def test_downsample_success(self):
        downsize = down_sample(ProcessTest.gm_data, ProcessTest.time_step, 0.02)
        self.assertEqual(downsize.shape[0], int(ProcessTest.gm_data.shape[0] / int(0.02 / ProcessTest.time_step)))

    def test_length_normalize_success(self):
        normalized_wave = length_normalize(ProcessTest.gm_data, 1000)
        self.assertEqual(normalized_wave.shape[0], 1000)
        normalized_wave = length_normalize(ProcessTest.gm_data, 3000)
        self.assertEqual(normalized_wave.shape[0], 3000)

    def test_pga_adjust_success(self):
        gm_data_many = np.zeros((100, ProcessTest.gm_data.shape[0]))
        for i in range(100):
            gm_data_many[i, :] = ProcessTest.gm_data

        adjusted_wave = pga_adjust(ProcessTest.gm_data, 2.2)
        self.assertEqual(np.abs(adjusted_wave).max(), 2.2)
        adjusted_wave = pga_adjust(gm_data_many, 2.2)
        self.assertEqual(
            np.abs(adjusted_wave).max(axis=1).all(),
            np.array([2.2 for i in range(adjusted_wave.shape[0])]).all()
        )
