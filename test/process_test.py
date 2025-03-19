# -*- coding:utf-8 -*-
# @Time:    2025/3/17 20:52
# @Author:  RichardoGu
import unittest

from ground_motion_tools import read_from_kik, fourier, gm_data_fill, butter_worth_filter, down_sample, length_normalize


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
