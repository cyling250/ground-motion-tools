# -*- coding:utf-8 -*-
# @Time:    2025/3/17 20:37
# @Author:  RichardoGu
"""
im test
"""
import unittest

import numpy as np

from enums import GMIMEnum
from ground_motion_tools import read_from_kik, down_sample, GMIntensityMeasures

GM_DATA, TIME_STEP = read_from_kik("../../../severest-fragility/ABSH010011140057.EW2")
GM_DATA = down_sample(GM_DATA, TIME_STEP, 0.02)
TIME_STEP = 0.02

IM_WITHOUT_SPECTRUM = [GMIMEnum.PGA, GMIMEnum.PGV, GMIMEnum.PGD]
IM_SPECTRUM = [GMIMEnum.ASI, GMIMEnum.HI, GMIMEnum.VSI]


class GMIMTest(unittest.TestCase):

    def test_get_im_success(self):
        im_without_spectrum = GMIntensityMeasures(GM_DATA, TIME_STEP).get_im(IM_WITHOUT_SPECTRUM)
        self.assertEqual(im_without_spectrum[GMIMEnum.PGA.name.upper()].shape[0], 1)
        im_with_spectrum = GMIntensityMeasures(GM_DATA, TIME_STEP).get_im(IM_WITHOUT_SPECTRUM + IM_SPECTRUM)
        self.assertEqual(im_with_spectrum[GMIMEnum.ASI.name.upper()].shape[0], 1)

        batch_gm_data = np.zeros((1000, GM_DATA.shape[0]))
        for i in range(1000):
            batch_gm_data[i, :] = GM_DATA
        im_without_spectrum_batch = GMIntensityMeasures(batch_gm_data, TIME_STEP).get_im(IM_WITHOUT_SPECTRUM)
        self.assertEqual(im_without_spectrum_batch[GMIMEnum.PGA.name.upper()].shape[0], 1000)
        im_with_spectrum_batch = GMIntensityMeasures(batch_gm_data, TIME_STEP).get_im(IM_WITHOUT_SPECTRUM + IM_SPECTRUM)
        self.assertEqual(im_with_spectrum_batch[GMIMEnum.ASI.name.upper()].shape[0], 1000)
