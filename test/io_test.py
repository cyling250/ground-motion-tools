# -*- coding:utf-8 -*-
# @Time:    2025/3/17 20:37
# @Author:  RichardoGu
import unittest
import numpy as np

from ground_motion_tools import read_from_kik, read_from_peer, read_from_single, save_to_single


class GroundMotionTest(unittest.TestCase):
    ACC_FILE_PEER = "RSN88_SFERN_FSD172.AT2"
    VEL_FILE_PEER = "RSN88_SFERN_FSD172.VT2"
    DISP_FILE_PEER = "RSN88_SFERN_FSD172.DT2"
    ACC_FILE_KIK = "ABSH010011140057.EW2"
    ACC_FILE_SINGLE = "RSN88_SFERN_FSD172.txt"
    DATA_DIR = "./data/"

    def test_read_from_kik_success(self):
        file_path = GroundMotionTest.DATA_DIR + GroundMotionTest.ACC_FILE_KIK
        gm_data, time_step = read_from_kik(file_path)
        self.assertEqual(type(gm_data), np.ndarray)
        self.assertEqual(time_step, 0.005)

    def test_read_from_peer_success(self):
        file_path = GroundMotionTest.DATA_DIR + GroundMotionTest.ACC_FILE_PEER
        gm_data, time_step = read_from_peer(file_path)
        self.assertEqual(gm_data.shape, (8000,))
        self.assertEqual(time_step, 0.005)

    def test_save_to_single_success(self):
        ori_file = GroundMotionTest.DATA_DIR + GroundMotionTest.ACC_FILE_PEER
        desc_file = GroundMotionTest.DATA_DIR + GroundMotionTest.ACC_FILE_SINGLE
        gm_data, time_step = read_from_peer(file_path=ori_file)
        self.assertEqual(gm_data.shape, (8000,))
        self.assertEqual(time_step, 0.005)
        save_to_single(desc_file, gm_data, time_step)

    def test_read_from_single_success(self):
        file_path = GroundMotionTest.DATA_DIR + GroundMotionTest.ACC_FILE_SINGLE
        gm_data, time_step = read_from_single(file_path, 1, None, 0)
        self.assertEqual(gm_data.shape, (8000,))
        self.assertEqual(time_step, 0.005)
