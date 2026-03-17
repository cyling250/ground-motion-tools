# -*- coding: utf-8 -*-
"""
@File    :   visualization_test.py
@Time    :   2026/03/17 12:41:09
@Author  :   Xiaopeng Gu
@Version :   1.0
@Desc    :   Test the visualization module
"""

import unittest
import numpy as np
from ground_motion_tools import read_from_kik
from ground_motion_tools.process import down_sample
from ground_motion_tools import spectrum
from ground_motion_tools import visualization
import os


class VisualizationTest(unittest.TestCase):
    ACC_FILE_KIK = "ABSH010011140057.EW2"
    DATA_DIR = "./resource/"
    file_path = DATA_DIR + ACC_FILE_KIK
    gm_data, time_step = read_from_kik(file_path)
    gm_data = down_sample(gm_data, time_step, 0.02)
    spectrum_data, _, _, _, _ = spectrum.get_spectrum(gm_data, time_step)
    show_plot = True
    save_path = "./resource/visualization_test.png"

    def test_show_gm(self):
        """Test basic functionality of show_gm function"""
        # Test with basic parameters
        passed = True
        try:
            visualization.show_gm(
                self.gm_data,
                self.time_step,
                show_plot=self.show_plot,
                save_path=self.save_path,
            )
            # If no exception is raised, the test passes
            if os.path.exists(self.save_path):
                os.remove(self.save_path)
            else:
                passed = False
        except Exception as e:
            print(f"Error in test_show_gm: {e}")
            passed = False
        self.assertTrue(passed)

        gm_datas = np.array([self.gm_data, self.gm_data + 0.001])
        try:
            visualization.show_gm(
                gm_datas,
                self.time_step,
                show_plot=self.show_plot,
                save_path=self.save_path,
                component_names=["Component 1", "Component 2"],
                title="Test Ground Motion",
            )
            if os.path.exists(self.save_path):
                os.remove(self.save_path)
            else:
                passed = False
        except Exception as e:
            print(f"Error in test_show_gm: {e}")
            passed = False
        self.assertTrue(passed)

    def test_show_gm_spectrum(self):
        """Test basic functionality of show_gm_spectrum function"""
        # Test with basic parameters
        passed = True
        try:
            visualization.show_gm_spectrum(
                self.spectrum_data,
                show_plot=self.show_plot,
                save_path=self.save_path,
            )
            # If no exception is raised, the test passes
            if os.path.exists(self.save_path):
                os.remove(self.save_path)
            else:
                passed = False
        except Exception as e:
            print(f"Error in test_show_gm_spectrum: {e}")
            passed = False
        self.assertTrue(passed)

        # Test with component_names
        spectrum_datas = np.array([self.spectrum_data, self.spectrum_data + 0.001])
        try:
            visualization.show_gm_spectrum(
                spectrum_datas,
                show_plot=self.show_plot,
                save_path=self.save_path,
                component_names=["Component 1", "Component 2"],
                title="Test Spectrum",
            )
            if os.path.exists(self.save_path):
                os.remove(self.save_path)
            else:
                passed = False
        except Exception as e:
            print(f"Error in test_show_gm_spectrum: {e}")
            passed = False
        self.assertTrue(passed)
