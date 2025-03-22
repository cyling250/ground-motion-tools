# -*- coding:utf-8 -*-
# @Time:    2025/3/17 16:49
# @Author:  RichardoGu
"""
Some utils for processing ground motion.
"""
import numpy as np
from scipy import signal
from .enums import GMDataEnum


def gm_data_fill(gm_data: np.ndarray,
                 time_step: float = 0.02,
                 wave_type: GMDataEnum = GMDataEnum.ACC) -> (
        np.ndarray[np.float64], np.ndarray[np.float64], np.ndarray[np.float64]
):
    """
    Fit wave by input.

    Args:
        gm_data:
        wave_type:
        time_step:

    Returns:
        None
    """
    if gm_data.ndim == 1:
        # All the wave_array hereafter should have 2 dims as [batch_size, seq_len]
        gm_data = np.expand_dims(gm_data, axis=0)
    if wave_type.value == GMDataEnum.ACC.value:
        acc = gm_data
        vel = np.cumsum(acc, axis=1) * time_step
        disp = np.cumsum(vel, axis=1) * time_step
    elif wave_type.value == GMDataEnum.VEL.value:
        vel = gm_data
        disp = np.cumsum(vel, axis=1) * time_step
        acc = np.gradient(vel, time_step, axis=1)
    elif wave_type.value == GMDataEnum.DISP.value:
        disp = gm_data
        vel = np.gradient(disp, time_step, axis=1)
        acc = np.gradient(vel, time_step, axis=1)
    else:
        raise ValueError("Parameter wave_type must be included in [ACC, VEL, DISP].")
    return np.squeeze(acc), np.squeeze(vel), np.squeeze(disp)


def fourier(gm_data: np.ndarray, time_step: float) -> (
        np.ndarray, np.ndarray, np.ndarray
):
    """
    Calculate the fourier spectrum of wave.
    Args:
        gm_data: Ground motion data.
        time_step: Time step.

    Returns:
        x-axis of fourier spectrum(HZ)
        amp: A

    """
    x_fourier = np.abs(np.fft.fftfreq(d=time_step, n=len(gm_data))[len(gm_data) // 2:])[::-1]
    # Take the absolute value of the complex number, i.e., the mode of the complex number (bilateral spectrum)
    amp = np.abs(np.fft.fft(gm_data)) / len(gm_data)
    # extract a unilateral spectrum
    amp = (amp[len(gm_data) // 2:])[::-1]
    return x_fourier, amp, amp ** 2


def butter_worth_filter(
        gm_data: np.ndarray[np.float64],
        time_step: float,
        order: int = 4,
        start_freq: float = 0.1,
        end_freq: float = 15,
        pass_way: str = 'band'):
    """

    The start frequency and stop frequency are suggested as 0.1HZ and 25HZ.
    Because commonly the effective frequency in ground motions is range from 0.1hz to 25hz.

    Args:
        gm_data: Ground motion data.
        time_step: Time step.
        order: The order of butterworth filter. Default 4.
        start_freq: The start freq of filter. Default 0.1.
        end_freq: The end freq of filter. Default 0.1.
        pass_way: The pass way of filter. Default bandpass.

    Returns: Filtered waves.

    """
    b, a = signal.butter(order, [2 * start_freq * time_step, 2 * end_freq * time_step], pass_way)
    # ! Result by using ``lfilter`` is sample to Seismic Signal, not filtfilt
    return signal.lfilter(b, a, gm_data)


def down_sample(gm_data: np.ndarray, ori_time_step: float, tar_time_step: float) -> np.ndarray:
    """
    Down-sample the wave data.

    The method used for down-sampling is mean-down-sampling.
    Use mean down-samping method can let the calculated displacement and velocity be the same as before.

    Args:
        gm_data: The input wave data.
        ori_time_step: Origin time step.
        tar_time_step: Target time step.
    Returns:
        Downsized wave_data. Using scipy.signal.resample
    """

    # The two lines that are commented out are the previous methods.
    # tar_data_size = int(ori_time_step / tar_time_step * wave_data.shape[wave_data.ndim - 1])
    # return signal.resample(wave_data, tar_data_size, axis=axis).mean()

    num_samples = int(gm_data.shape[-1] * ori_time_step / tar_time_step)
    return signal.resample(gm_data, num_samples, axis=gm_data.ndim - 1)


def length_normalize(gm_data: np.ndarray[np.float64], normal_length: int):
    """
    Seismic wave length normalisation method.

    Normalisation algorithm:
        1. If the original seismic wave length l1 is less than the normalised seismic wave length ln,
            then zero is added directly at the end.
        2. If the original seismic wave length l1 is greater than the seismic wave length ln to be normalised,
            the following operation is performed.
            2.1 Extraction of ground shaking PGA occurrences i1
            2.2 Calculate the ratio a(0<a<1) of the original length to the normalised length,
                then the original ground shaking should be extracted int(i1*a) units, before the peak appears.
                When the peak appears, it is dealt with directly by the truncation and zero filling method.
    Args:
        gm_data: Ground motion data.
        normal_length: 要归一化的长度

    Returns:

    """
    # 1 The normalised length is greater than the original length
    if gm_data.shape[0] <= normal_length:
        return np.pad(
            gm_data,
            (0, normal_length - gm_data.shape[0]),
            'constant',
            constant_values=(0, 0)
        )
    # 2 The normalised length is less than the original length
    else:
        cut_rate = normal_length / gm_data.shape[0]
        pga_loca = np.argmax(np.abs(gm_data))
        forward_length = int(pga_loca * cut_rate)
        res = gm_data[pga_loca - forward_length:normal_length - forward_length + pga_loca]
        return res


def pga_adjust(ground_motion_data: np.ndarray, target_pga):
    """
    按照目标PGA对目标地震波进行调幅。
    Args:
        ground_motion_data: 地震波
        target_pga: 目标地震波PGA

    Returns:

    """
    if ground_motion_data.ndim == 1:
        return ground_motion_data * target_pga / np.abs(ground_motion_data).max()
    elif ground_motion_data.ndim == 2:
        return ground_motion_data / np.abs(ground_motion_data).max(axis=1).reshape(-1, 1) * target_pga
    else:
        raise ValueError("ndim of parameter 'ground_motion_data' must be 1 or 2.")
