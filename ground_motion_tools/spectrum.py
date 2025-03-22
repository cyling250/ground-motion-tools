# -*- coding:utf-8 -*-
# @Time:    2025/3/17 17:32
# @Author:  RichardoGu
"""
Ground motion spectrum.
本程序包含几个功能：
    - 计算地震波的反应谱
    - 根据输入得到中国规范规定的建筑结构设计反应谱
    - 根据输入得到中国规范规定的桥梁结构设计反应谱
"""
import numpy as np
from .sbs_integration_linear import newmark_beta_single

SPECTRUM_PERIOD = [
    0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09,
    0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.8, 1.0,
    1.2, 1.4, 1.6, 1.8, 2.0, 2.5, 3.0,
    3.5, 4.0, 5.0, 6.0
]  # 反应谱取点，单位秒


def get_spectrum(
        gm_acc_data: np.ndarray,
        time_step: float,
        damping_ratio: float = 0.05,
        # Calculation way options
        calc_func: any = newmark_beta_single,
        calc_opt: int = 0,
        max_process: int = 4) -> (
        np.ndarray[np.float64],
        np.ndarray[np.float64],
        np.ndarray[np.float64],
        np.ndarray[np.float64],
        np.ndarray[np.float64]):
    """
    There are three types of response spectrum of ground motion: acceleration, velocity and displacement.
    Type must in tuple ("ACC", "VEL", "DISP"), and can be both upper and lower.

    Class :class:`GroundMotionData` use ``self.spectrum_acc`` , ``self.spectrum_acc`` , ``self.spectrum_acc``
    to save. And this three variants will be calculated when they are first used.

    TODO we use default damping ratio 0.05, try to use changeable damping ratio as input.

    Warnings:
    ------
    The programme starts multi-threaded calculations by default

    Args:
        gm_acc_data: Ground motion acc data.
        time_step: Time step.
        damping_ratio: Damping ratio.
        calc_opt: The type of calculation to use.

            - 0 Use single_threaded. Slow

            - 1 Use multi_threaded. Faster TODO This func not completed.

        calc_func: The type of calculation to use.
            The return of calc_func should be a tuple ``(acc, vel, disp)``.

        max_process: if calc_opt in [1,2], the multi thread will be used.
            This is the maximum number of threads.

    Returns:
        Calculated spectrum.
        spectrum_acc, spectrum_vel, spectrum_disp, spectrum_pse_acc, spectrum_pse_vel

    """
    if gm_acc_data.ndim == 1:
        gm_acc_data = np.expand_dims(gm_acc_data, axis=0)
    elif gm_acc_data.ndim == 2:
        pass
    else:
        raise ValueError("ndim of gm_acc_data must be 1 or 2.")

    batch_size = gm_acc_data.shape[0]
    seq_len = gm_acc_data.shape[1]
    spectrum_acc = np.zeros((batch_size, len(SPECTRUM_PERIOD)))
    spectrum_vel = np.zeros((batch_size, len(SPECTRUM_PERIOD)))
    spectrum_disp = np.zeros((batch_size, len(SPECTRUM_PERIOD)))
    spectrum_pse_acc = np.zeros((batch_size, len(SPECTRUM_PERIOD)))
    spectrum_pse_vel = np.zeros((batch_size, len(SPECTRUM_PERIOD)))

    if calc_opt == 0:
        for i in range(len(SPECTRUM_PERIOD)):
            acc, vel, disp = calc_func(
                mass=1,
                stiffness=(2 * np.pi / SPECTRUM_PERIOD[i]) ** 2,
                load=gm_acc_data,
                damping_ratio=damping_ratio,
                time_step=time_step,
                result_length=seq_len
            )
            spectrum_acc[:, i] = np.abs(acc).max(1)
            spectrum_vel[:, i] = np.abs(vel).max(1)
            spectrum_disp[:, i] = np.abs(disp).max(1)
            spectrum_pse_acc[:, i] = np.abs(disp).max(1) * (2 * np.pi / SPECTRUM_PERIOD[i]) ** 2
            spectrum_pse_vel[:, i] = np.abs(disp).max(1) * (2 * np.pi / SPECTRUM_PERIOD[i])

    elif calc_opt == 1:
        # Create Processes
        # TODO IF really need mutil-process, use cpp dll. Don't use python's mutil-process.
        pass

    else:
        raise KeyError("Parameter 'calc_opt' should be 0 or 1.")

    if gm_acc_data.shape[0] == 1:
        spectrum_acc = spectrum_acc.squeeze()
        spectrum_vel = spectrum_vel.squeeze()
        spectrum_disp = spectrum_disp.squeeze()
        spectrum_pse_acc = spectrum_pse_acc.squeeze()
        spectrum_pse_vel = spectrum_pse_vel.squeeze()

    return spectrum_acc, spectrum_vel, spectrum_disp, spectrum_pse_acc, spectrum_pse_vel


def design_spectrum_building(
        period: float,
        *,
        damping_ratio: float = 0.05,
        t_g: float = 0.35,
        alpha_max: float = 0.08
) -> float:
    """
    Design Response Spectrum Functions Defined According to Seismic Codes.

    Args:
        period: Structural period
        damping_ratio: Structural damping ratio
        t_g: Characteristic period
        alpha_max: Maximum seismic impact factor

    Returns: Seismic impact factor

    """
    if 0 > period:
        raise ValueError("Parameter 'structural_period' need in range (0, inf).")
    if 0 > damping_ratio or damping_ratio >= 1:
        raise ValueError("Parameter 'damping_ratio' need in range [0, 1).")

    gamma = 0.9 + (0.05 - damping_ratio) / (0.3 + 6 * damping_ratio)
    nita1 = 0.02 + (0.05 - damping_ratio) / (4 + 32 * damping_ratio)
    nita2 = 1 + (0.05 - damping_ratio) / (0.08 + 1.6 * damping_ratio)

    if nita1 < 0:
        nita1 = 0
    if nita2 < 0.55:
        nita2 = 0.55

    if period < 0.1:  # 上升段
        return (0.45 + 5.5 * period) * nita2 * alpha_max
    elif period < t_g:  # 水平段
        return nita2 * alpha_max
    elif period < 5 * t_g:  # 下降段
        return (t_g / period) ** gamma * nita2 * alpha_max
    else:  # 倾斜段
        return (0.2 ** gamma - nita1 / nita2 * (period - 5 * t_g)) * alpha_max * nita2


def design_spectrum_bridge(
        period: float,
        damping_ratio: float = 0.05,
        t_g: float = 0.35,
        alpha_max: float = 0.08
) -> float:
    """
    Design Response Spectrum Functions Defined According to Seismic Codes.

    Args:
        period: Structural period
        damping_ratio: Structural damping ratio
        t_g: Characteristic period
        alpha_max: Maximum seismic impact factor

    Returns: Seismic impact factor

    """
    pass


def match_sort(
        ground_motion_spectrum: np.ndarray[float, float],
        target_spectrum_func: callable
) -> np.ndarray[int]:
    """
    Rank the degree of match between the seismic spectrum and the target spectrum and output the ranking results.
    Args:
        ground_motion_spectrum: Seismic spectrum [period, spectrum]
        target_spectrum_func: Target Spectrum Calculation Function
    Returns:

    """
    pass


def match_discrete_periodic_point(
        ground_motion_data: np.ndarray[float, float] | np.ndarray,
        target_spectrum_func: callable,
        target_spectrum_params: dict,
        periodic_point: np.ndarray[float] | list[float],
        tremble: float | list | np.ndarray = 0.2,
        time_step: float = 0.02,
        damping_ratio: float = 0.05,
) -> list[int]:
    """
    Discrete Periodic Point Matching.
    Args:
        ground_motion_data: Ground Motion data. [batch_size, seq_length]
        time_step: Time step.
        periodic_point: Periodic point.
        target_spectrum_func: Target Spectrum Calculation Function
        target_spectrum_params: Parameters for the calculation of the target spectrum.
        damping_ratio: Damping ratio
        tremble: Permitted jitter deviation from the target spectrum.
    Returns: Array of matched seismic wave numbers.
    """
    if type(tremble) == float:
        tremble = np.array([tremble for i in range(len(periodic_point))])

    if len(tremble) != len(periodic_point):
        raise ValueError("Length of 'tremble' and 'periodic_point' must be equal.")

    reservation_ground_motion_idx = [i for i in range(ground_motion_data.shape[0])]

    for period_idx in range(len(periodic_point)):
        target_spectrum_value = target_spectrum_func(period=periodic_point[period_idx], **target_spectrum_params)
        temp_ground_motion_data = ground_motion_data[reservation_ground_motion_idx]
        acc, _, _ = newmark_beta_single(
            1, (2 * np.pi / periodic_point[period_idx]) ** 2,
            temp_ground_motion_data, time_step, damping_ratio
        )
        acc = acc.max(axis=1)

        for i in range(len(reservation_ground_motion_idx)):
            if (1 - tremble[period_idx]) * target_spectrum_value > acc[i] or (
                    1 + tremble[period_idx]) * target_spectrum_value < acc[i]:
                reservation_ground_motion_idx.remove(i)

            if len(reservation_ground_motion_idx) == 0:
                break

    return reservation_ground_motion_idx
