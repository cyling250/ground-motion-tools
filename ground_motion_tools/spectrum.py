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
        acc_max: float = 0.08 * 9.8
) -> float:
    """
    Design Response Spectrum Functions Defined According to Seismic Codes.

    Args:
        period: Structural period
        damping_ratio: Structural damping ratio
        t_g: Characteristic period
        acc_max: Maximum spectrum value of accelerate

    Returns: Spectrum value of acceleration.

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
        return (0.45 + 5.5 * period) * nita2 * acc_max
    elif period < t_g:  # 水平段
        return nita2 * acc_max
    elif period < 5 * t_g:  # 下降段
        return (t_g / period) ** gamma * nita2 * acc_max
    else:  # 倾斜段
        return (0.2 ** gamma - nita1 / nita2 * (period - 5 * t_g)) * acc_max * nita2


def design_spectrum_bridge(
        period: float,
        *,
        damping_ratio: float = 0.05,
        t_g: float = 0.35,
        c_i: float = 0.43,
        c_s: float = 1,
        acc_max: float = 0.35
) -> float:
    """
    依据《公路桥梁抗震设计规范》JTG/T  2231-01——2020(下称 桥梁抗规)所规定的桥梁地震设计反应谱

    According to the seismic design code for highway bridges JTG/T 2231-01--2020,
    (hereinafter referred to as the code)
    the seismic design response spectrum of the bridge is as follows.

    Args:
        acc_max:
        水平向基本地震动峰值加速度 按照《桥梁抗规》表3.2.2取值
        Horizontal peak acceleration of basic ground vibration According to Table 3.2.2 of the Code.
        c_s:
        场地系数 按照《桥梁抗规》表5.2.2取值
        Site factor Taken from Table 5.2.2 of the Code.
        c_i:
        抗震重要性系数 按照《桥梁抗规》表3.1.3-2取值
        Seismic Importance Coefficient According to Table 3.1.3-2 of the Code.
        period: 结构周期 Structural period
        damping_ratio: 结构阻尼比 Structural damping ratio
        t_g: 场地特征周期 Site Characteristic period

    Returns: Spectrum value of acceleration.

    """
    if t_g < 0 or t_g > 10:
        raise ValueError("Parameter 't_g' can not letter than -0 or bigger than 10.")

    # 计算阻尼调整系数 Calculate the damping adjustment factor
    c_d = 1 + (0.05 - damping_ratio) / (0.08 + 1.6 * damping_ratio)
    c_d = c_d if c_d >= 0.55 else 0.55
    # S_max
    s_max = 2.5 * c_i * c_s * c_d * acc_max

    # 上升段 upward trend
    if period < 0.1:
        return s_max * (0.6 * period / 0.1 + 0.4)
    elif 0.1 <= period <= t_g:
        return s_max
    elif t_g < period <= 10:
        return s_max * (t_g / period)
    else:
        raise RuntimeError("Parameter 'period' must be in the interval (0, inf)")


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
        periodic_point: np.ndarray[float] | list[float],
        target_spectrum: np.ndarray[float] | list[float],
        tolerance: float | list | np.ndarray = 0.2,
        time_step: float = 0.02,
        damping_ratio: float = 0.05):
    """
    反应谱匹配 基于离散周期点匹配
    Reaction Spectrum Matching Based on Discrete Periodic Point Matching.
    Args:
        ground_motion_data:
            地震动数据
            Ground Motion data. [batch_size, seq_length]
        time_step:
            地震动时间步长
            Time step.
        periodic_point:
            离散周期点的周期
            Periodic point need to be calculated.
        target_spectrum:
            目标反应谱的值 长度和periodic_point对应
            The value of the target response spectrum The length corresponds to the periodic_point.
        damping_ratio:
            阻尼比
            Damping ratio
        tolerance:
            离散周期点的容许误差
            Permitted jitter deviation from the target spectrum.
    Examples:
        >>> data = np.array([[1,2,3],[4,5,6]]) # 地震波数据 ground motion data
        >>> period_list = np.array([0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1, 1.2, 1.5, 1.6, 1.8, 2])
        >>> target_spectrum = []
        >>> for period in period_list:
        >>>     target_spectrum.append(design_spectrum_building(period, damping_ratio=0.05, t_g=0.35, acc_max=0.08 * 9.8))
        >>> idx = match_discrete_periodic_point(
        >>>     data,
        >>>     period_list,
        >>>     target_spectrum,
        >>>     [0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3]
        >>> )
    Returns:
        Array of matched seismic wave numbers.
    """
    if type(tolerance) == float:
        tolerance = np.array([tolerance for i in range(len(periodic_point))])

    if len(tolerance) != len(periodic_point):
        raise ValueError("Length of 'tolerance' and 'periodic_point' must be equal.")

    reserve_ground_motion_idx = np.arange(ground_motion_data.shape[0], dtype=int)
    for period_idx in range(len(periodic_point)):
        temp_ground_motion_data = ground_motion_data.take(reserve_ground_motion_idx, axis=0)

        _, _, disp = newmark_beta_single(
            1, (2 * np.pi / periodic_point[period_idx]) ** 2,
            temp_ground_motion_data, time_step, damping_ratio
        )
        acc = np.abs(disp).max(axis=1) * (2 * np.pi / periodic_point[period_idx]) ** 2

        target_spectrum_lower = (1 - tolerance[period_idx]) * target_spectrum[period_idx]
        target_spectrum_upper = (1 + tolerance[period_idx]) * target_spectrum[period_idx]

        temp_delete_idx = []
        for i in range(temp_ground_motion_data.shape[0]):
            if target_spectrum_lower > acc[i] or target_spectrum_upper < acc[i]:
                temp_delete_idx.append(i)

        reserve_ground_motion_idx = np.delete(reserve_ground_motion_idx, temp_delete_idx)

        if len(reserve_ground_motion_idx) == 0:
            break
    return reserve_ground_motion_idx
