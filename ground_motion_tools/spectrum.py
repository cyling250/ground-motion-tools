# -*- coding:utf-8 -*-
# @Time:    2025/3/17 17:32
# @Author:  RichardoGu
"""
Ground motion spectrum Calc
"""
import numpy as np
from numpy import ndarray, float64
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
        ndarray[float64],
        ndarray[float64],
        ndarray[float64],
        ndarray[float64],
        ndarray[float64]):
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
        Calculated spectrum. np.ndarray[float] (batch size)

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
        spectrum_pse_acc = spectrum_pse_acc.squeeze()

    return spectrum_acc, spectrum_vel, spectrum_disp, spectrum_pse_acc, spectrum_pse_acc
