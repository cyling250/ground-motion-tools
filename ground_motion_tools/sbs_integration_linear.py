# -*- coding:utf-8 -*-
# @FileName  :sbs_integration_linear.py
# @Time      :2024/8/25 下午7:57
# @Author    :RichardoGu
from typing import Tuple
import numpy as np


def newmark_beta_sdof_gms(
    mass: float,
    stiffness: float,
    load: np.ndarray,
    time_step: float,
    result_length: int,
    *,
    damping_ratio: float = 0.05,
    disp_0: float = 0.0,
    vel_0: float = 0.0,
    acc_0: float = 0.0,
    beta: float = 0.25,
    gamma: float = 0.5,
    **kwargs,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Newmark-beta integration for singleDOF system.
    Args:
        mass: Mass of the system.
        stiffness: Stiffness of the system.
        load: Load of the system.
        time_step: Time step.
        damping_ratio: Damping ratio.
        disp_0: Initial displacement.
        vel_0: Initial velocity.
        acc_0: Initial acceleration.
        beta: Beta parameter.
        gamma: Gamma parameter.
        result_length: Result length.
    Returns:
        acc: Acceleration of the system.
        vel: Velocity of the system.
        disp: Displacement of the system.
    """
    batch_size = load.shape[0]
    seq_length = load.shape[1]

    if result_length > seq_length:
        load = np.append(
            load, np.zeros((batch_size, result_length - seq_length)), axis=1
        )

    disp = np.zeros((batch_size, result_length))
    vel = np.zeros((batch_size, result_length))
    acc = np.zeros((batch_size, result_length))

    disp[:, 0] = disp_0
    vel[:, 0] = vel_0
    acc[:, 0] = acc_0

    a_0 = 1 / (beta * time_step**2)
    a_1 = gamma / (beta * time_step)
    a_2 = 1 / (beta * time_step)
    a_3 = 1 / (2 * beta) - 1
    a_4 = gamma / beta - 1
    a_5 = time_step / 2 * (a_4 - 1)
    a_6 = time_step * (1 - gamma)
    a_7 = gamma * time_step

    omega_n = np.sqrt(stiffness / mass)
    damping = 2 * mass * omega_n * damping_ratio
    equ_k = stiffness + a_0 * mass + a_1 * damping

    for i in range(result_length - 1):
        equ_p = (
            load[:, i + 1]
            + mass * (a_0 * disp[:, i] + a_2 * vel[:, i] + a_3 * acc[:, i])
            + damping * (a_1 * disp[:, i] + a_4 * vel[:, i] + a_5 * acc[:, i])
        )

        disp[:, i + 1] = equ_p / equ_k

        acc[:, i + 1] = (
            a_0 * (disp[:, i + 1] - disp[:, i]) - a_2 * vel[:, i] - a_3 * acc[:, i]
        )

        vel[:, i + 1] = vel[:, i] + a_6 * acc[:, i] + a_7 * acc[:, i + 1]

    return acc, vel, disp
