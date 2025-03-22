# -*- coding:utf-8 -*-
# @FileName  :sbs_integration_linear.py
# @Time      :2024/8/25 下午7:57
# @Author    :RichardoGu
import numpy as np


def segmented_parsing(
        mass: float,
        stiffness: float,
        load: np.ndarray,
        time_step: float,
        damping_ratio: float = 0.05,
        disp_0: float = 0,
        vel_0: float = 0) -> (np.ndarray, np.ndarray, np.ndarray):
    """
    This function is Segmented Parsing method, which is generally applicable
    for solving the dynamic response of single degree of freedom system.

    Args:
        mass:
        stiffness:
        load: The dynamic load array, changeable over time, is often the ground motion.
            This parameter should have 2 dims as (batch size, sequence length)
        time_step: The time step of load
        damping_ratio:
        disp_0: The init displacement of system, is often 0
        vel_0: The init velocity of system, is often 0

    Returns:
        The result is tuple, which consist of the ``(acceleration, velocity, displacement)`` response in order.
    -------

    """
    # Array fit
    if load.ndim == 1:
        load = np.expand_dims(load, 0)
    # Data preparation
    batch_size = load.shape[0]
    seq_length = load.shape[1]

    omega_n = np.sqrt(stiffness / mass)
    omega_d = omega_n * np.sqrt(1 - damping_ratio ** 2)
    temp_1 = np.e ** (-damping_ratio * omega_n * time_step)
    temp_2 = damping_ratio / np.sqrt(1 - damping_ratio ** 2)
    temp_3 = 2 * damping_ratio / (omega_n * time_step)
    temp_4 = (1 - 2 * damping_ratio ** 2) / (omega_d * time_step)
    temp_5 = omega_n / np.sqrt(1 - damping_ratio ** 2)
    sin = np.sin(omega_d * time_step)
    cos = np.cos(omega_d * time_step)

    p_a = temp_1 * (temp_2 * sin + cos)
    p_b = temp_1 * (sin / omega_d)
    p_c = 1 / stiffness * (temp_3 + temp_1 * (
            (temp_4 - temp_2) * sin - (1 + temp_3) * cos
    ))
    p_d = 1 / stiffness * (1 - temp_3 + temp_1 * (
            -temp_4 * sin + temp_3 * cos
    ))
    p_a_prime = -temp_1 * (temp_5 * sin)
    p_b_prime = temp_1 * (cos - temp_2 * sin)
    p_c_prime = 1 / stiffness * (-1 / time_step + temp_1 * (
            (temp_5 + temp_2 / time_step) * sin + 1 / time_step * cos
    ))
    p_d_prime = 1 / (stiffness * time_step) * (
            1 - temp_1 * (temp_2 * sin + cos)
    )

    # Init the start displacement and velocity.
    disp = np.zeros((batch_size, seq_length))
    vel = np.zeros((batch_size, seq_length))
    acc = np.zeros((batch_size, seq_length))

    if type(disp_0) is not np.ndarray:
        disp_0 = np.zeros(batch_size)
    if type(vel_0) is not np.ndarray:
        vel_0 = np.zeros(batch_size)
    disp[:, 0] = disp_0
    vel[:, 0] = vel_0

    # Start Iteration
    for i in range(seq_length - 1):
        disp[:, i + 1] = p_a * disp[:, i] + p_b * vel[:, i] + p_c * load[:, i] + p_d * load[:, i + 1]
        vel[:, i + 1] = (
                p_a_prime * disp[:, i] +
                p_b_prime * vel[:, i] + p_c_prime * load[:, i] + p_d_prime * load[:, i + 1])
        acc[:, i + 1] = -2 * damping_ratio * omega_n * vel[:, i + 1] - stiffness / mass * disp[:, i + 1]

    return acc, vel, disp


def newmark_beta_single(mass, stiffness, load, time_step,
                        damping_ratio=0.05, disp_0=0, vel_0=0,
                        acc_0=0, beta=0.25, gamma=0.5,
                        result_length=0):
    batch_size = load.shape[0]
    seq_length = load.shape[1]
    if result_length == 0:
        result_length = int(1.2 * load.shape[1])  # 计算持时
    load = np.append(load, np.zeros((batch_size, result_length - seq_length)), axis=1)

    disp = np.zeros((batch_size, result_length))
    vel = np.zeros((batch_size, result_length))
    acc = np.zeros((batch_size, result_length))
    if type(disp_0) is not np.ndarray:
        disp_0 = np.zeros(batch_size)
    if type(vel_0) is not np.ndarray:
        vel_0 = np.zeros(batch_size)
    if type(acc_0) is not np.ndarray:
        acc_0 = np.zeros(batch_size)
    disp[:, 0] = disp_0
    vel[:, 0] = vel_0
    acc[:, 0] = acc_0
    a_0 = 1 / (beta * time_step ** 2)
    a_1 = gamma / (beta * time_step)
    a_2 = 1 / (beta * time_step)
    a_3 = 1 / (2 * beta) - 1
    a_4 = gamma / beta - 1
    a_5 = time_step / 2 * (a_4 - 1)
    a_6 = time_step * (1 - gamma)
    a_7 = gamma * time_step
    omega_n = np.sqrt(stiffness / mass)
    damping = 2 * mass * omega_n * damping_ratio
    equ_k = stiffness + a_0 * mass + a_1 * damping  # 计算等效刚度
    # 迭代正式开始
    for i in range(result_length - 1):
        equ_p = load[:, i + 1] + mass * (
                a_0 * disp[:, i] + a_2 * vel[:, i] + a_3 * acc[:, i]) + damping * (
                        a_1 * disp[:, i] + a_4 * vel[:, i] + a_5 * acc[:, i])  # 计算等效荷载
        disp[:, i + 1] = equ_p / equ_k  # 计算位移
        acc[:, i + 1] = a_0 * (disp[:, i + 1] - disp[:, i]) - a_2 * vel[:, i] - a_3 * acc[:, i]  # 计算加速度
        vel[:, i + 1] = vel[:, i] + a_6 * acc[:, i] + a_7 * acc[:, i + 1]  # 计算速度
    return acc, vel, disp
