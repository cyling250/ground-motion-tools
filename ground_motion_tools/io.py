# -*- coding:utf-8 -*-
# @Time:    2025/3/17 16:10
# @Author:  RichardoGu
"""
This file is mainly used to read or write ground motion.
"""
import re
import numpy as np

TIME_RE = r"\d{4}/\d{2}/\d{2} \d{2}:\d{2}:\d{2}"
NUMBER_RE = r"[+-]?(\d+([.]\d*)?([eE][+-]?\d+)?|[.]\d+([eE][+-]?\d+)?)"
TIME_FORMAT = "%Y/%m/%d %H:%M:%S"


def read_from_kik(file_path: str) -> (
        np.ndarray[np.float64], float
):
    """
    Read ground motion data from KIK format.
    Args:
        file_path: File path

    Returns:
        gm_data, time_step
    """
    gm_data = None
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if re.match('^Memo.', line):
                # if line is begin of "Memo.", it means all the params are read.
                # The following data is time-history data.
                idx = lines.index(line) + 1
                gm_data = []
                for i in range(idx, len(lines)):
                    temp = lines[i].split()
                    for j in temp:
                        gm_data.append(eval(j) * 0.01)  # Times 0.01 to convert gal(cm/s^2) to SI(m/s^2)
                break
            elif re.match("^Scale Factor.", line):
                match_result = re.search(r'(\d+)\D*(\d+)/?$', line)
                scale_factor = eval(match_result.group(1)) / eval(match_result.group(2))
            elif re.match("^Sampling Freq.", line):
                time_step = 1 / eval(re.search(NUMBER_RE, line).group())

    if gm_data is not None and scale_factor:
        gm_data = np.array(gm_data, dtype=np.float64)
        gm_data = (gm_data - gm_data.mean()) * scale_factor
    else:
        raise ValueError("Read ground motion data error. Parameter 'scale_factor' or gm_data may be None.")
    return gm_data, time_step


def read_from_peer(file_path: str) -> (
        np.ndarray[np.float64], float
):
    """
    Read ground motion data from PEER format.
    Args:
        file_path:

    Returns:
        gm_data, time_step
    """
    with open(file_path, 'r') as fp:
        lines = fp.readlines()
        time_step = eval(re.findall(r"\.[0-9]*", lines[3])[0])
        gm_data = []
        for i in range(4, len(lines)):
            temp = lines[i].split()
            for j in temp:
                gm_data.append(eval(j) * 9.8)  # Times 9.8 to convert G(9.8m/s^2) to m/s^2
    return np.array(gm_data, dtype=np.float64), time_step


def read_from_single(file_path: str, start_line: int = 1,
                     end_line: int = None, time_step: [int, float] = 0) -> (
        np.ndarray[np.float64], float
):
    """
    Reading seismic wave data from a single column file
    The default single column file format is: the first row is the sampling interval,
    the second row to the end of the file is the seismic wave data.
    Args:
        file_path: File path
        start_line: Number of rows of ground motion. Default is the second row.
        end_line: Number of end rows of ground motion. Default is None
        time_step:
            Ground motion Sampling Interval Directly indicates the sampling interval.
            if it is a floating point number,
            otherwise it indicates the number of rows where the sampling interval is located.

    Returns:
        gm_data, time_step
    """
    with open(file_path, 'r') as fp:
        lines = fp.readlines()
        wave_data = [float(line) for line in lines[start_line:end_line]]
        if type(time_step) is float:
            pass
        elif type(time_step) is int:
            time_step = float(lines[time_step].split(" ")[-1])
        else:
            raise ValueError(f"Type of parameter 'time_step' need to be float or int. But got {type(time_step)}.")
        return np.array(wave_data, dtype=np.float64), time_step


def save_to_single(file_path: str, gm_data: np.ndarray[np.float64], time_step: float = None) -> None:
    """
    Save ground motion to single file.
    Args:
        gm_data: Data of ground motion.
        time_step: Time step of ground motion.
        file_path: Path to save.

    Returns:
        None
    """
    with open(file_path, 'w') as fp:
        if time_step is not None:
            fp.write(f"Time Step: {time_step}\n")
        for data in gm_data:
            fp.write(f"{data}\n")
