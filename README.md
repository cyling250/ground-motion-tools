# Ground-Motion-Tools

The main functions of this programme include:

1. reading and writing seismic waves in various formats
2. seismic wave processing, such as the calculation of Fourier spectra, filtering, down-sampling, normalisation
3. ground vibration response spectrum calculation
4. ground shaking IM index calculation

## Installation

```commandline
pip install ground-motion-tools
```

## Examples

### Read

```python
from ground_motion_tools import read_from_kik, read_from_single, read_from_peer

file_path = "your kik file path"

# read from kik
gm_data1, time_step1 = read_from_kik(file_path)

# read from peer
gm_data2, time_step2 = read_from_peer(file_path)

# read from single
gm_data3, time_step3 = read_from_single(file_path, 1, None, 0)
```

### Write

```python
from ground_motion_tools import read_from_peer, save_to_single

ori_file = "your origin ground motion file"
desc_file = "your target ground motion file"
gm_data, time_step = read_from_peer(file_path=ori_file)
save_to_single(desc_file, gm_data, time_step)

# The single file is like following.
"""
Time Step: 0.02
data1
data2
data3
...
"""
```

### Process

```python
from ground_motion_tools import *
from matplotlib import pyplot as plt

gm_data, time_step = read_from_kik("your acc file path")

# calculate vel, disp from acc
acc, vel, disp = gm_data_fill(gm_data, time_step, GMDataEnum.ACC)

# fourier_spectrum
gm_data, time_step = read_from_kik("your acc file path")
x, y1, y2 = fourier(gm_data, time_step)
plt.plot(x, y1)
plt.show()

# butter_worth_filter
filtered_gm_data = butter_worth_filter(gm_data, time_step, 4, 0.1, 25)

# downsample
down_sampled = down_sample(gm_data, time_step, 0.02)

# length_normalize
normalized_wave = length_normalize(gm_data, 1000)
```

## Spectrum

```python
from ground_motion_tools import *
import numpy as np

# one ground motion
gm_data, time_step = read_from_kik("your acc file path")
acc_spectrum, vel_spectrum, disp_spectrum, _, _ = get_spectrum(gm_data, time_step)

# The programme supports batch calculations
gm_data_many = np.zeros((100, gm_data.shape[0]))
for i in range(100):
    gm_data_many[i, :] = gm_data
acc_spectrum, vel_spectrum, disp_spectrum, _, _ = get_spectrum(gm_data_many, time_step)
```

## Intensity measures

```python
from ground_motion_tools import *
import numpy as np

IM_WITHOUT_SPECTRUM = [GMIMEnum.PGA, GMIMEnum.PGV, GMIMEnum.PGD]
IM_SPECTRUM = [GMIMEnum.ASI, GMIMEnum.HI, GMIMEnum.VSI]

# one ground motion
gm_data, time_step = read_from_kik("your acc file path")

# Some IM metrics related to response spectra can consume a lot of time to calculate
im_without_spectrum = GMIntensityMeasures(gm_data, time_step).get_im(IM_WITHOUT_SPECTRUM)
im_with_spectrum = GMIntensityMeasures(gm_data, time_step).get_im(IM_WITHOUT_SPECTRUM + IM_SPECTRUM)

# The programme supports batch calculations.
batch_gm_data = np.zeros((1000, gm_data.shape[0]))
for i in range(1000):
    batch_gm_data[i, :] = gm_data
im_without_spectrum_batch = GMIntensityMeasures(batch_gm_data, time_step).get_im(IM_WITHOUT_SPECTRUM)
im_with_spectrum_batch = GMIntensityMeasures(batch_gm_data, time_step).get_im(IM_WITHOUT_SPECTRUM + IM_SPECTRUM)
```