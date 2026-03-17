# Ground Motion Tools

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![PyPI Version](https://img.shields.io/pypi/v/ground-motion-tools.svg)](https://pypi.org/project/ground-motion-tools/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A comprehensive Python library for processing and analyzing ground motion data, including seismic wave reading/writing, signal processing, response spectrum calculation, and intensity measure computation.

## Features

- **Multi-format Support**: Read and write seismic waves in various formats (KIK, PEER, single-column)
- **Signal Processing**: Fourier spectrum analysis, Butterworth filtering, down-sampling, normalization
- **Response Spectrum**: Calculate acceleration, velocity, and displacement response spectra
- **Intensity Measures**: Compute various ground motion intensity measures (PGA, PGV, PGD, ASI, HI, VSI)
- **Visualization**: Professional ground motion and spectrum visualization with customizable styling
- **Batch Processing**: Efficient batch calculations for large datasets
- **Performance Optimized**: Built with NumPy and SciPy for high-performance computations

## Installation

Install the package using pip:

```bash
pip install ground-motion-tools
```

For development installation:

```bash
git clone https://github.com/your-username/ground-motion-tools.git
cd ground-motion-tools
pip install -e .
```

## Requirements

- Python 3.10 or higher
- NumPy >= 2.2.6
- SciPy >= 1.15.3

## Quick Start

```python
import numpy as np
from ground_motion_tools import read_from_kik, GMIntensityMeasures, GMIMEnum

# Read ground motion data
gm_data, time_step = read_from_kik("path/to/your/seismic/data.kik")

# Calculate basic intensity measures
im_calculator = GMIntensityMeasures(gm_data, time_step)
basic_im = im_calculator.get_im([GMIMEnum.PGA, GMIMEnum.PGV, GMIMEnum.PGD])

print(f"PGA: {basic_im[GMIMEnum.PGA]:.3f}")
print(f"PGV: {basic_im[GMIMEnum.PGV]:.3f}")
print(f"PGD: {basic_im[GMIMEnum.PGD]:.3f}")
```

## Usage Examples

### Reading Ground Motion Data

```python
from ground_motion_tools import read_from_kik, read_from_single, read_from_peer

# Read from KIK format
file_path = "path/to/your/kik/file.kik"
gm_data, time_step = read_from_kik(file_path)

# Read from PEER format  
gm_data, time_step = read_from_peer(file_path)

# Read from single-column format
gm_data, time_step = read_from_single(file_path, column_index=1, skip_rows=None, time_step=0)
```

### Writing Ground Motion Data

```python
from ground_motion_tools import read_from_peer, save_to_single

# Read original data
ori_file = "path/to/original/ground/motion/file"
desc_file = "path/to/target/file.txt"

gm_data, time_step = read_from_peer(file_path=ori_file)

# Save to single-column format
save_to_single(desc_file, gm_data, time_step)

# Output format:
# Time Step: 0.02
# data1
# data2
# data3
# ...
```

### Signal Processing

```python
from ground_motion_tools import *
import matplotlib.pyplot as plt

# Read ground motion data
gm_data, time_step = read_from_kik("path/to/your/acceleration/data.kik")

# Calculate velocity and displacement from acceleration
acc, vel, disp = gm_data_fill(gm_data, time_step, GMDataEnum.ACC)

# Fourier spectrum analysis
frequencies, amplitude_spectrum, phase_spectrum = fourier(gm_data, time_step)
plt.plot(frequencies, amplitude_spectrum)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.title('Fourier Spectrum')
plt.show()

# Butterworth filtering
filtered_gm_data = butter_worth_filter(gm_data, time_step, order=4, low_cut=0.1, high_cut=25)

# Downsampling
down_sampled = down_sample(gm_data, time_step, target_time_step=0.02)

# Length normalization
normalized_wave = length_normalize(gm_data, target_length=1000)
```

### Response Spectrum Calculation

```python
from ground_motion_tools import *
import numpy as np

# Single ground motion analysis
gm_data, time_step = read_from_kik("path/to/your/acceleration/data.kik")
acc_spectrum, vel_spectrum, disp_spectrum, periods, damping = get_spectrum(gm_data, time_step)

# Batch processing for multiple ground motions
gm_data_many = np.zeros((100, gm_data.shape[0]))
for i in range(100):
    gm_data_many[i, :] = gm_data

acc_spectrum, vel_spectrum, disp_spectrum, periods, damping = get_spectrum(gm_data_many, time_step)
```

### Visualization

```python
from ground_motion_tools import *
import numpy as np

# Read ground motion data
gm_data, time_step = read_from_kik("path/to/your/acceleration/data.kik")

# Visualize ground motion data
show_gm(
    gm_data, 
    time_step, 
    title="Ground Motion Acceleration",
    y_label="Acceleration (g)",
    save_path="ground_motion_plot.png"
)

# Visualize multiple ground motion components
gm_data_2d = np.column_stack([gm_data, gm_data * 0.8, gm_data * 0.6])
component_names = ["Component 1", "Component 2", "Component 3"]
show_gm(
    gm_data_2d,
    time_step,
    title="Multiple Ground Motion Components",
    y_label="Acceleration (g)",
    component_names=component_names,
    save_path="multiple_components_plot.png"
)

# Calculate and visualize response spectrum
spectrum_data, _, _, _, _ = get_spectrum(gm_data, time_step)
show_gm_spectrum(
    spectrum_data,
    title="Response Spectrum",
    y_label="Spectral Acceleration (g)",
    save_path="response_spectrum_plot.png"
)

# Visualize multiple response spectra
spectrum_data_2d = np.column_stack([spectrum_data, spectrum_data * 0.8])
component_names = ["Damping 5%", "Damping 10%"]
show_gm_spectrum(
    spectrum_data_2d,
    title="Multiple Response Spectra",
    y_label="Spectral Acceleration (g)",
    component_names=component_names,
    save_path="multiple_spectra_plot.png"
)
```

### Intensity Measures

```python
from ground_motion_tools import *
import numpy as np

# Define intensity measure types
IM_WITHOUT_SPECTRUM = [GMIMEnum.PGA, GMIMEnum.PGV, GMIMEnum.PGD]
IM_SPECTRUM = [GMIMEnum.ASI, GMIMEnum.HI, GMIMEnum.VSI]

# Read ground motion data
gm_data, time_step = read_from_kik("path/to/your/acceleration/data.kik")

# Calculate intensity measures
im_calculator = GMIntensityMeasures(gm_data, time_step)

# Basic intensity measures (fast)
basic_im = im_calculator.get_im(IM_WITHOUT_SPECTRUM)

# Comprehensive intensity measures (includes spectrum-based measures)
comprehensive_im = im_calculator.get_im(IM_WITHOUT_SPECTRUM + IM_SPECTRUM)

# Batch processing
batch_gm_data = np.zeros((1000, gm_data.shape[0]))
for i in range(1000):
    batch_gm_data[i, :] = gm_data

im_calculator_batch = GMIntensityMeasures(batch_gm_data, time_step)
basic_im_batch = im_calculator_batch.get_im(IM_WITHOUT_SPECTRUM)
comprehensive_im_batch = im_calculator_batch.get_im(IM_WITHOUT_SPECTRUM + IM_SPECTRUM)
```

## API Reference

### Core Functions

- `read_from_kik(file_path)`: Read ground motion data from KIK format
- `read_from_peer(file_path)`: Read ground motion data from PEER format
- `read_from_single(file_path, column_index, skip_rows, time_step)`: Read from single-column format
- `save_to_single(file_path, gm_data, time_step)`: Save data to single-column format

### Signal Processing

- `gm_data_fill(gm_data, time_step, data_type)`: Convert between acceleration, velocity, displacement
- `fourier(gm_data, time_step)`: Compute Fourier spectrum
- `butter_worth_filter(gm_data, time_step, order, low_cut, high_cut)`: Apply Butterworth filter
- `down_sample(gm_data, time_step, target_time_step)`: Downsample ground motion data
- `length_normalize(gm_data, target_length)`: Normalize data length

### Spectrum Analysis

- `get_spectrum(gm_data, time_step)`: Calculate response spectra

### Visualization

- `show_gm(gm_data, time_step, save_path, y_label, show_plot, component_names, title)`: Visualize ground motion data
- `show_gm_spectrum(spectrum_data, save_path, y_label, show_plot, component_names, title)`: Visualize response spectrum data

### Intensity Measures

- `GMIntensityMeasures(gm_data, time_step)`: Main class for intensity measure calculations
- `get_im(im_list)`: Compute specified intensity measures

### Enumerations

- `GMDataEnum`: Data types (ACC, VEL, DISP)
- `GMIMEnum`: Intensity measure types (PGA, PGV, PGD, ASI, HI, VSI)
- `GMSpectrumEnum`: Spectrum types

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

For questions, issues, or feature requests, please open an issue on the [GitHub repository](https://github.com/your-username/ground-motion-tools).