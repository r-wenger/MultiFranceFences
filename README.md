# Fence Detection using Multimodal Deep Learning Approaches

This repository contains the implementation code for detecting fences using multimodal aerial imagery (orthophotographs and Lidar DSM data). The repository includes different deep learning architectures (e.g., D-LinkNet, Handcrafted approaches) for training and inference. This code is part of the research presented in the paper: **"Where are the fences? A new deep learning approach to detect fences using multimodal aerial imagery"**.

## Table of Contents
- [Overview](#overview)
- [Directory Structure](#directory-structure)
- [Installation](#installation)
- [Data](#data)
- [Training](#training)
- [Testing](#testing)
- [Model Architectures](#model-architectures)
- [Citation](#citation)
- [License](#license)

## Overview

This repository provides the code to train and test models for semantic segmentation of fences using RGB orthophotographs and DSM data. The models are trained and evaluated on the **MultiFranceFences** dataset, which includes labeled fence data and corresponding aerial imagery.

## Directory Structure

Here is a brief description of the key files and folders in this repository:

```
- Departments/                           # Geographic split files for training/validation/test sets
- README.md                              # Project documentation
- data_loader_ram.py                     # Data loader for handling orthophotograph and DSM data
- dlinknet.py                            # Implementation of the D-LinkNet model
- test_concatenate_dlinknet.py           # Script for testing D-LinkNet model on concatenated inputs (RGB+DSM)
- test_concatenate.py                    # Script for testing UNet model
- test_concatenate_handcrafted.py        #  Script for training D-LinkNet model
- train_concatenate.py                   # Script for training UNet model
- train_concatenate_dlinknet.py          # Script for training D-LinkNet model
- train_concatenate_handcrafted.py       # Script for training handcrafted models
- graphical_validation.py                # Script to get the tool to visualize each ortho patch and reference for visual cleaning
```

## Installation

1. Clone the repository:

```bash
git clone https://github.com/r-wenger/MultiFranceFences-HIncepUNet
cd MultiFranceFences-HIncepUNet
```

2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

3. Ensure that you have `PyTorch` and `segmentation-models-pytorch` installed for model development.

## Data

This repository works with the **MultiFranceFences** dataset, which includes RGB orthophotographs and DSM data. You can download the dataset from [this link](https://www.easydata.earth/#/my-projects/75ba2410-a74f-4cd3-98f3-ebbec6cb52e5).

```
- MultiFranceFences/
  - ortho/         # Contains RGB orthophotographs
  - lidar/         # Contains DSM patches
  - fences_2m/     # Contains fence segmentation masks (2m buffers)
  - fences_3m/     # Contains fence segmentation masks (3m buffers)
```

## Training

**IMPORTANT:** The training and testing scripts are not fully parameterized via command line arguments. Users need to manually edit the necessary variables in the scripts before running them. Also, users need to adapt the variables according to their hardware configurations (e.g. batch size).

### Step-by-step training instructions:

1. Open the relevant training script (`train_concatenate.py`, `train_concatenate_dlinknet.py`, or `train_concatenate_handcrafted.py`) in a text editor.
   
2. Edit the different variables in the script according to your system setup and desired parameters.

3. Save your changes.

4. Run the script:

```bash
python train_concatenate_dlinknet.py
```

### Model-specific instructions:

- For D-LinkNet: Open and edit `train_concatenate_dlinknet.py`.
- For Handcrafted Model: Open and edit `train_concatenate_handcrafted.py`.
- For other models like UNet and H-IncepUNet: Open and edit `train_concatenate.py`.

## Testing

Testing works similarly to training. You need to manually edit the test scripts before running them.

### Step-by-step testing instructions:

1. Open the relevant test script (`test_concatenate_dlinknet.py`, `test_concatenate.py`, or `test_concatenate_handcrafted.py`).
   
2. Modify the different variables in the script:

3. Save the changes.

4. Run the script:

```bash
python test_concatenate_dlinknet.py
```

### Model-specific instructions:

- For D-LinkNet: Open and edit `test_concatenate_dlinknet.py`.
- For Handcrafted Model: Open and edit `test_concatenate_handcrafted.py`.
- For UNet Model: Open and edit `test_concatenate.py`.

## Model Architectures

- **D-LinkNet**: A model optimized for detecting linear objects like roads and fences using a dilated convolutional network. Implementation is available in `dlinknet.py`.
- **H-IncepUNet**: Custom deep learning model combining an Inception module and handcrafted features with UNet for precise fence segmentation.
- **UNet**: Baseline semantic segmentation model.

## Citation

If you use this repository in your research, please cite the corresponding paper:

```
@article{wenger2024fences,
  title={Where are the fences? A new deep learning approach to detect fences using multimodal aerial imagery},
  author={Wenger, Romain and Maire, Eric and Buton, Caryl and Moulherat, Sylvain and Staentzel, Cybill},
  journal={Submitted},
  year={2024},
  publisher={}
}
```

## License

This repository is licensed under the MIT License. See `LICENSE` for more details.
