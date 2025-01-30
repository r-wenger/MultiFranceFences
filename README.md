# Fence Detection with Deep Learning

This repository contains the implementation of various deep learning models configuration to detect fences using multimodal imagery (RGB and DSM). The project, related to the paper **Where are the fences? A new deep learning approach to detect fences using multimodal aerial imagery**, explores the impact of different fusion strategies, loss functions, and sampling techniques on model performance. It also includes tools for dataset visualization and evaluation.

## 🚀 Features

- Multiple architectures: **UNet**, **UNetLateFusion**, **UNetConcatenate**, and **D-LinkNet**.
- Multimodal input support (RGB + DSM).
- Flexible loss functions: Dice Loss, Binary Cross-Entropy (BCE), and Combined Loss (Dice + BCE).
- Sampling strategies: random and geographic.
- Tools for dataset visualization and validation.

---

## 📂 Project Structure

```
├── Departments/             # GPKG dataset for geographic stratification (train, val, test)
├── LICENSE                  # Code license
├── README.md                
├── data_loader.py           # Data loading pipeline for RGB, DSM, and masks, adapted to each network
├── graphical_validation.py  # Dataset visualization tool for inspecting masks and features (launch it separately)
├── loss.py                  # Implementation of Dice Loss, BCE, and Combined Loss
├── main.py                  # Script for training models (modify each path/variable in the config function)
├── main_test.py             # Script for testing and evaluating models (modify each path/variable in the config function)
├── model.py                 # Implementation of UNet, UNetLateFusion, UNetConcatenate, and D-LinkNet
├── requirements.txt         # Python dependencies
├── test.py                  # Evaluation metrics and patch analysis
└── train.py                 # Training logic and scheduler implementation
```

---

## 📦 Dataset Preparation

### 1. Input Data Structure
The project expects the following directory structure for the dataset:
```
/dataset/
├── ortho/        # RGB orthophotos
├── lidar/        # DSM (Digital Surface Model)
└── fences_3m/    # Binary masks (3-meter buffer for fences)
└── fences_2m/    # Binary masks (2-meter buffer for fences)
```

You can download it here: [MultiFranceFences](https://zenodo.org/records/13902550)

### 2. Geographic Stratification
The `Departments/` folder contains a **GeoPackage (GPKG)** file, which ensures balanced geographic stratification for training, validation, and testing splits. This is critical for evaluating generalization across regions.

---

## 🔍 Dataset Visualization

The `graphical_validation.py` script is designed to help you visually inspect the dataset quality, including masks, RGB, and DSM alignment. Run the script as follows:

```bash
python graphical_validation.py
```

It saves the name of the patch and the tag (valid or not) in a CSV file.

---

## ⚙️ Training

### 1. Configurable Parameters
The training script (`main.py`) is built around a flexible `Config` class. Key parameters include:
- **Models**: `UNet`, `UNetLateFusion`, `UNetConcatenate`, `D-LinkNet`.
- **Input modalities**: RGB, DSM, or both.
- **Sampling strategies**: random or geographic.
- **Loss functions**: Dice Loss, BCE, or Combined Loss.
- **Learning rate and scheduler**: Supports dynamic learning rate reduction.

The code is commented. You can adapt the class depending on your needs. It is developed in order to use multi-GPU with `nn.DataParallel`.

### 2. Running the Training
To start training, modify the `Config` class in `main.py` and execute:
```bash
python main.py
```

Model checkpoints are saved in the `models/` directory with automatically generated filenames based on the configuration.

---

## 🧪 Testing and Evaluation

### 1. Testing
To evaluate a trained model, use the `main_test.py` script and modify the path to the .pth file:
```bash
python main_test.py
```

This will:
- Compute predictions and save them as GeoTIFF files.
- Generate classification metrics (precision, recall, IoU, Dice coefficient).
- Identify the best and worst-performing patches.

### 2. Example Output
The script outputs:
- Evaluation metrics in a text file.
- Visualizations of predictions and ground truths for selected patches.

---

## 🧰 Requirements

Install dependencies using:
```bash
pip install -r requirements.txt
```

Key libraries:
- PyTorch
- segmentation-models-pytorch
- GDAL
- NumPy
- Matplotlib

---

## 📜 Citation
If you use this work, please cite:

```
@article{wenger2024fences,
  title={Where are the fences? A new deep learning approach to detect fences using multimodal aerial imagery},
  author={Wenger, Romain and Maire, Eric and Buton, Caryl and Moulherat, Sylvain and Staentzel, Cybill},
  journal={Submitted},
  year={2024},
  publisher={}
}
```

---

## 📜 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
