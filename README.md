# Edge-Guided Multi-Branch Network for Underwater Image Enhancement

## Introduction

This project implements an **Edge-Guided Multi-Branch Network** for underwater image enhancement. It leverages a commonality-specificity model to fuse multi-branch features, improving the visual quality of underwater images. The method effectively restores image details and colors while reducing underwater light attenuation and color distortion.

## Architecture

![images](https://github.com/wuxianjingong1/EGMCS-Net/blob/main/model.png)
*Illustration of the Edge-Guided Multi-Branch network, including the RGB branch, edge branch, and fusion modules.*

> Replace `docs/architecture.png` with your actual network diagram.

## Environment

* Python 3.11
* PyTorch 2.1.0

Example of creating the environment:

```bash
conda create -n uienv python=3.11
conda activate uienv
pip install torch==2.1.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

## Datasets

Place the datasets in the project root directory.

### 1. UIEB

The UIEB (Underwater Image Enhancement Benchmark) dataset can be found at:
[https://li-chongyi.github.io/proj_benchmark.html](https://li-chongyi.github.io/proj_benchmark.html)
It includes 890 raw underwater images with corresponding high-quality reference images, and an additional 60 challenging underwater images (C60) without references.

### 2. LSUI

The LSUI (Large-Scale Underwater Image) dataset is provided in the repository at:
[https://github.com/LintaoPeng/U-shape_Transformer_for_Underwater_Image_Enhancement](https://github.com/LintaoPeng/U-shape_Transformer_for_Underwater_Image_Enhancement)
It contains 4,279 real-world underwater image pairs along with semantic segmentation maps and medium transmission maps.

### 3. C60

The “60 challenging” subset (C60) is also available from the UIEB project page:
[https://li-chongyi.github.io/proj_benchmark.html](https://li-chongyi.github.io/proj_benchmark.html)

**Directory structure example**:

```
project_root/
├── UIEB/
│   ├── raw-890/
│   ├── reference-890/
│   ├── challenging-60/    ← C60
└── LSUI/
    ├── input/
    ├── GT/
    ├── segmentation_maps/
    ├── transmission_maps/
```

## Usage

1. **Generate Edge Maps**

```bash
python utils.py
```

Generates edge maps for the dataset (e.g., UIEB) for both training and testing.

2. **Train the Model**

```bash
python train.py
```

Starts training the Edge-Guided Multi-Branch network.

3. **Test the Model**

```bash
python test.py
```

Uses the trained model to perform enhancement on the test set and saves the results.

---

You may optionally add:

* Training parameters (batch size, learning rate, epochs)
* Pretrained model checkpoints and instructions to load them
* Output/result directory structure
* Example comparison results (before/after images)
