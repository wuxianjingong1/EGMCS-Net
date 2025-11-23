# Edge-Guided Multi-Branch Network for Underwater Image Enhancement

## Introduction

This project implements an **Edge-Guided Multi-Branch Network** for underwater image enhancement. It leverages a Commonality-Specificity model to fuse multi-branch features, improving the visual quality of underwater images. The method effectively restores image details and colors while reducing underwater light attenuation and color distortion.

## Architecture

![architecture](docs/architecture.png)
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

## Dataset

Place the **UIEB** dataset in the project root directory:

```
project_root/
└── UIEB/
```

## Usage

1. **Generate Edge Maps**

```bash
python utils.py
```

Generates edge maps for the UIEB dataset for training and testing.

2. **Train the Model**

```bash
python train.py
```

Starts training the Edge-Guided Multi-Branch network.

3. **Test the Model**

```bash
python test.py
```

Tests the trained model on the dataset and saves the enhanced results.

---

You can optionally add:

* Training parameters (batch size, learning rate, etc.)
* Pretrained model paths
* Output directories
* Example comparison results
