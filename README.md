# Edge-Guided Multi-Branch Network for Underwater Image Enhancement

## 简介

本项目实现了基于 **Edge-Guided Multi-Branch Network** 的水下图像增强方法，通过 Commonality-Specificity 模型对多分支特征进行融合，提高水下图像的视觉质量。该方法可以有效恢复水下图像的细节和颜色信息，同时抑制水下光照衰减和颜色偏差。

## 架构图

![architecture](docs/architecture.png)
*图示：Edge-Guided Multi-Branch 网络结构，包括 RGB 分支、边缘分支和融合模块。*

> 请将 `docs/architecture.png` 替换为你的实际网络结构图。

## 环境要求

* Python 3.11
* PyTorch 2.1.0

创建环境示例：

```bash
conda create -n uienv python=3.11
conda activate uienv
pip install torch==2.1.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

## 数据集

请将 **UIEB** 数据集放置在项目根目录下：

```
project_root/
└── UIEB/
```

## 使用说明

1. **生成边缘图**

```bash
python utils.py
```

生成 UIEB 数据集对应的边缘图，用于训练和测试。

2. **训练模型**

```bash
python train.py
```

开始训练 Edge-Guided Multi-Branch 网络。

3. **测试模型**

```bash
python test.py
```

使用训练好的模型对测试集进行增强并保存结果。

---

你可以根据需要补充：

* 训练参数说明（batch size、学习率等）
* 预训练模型路径
* 输出结果目录
* 样例效果对比图
