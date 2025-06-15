# LeNet5 和EnhancedLeNet5 在 FashionMNIST 上的应用  🚀

![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-EE4C2C.svg)
![License](https://img.shields.io/badge/License-MIT-blue.svg)
![FashionMNIST](https://img.shields.io/badge/Dataset-FashionMNIST-lightgrey)

本项目实现了原始的 LeNet-5 架构和一个增强版本，用于对 FashionMNIST 数据集中的图像进行分类。

## 🌟 主要特性

- LeNet-5 的实现 (1998 Yann LeCun)
- 增强型 LeNet-5 的实现
- 在 FashionMNIST 数据集上进行训练和评估。

### EnhancedLeNet5 细节

EnhancedLeNet5 包括以下修改：

```python
# 增强型 LeNet5
self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)  # 卷积核深度和宽度均数量增加
self.bn1 = nn.BatchNorm2d(32)  # 批量归一化
self.pool = nn.AdaptiveAvgPool2d((4, 4))  # 自适应池化
self.dropout = nn.Dropout(0.5)  # Dropout 层
```

## 🛠️ 安装

1.  **先决条件:**

    *   Python 3.6+
    *   PyTorch 1.9+ (或兼容版本)
    *   `torchvision`
    *   其他常用的 Python 库 (例如, `numpy`, `matplotlib`)。 使用 pip 安装它们:

    ```bash
    pip install torch torchvision numpy matplotlib
    ```

2.  **克隆仓库:**

    ```bash
    git clone https://github.com/XC-YiXian/LeNet5_and_EnhacedLeNet5.git
    cd LeNet5_and_EnhacedLeNet5
    ```

## ⚙️ 项目结构

```
LeNet5_and_EnhacedLeNet5/
├── dataset/
│ └── Fashion MNIST/ # Fashion MNIST数据集存放路径
│ │ ├── train-images-idx3-ubyte
│ │ ├── train-labels-idx1-ubyte
│ │ ├── t10k-images-idx3-ubyte
│ │ └── t10k-labels-idx1-ubyte
├── LetNet-5/
│ ├── config.py # 配置文件，包含路径、训练参数等
│ ├── main.py # 主脚本，运行训练和评估流程
│ ├── models/
│ │ └── lenet5.py # LeNet5 和增强版 LeNet5 模型定义
│ ├── data/
│ │ ├── dataset.py # Fashion MNIST 自定义数据集类
│ │ └── utils.py # 读取 ubyte 文件的工具函数
│ ├── training/
│ │ ├── trainer.py # 模型训练函数
│ │ └── evaluator.py # 模型评估函数
│ ├── visualize.py # 可视化训练和评估结果的函数
│ └── requirements.txt # 所需 Python 包列表
└── ...
```

## 💾 数据准备

本项目使用 FashionMNIST 数据集，`torchvision` 库可以自动下载和加载该数据集。  在 `main.py` 中，数据集会被自动下载到指定目录（默认为当前目录）。

## 🚀 运行步骤

1.  **安装依赖:**

    ```bash
    pip install -r requirements.txt
    ```

2.  **运行主程序:**

    ```bash
    python main.py
    ```

    你可以通过命令行参数来配置训练过程，例如：

    *   `--model`:  选择使用的模型 (LeNet5 或 EnhancedLeNet5)。
    *   `--epochs`:  训练的轮数。
    *   `--batch_size`:  批量大小。
    *   `--learning_rate`:  学习率。

    例如，使用增强型 LeNet5 训练 20 轮，批量大小为 64，学习率为 0.001：

    ```bash
    python main.py --model EnhancedLeNet5 --epochs 20 --batch_size 64 --learning_rate 0.001
    ```

## 🤝 贡献

欢迎贡献! 如果你想为本项目做出贡献，请遵循以下指南:

1.  Fork 仓库.
2.  为你想要实现的功能或 bug 修复创建一个新的分支.
3.  进行修改并提交，提交信息应具有描述性.
4.  提交一个 pull request.

## 🧑‍💻 贡献者

*   XC-YiXian
```
