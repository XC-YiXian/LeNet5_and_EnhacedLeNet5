# LeNet5 和EnhancedLeNet5 在 FashionMNIST 上的应用  🚀

![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-EE4C2C.svg)
![License](https://img.shields.io/badge/License-MIT-blue.svg)
![FashionMNIST](https://img.shields.io/badge/Dataset-FashionMNIST-lightgrey)

本项目实现了原始的 LeNet-5 架构和一个增强版本，用于对 FashionMNIST 数据集中的图像进行分类。

## 🌟 主要特性

- LeNet-5 的实现 (1998 Yann LeCun)
- EnhancedLeNet5 的实现
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
    *   其他常用的 Python 库：
    *       matplotlib==3.10.3
            numpy==2.3.0
            pandas==2.3.0
            scikit_learn==1.7.0
            seaborn==0.13.2
            torch==2.7.0+cu118
            tqdm==4.67.1

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
│ │ └── lenet5.py # LeNet5 和EnhancedLeNet5 模型定义
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
2.  **数据集准备:**
   下载Realses中的数据集，按照项目结构放在指定位置（也可以自选位置，而后在config.py中修改数据集路径）

3.  **运行主程序:**

    ```bash
    python main.py
    ```

    你可以通过修改config.py中的训练参数来配置训练过程：
    ```bash
    TRAIN_CONFIG = {
    'batch_size': 128,          # 训练时的批量大小（每次迭代处理的样本数）
    'num_epochs': 30,          # 训练总轮数（整个数据集完整训练的次数）
    'learning_rate': 0.001,    # 学习率（控制参数更新的步长）
    'test_size': 0.2,          # 测试集比例（20%数据作为测试集）
    'random_state': 42,         # 随机种子（保证实验可复现性）
    'weight_decay': 1e-4,       # 权重衰减（L2正则化系数）
    'early_stopping_patience': 5,  # 提前停止的耐心值（连续多少轮验证集没有提升后停止训练）
    'augmentation': True        # 是否使用数据增强（True表示使用，False表示不使用）
    }
    ```


## 🧑‍💻 贡献者

*   XC-YiXian
```
