# LeNet5 and EnhancedLeNet5 on FashionMNIST

![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-EE4C2C.svg)
![License](https://img.shields.io/badge/License-MIT-blue.svg)
![FashionMNIST](https://img.shields.io/badge/Dataset-FashionMNIST-lightgrey)

经典LeNet5及其现代化改进版的对比实现，针对FashionMNIST数据集优化。

## 🚀 主要特性

- **双模型对比**：
  - 原始LeNet5 (1998 Yann LeCun)
  - 增强版EnhancedLeNet5 (现代改进)
- **关键技术改进**：
  ```python
  # EnhancedLeNet5的核心改进
  self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)  # 新增卷积层
  self.bn1 = nn.BatchNorm2d(32)  # 批归一化
  self.pool = nn.AdaptiveAvgPool2d((4, 4))  # 自适应池化
  self.dropout = nn.Dropout(0.5)  # 正则化
