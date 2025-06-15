# -*- coding: gb2312 -*-
import torch

# 路径配置
PATHS = {
    'train_images': r'C:\Users\13157\Desktop\Study\大三\人工智能\dataset\Fashion MNIST\train-images-idx3-ubyte',
    'train_labels': r'C:\Users\13157\Desktop\Study\大三\人工智能\dataset\Fashion MNIST\train-labels-idx1-ubyte',
    'test_images': r'C:\Users\13157\Desktop\Study\大三\人工智能\dataset\Fashion MNIST\t10k-images-idx3-ubyte',
    'test_labels': r'C:\Users\13157\Desktop\Study\大三\人工智能\dataset\Fashion MNIST\t10k-labels-idx1-ubyte'
}

# 训练配置
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

# 设备配置
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 类别标签
LABELS_MAP = (
    "T-shirt", "Trouser", "Pullover", "Dress", "Coat", 
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
)