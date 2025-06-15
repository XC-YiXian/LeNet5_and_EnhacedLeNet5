import torch
from torch.utils.data import Dataset
import numpy as np
from torchvision import transforms

class FashionMNIST(Dataset):
    def __init__(self, images, labels, augment=False):
        self.images = images.copy()
        self.labels = labels
        self.augment = augment
        
        # 定义数据增强变换
        self.transform = transforms.Compose([
            transforms.ToPILImage(),  # 先将numpy数组转换为PIL图像
            transforms.RandomRotation(degrees=15),  # 随机旋转±15度
            transforms.RandomHorizontalFlip(p=0.5),  # 50%概率水平翻转
            transforms.ToTensor(),  # 转换回张量
            transforms.Normalize((0.5,), (0.5,))  # 归一化到[-1,1]
        ])
        
        # 基本变换（无数据增强）
        self.basic_transform = transforms.Compose([
            transforms.ToTensor(),  # 转换为张量
            transforms.Normalize((0.5,), (0.5,))  # 归一化
        ])

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        
        if self.augment:
            image = self.transform(image)
        else:
            image = self.basic_transform(image)
            
        label = torch.tensor(label).long()
        return image, label