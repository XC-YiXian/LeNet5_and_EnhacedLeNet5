import torch
from torch.utils.data import Dataset
import numpy as np
from torchvision import transforms

class FashionMNIST(Dataset):
    def __init__(self, images, labels, augment=False):
        self.images = images.copy()
        self.labels = labels
        self.augment = augment
        
        # ����������ǿ�任
        self.transform = transforms.Compose([
            transforms.ToPILImage(),  # �Ƚ�numpy����ת��ΪPILͼ��
            transforms.RandomRotation(degrees=15),  # �����ת��15��
            transforms.RandomHorizontalFlip(p=0.5),  # 50%����ˮƽ��ת
            transforms.ToTensor(),  # ת��������
            transforms.Normalize((0.5,), (0.5,))  # ��һ����[-1,1]
        ])
        
        # �����任����������ǿ��
        self.basic_transform = transforms.Compose([
            transforms.ToTensor(),  # ת��Ϊ����
            transforms.Normalize((0.5,), (0.5,))  # ��һ��
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