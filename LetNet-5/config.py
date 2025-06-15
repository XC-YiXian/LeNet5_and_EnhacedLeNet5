# -*- coding: gb2312 -*-
import torch

# ·������
PATHS = {
    'train_images': r'C:\Users\13157\Desktop\Study\����\�˹�����\dataset\Fashion MNIST\train-images-idx3-ubyte',
    'train_labels': r'C:\Users\13157\Desktop\Study\����\�˹�����\dataset\Fashion MNIST\train-labels-idx1-ubyte',
    'test_images': r'C:\Users\13157\Desktop\Study\����\�˹�����\dataset\Fashion MNIST\t10k-images-idx3-ubyte',
    'test_labels': r'C:\Users\13157\Desktop\Study\����\�˹�����\dataset\Fashion MNIST\t10k-labels-idx1-ubyte'
}

# ѵ������
TRAIN_CONFIG = {
    'batch_size': 128,          # ѵ��ʱ��������С��ÿ�ε����������������
    'num_epochs': 30,          # ѵ�����������������ݼ�����ѵ���Ĵ�����
    'learning_rate': 0.001,    # ѧϰ�ʣ����Ʋ������µĲ�����
    'test_size': 0.2,          # ���Լ�������20%������Ϊ���Լ���
    'random_state': 42,         # ������ӣ���֤ʵ��ɸ����ԣ�
    'weight_decay': 1e-4,       # Ȩ��˥����L2����ϵ����
    'early_stopping_patience': 5,  # ��ǰֹͣ������ֵ��������������֤��û��������ֹͣѵ����
    'augmentation': True        # �Ƿ�ʹ��������ǿ��True��ʾʹ�ã�False��ʾ��ʹ�ã�
}

# �豸����
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ����ǩ
LABELS_MAP = (
    "T-shirt", "Trouser", "Pullover", "Dress", "Coat", 
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
)