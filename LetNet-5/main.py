# -*- coding: gb2312 -*-
#%%
import torch
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# 导入自定义模块
from data.utils import read_images_ubyte, read_labels_ubyte
from data.dataset import FashionMNIST
from models.lenet5 import LeNet5, EnhancedLeNet5
from training.trainer import train_epoch
from training.evaluator import evaluate, predict_image
from config import PATHS, TRAIN_CONFIG, DEVICE, LABELS_MAP
from visualize import plot_multiclass_evaluation, visualize_predictions, plot_comparison

def main():
    # 初始化
    torch.manual_seed(TRAIN_CONFIG['random_state'])
    np.random.seed(TRAIN_CONFIG['random_state'])
    
    # 加载数据
    train_images = read_images_ubyte(PATHS['train_images'])
    train_labels = read_labels_ubyte(PATHS['train_labels'])
    test_images = read_images_ubyte(PATHS['test_images'])
    test_labels = read_labels_ubyte(PATHS['test_labels'])

    # 划分训练集和验证集
    train_images, val_images, train_labels, val_labels = train_test_split(
        train_images, train_labels, 
        test_size=TRAIN_CONFIG['test_size'], 
        random_state=TRAIN_CONFIG['random_state']
    )
    
    # 创建数据集和数据加载器
    train_set = FashionMNIST(train_images, train_labels, augment=TRAIN_CONFIG['augmentation'])  
    val_set = FashionMNIST(val_images, val_labels)
    test_set = FashionMNIST(test_images, test_labels)
    
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=TRAIN_CONFIG['batch_size'], shuffle=True)
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=TRAIN_CONFIG['batch_size'], shuffle=False)
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=TRAIN_CONFIG['batch_size'], shuffle=False)
    
    # 初始化模型、损失函数和优化器
    models = {
        'LeNet5': LeNet5().to(DEVICE),
        'EnhancedLeNet5': EnhancedLeNet5().to(DEVICE)
    }
    history = {
        'LeNet5': {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []},
        'EnhancedLeNet5': {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    }


    criterion = torch.nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=TRAIN_CONFIG['learning_rate'],weight_decay=TRAIN_CONFIG['weight_decay'])
    
    # 训练循环
    for model_name, model in models.items():
        best_val_acc = 0.0
        print(f"\nTraining {model_name}...")
        optimizer = torch.optim.Adam(model.parameters(), lr=TRAIN_CONFIG['learning_rate'])
        
        for epoch in range(TRAIN_CONFIG['num_epochs']):
            # 训练并返回损失和准确率
            train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
            # 验证并返回损失和准确率
            val_loss, val_acc = evaluate(model, val_loader, criterion, DEVICE)  # 修改evaluate函数以返回损失
            history[model_name]['train_loss'].append(train_loss)
            history[model_name]['val_loss'].append(val_loss)
            history[model_name]['train_acc'].append(train_acc)
            history[model_name]['val_acc'].append(val_acc)

            # 保存最佳模型
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                torch.save(model.state_dict(), f'best_{model_name}_model_state.pth')
                torch.save(model, f'best_{model_name}_model.pth')
                print(f'Epoch [{epoch+1}/{TRAIN_CONFIG["num_epochs"]}], '
                f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, '
                f'Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}')
            else:
                patience_counter += 1
                print(f'Epoch [{epoch+1}/{TRAIN_CONFIG["num_epochs"]}], '
                f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, '
                f'Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f} - No improvement')
                if patience_counter >= TRAIN_CONFIG['early_stopping_patience']:
                    break
        # 类别名称
        class_names = list(LABELS_MAP)
        if model_name == 'EnhancedLeNet5':
            # 在测试集上获取预测结果和概率
            model.eval()
            all_preds = []
            all_probs = []
            all_labels = []

            with torch.no_grad():
                for images, labels in test_loader:
                    images = images.to(DEVICE)
                    outputs = model(images)
                    _, preds = torch.max(outputs, 1)
                    probs = torch.nn.functional.softmax(outputs, dim=1)
                    
                    all_preds.extend(preds.cpu().numpy())
                    all_probs.extend(probs.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())

            # 转换为numpy数组
            y_true = np.array(all_labels)
            y_pred = np.array(all_preds)
            y_scores = np.array(all_probs)
                # 绘制完整评估图
            plot_multiclass_evaluation(
                y_true=y_true,
                y_pred=y_pred,
                y_scores=y_scores,
                classes=class_names,
                train_loss=history[model_name]['train_loss'],
                val_loss=history[model_name]['val_loss'],
                train_acc=history[model_name]['train_acc'],
                val_acc=history[model_name]['val_acc'],
                figsize=(20, 15),
                save_path="enhanced_lenet_evaluation.png",
                combined=False
            )

            # 测试
            model.load_state_dict(torch.load(f'best_{model_name}_model_state.pth'))
            _, test_acc = evaluate(model, test_loader, criterion, DEVICE)
            print(f'Test Accuracy: {test_acc:.4f}')
            visualize_predictions(model, test_loader, class_names)
    plot_comparison(history, models, test_loader, list(LABELS_MAP))

        
        
    
    
'''
def main():
    # 初始化
    torch.manual_seed(TRAIN_CONFIG['random_state'])
    np.random.seed(TRAIN_CONFIG['random_state'])
    
    # 加载数据
    train_images = read_images_ubyte(PATHS['train_images'])
    train_labels = read_labels_ubyte(PATHS['train_labels'])
    test_images = read_images_ubyte(PATHS['test_images'])
    test_labels = read_labels_ubyte(PATHS['test_labels'])

    # 划分训练集和验证集
    train_images, val_images, train_labels, val_labels = train_test_split(
        train_images, train_labels, 
        test_size=TRAIN_CONFIG['test_size'], 
        random_state=TRAIN_CONFIG['random_state']
    )
    
    # 创建数据集和数据加载器
    train_set = FashionMNIST(train_images, train_labels, augment=TRAIN_CONFIG['augmentation'])  
    val_set = FashionMNIST(val_images, val_labels)
    test_set = FashionMNIST(test_images, test_labels)
    
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=TRAIN_CONFIG['batch_size'], shuffle=True)
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=TRAIN_CONFIG['batch_size'], shuffle=False)
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=TRAIN_CONFIG['batch_size'], shuffle=False)
    
    # 初始化模型、损失函数和优化器
    model = LeNet5().to(DEVICE)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=TRAIN_CONFIG['learning_rate'],weight_decay=TRAIN_CONFIG['weight_decay'])
    
    # 训练循环
    best_val_acc = 0.0
    train_loss_history = []
    train_acc_history = []  
    val_loss_history = []   
    val_acc_history = []

    for epoch in range(TRAIN_CONFIG['num_epochs']):
        # 训练并返回损失和准确率
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
        train_loss_history.append(train_loss)
        train_acc_history.append(train_acc)  # 记录训练准确率
        
        # 验证并返回损失和准确率
        val_loss, val_acc = evaluate(model, val_loader, criterion, DEVICE)  # 修改evaluate函数以返回损失
        val_loss_history.append(val_loss)    # 记录验证损失
        val_acc_history.append(val_acc)
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), 'best_lenet5_model_state.pth')
            torch.save(model, 'best_lenet5_model.pth')
            print(f'Epoch [{epoch+1}/{TRAIN_CONFIG["num_epochs"]}], '
            f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, '
            f'Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}')
        else:
            patience_counter += 1
            print(f'Epoch [{epoch+1}/{TRAIN_CONFIG["num_epochs"]}], '
            f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, '
            f'Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f} - No improvement')
            if patience_counter >= TRAIN_CONFIG['early_stopping_patience']:
                break

        
        
    
    # 可视化训练曲线
    # 在测试集上获取预测结果和概率
    model.eval()
    all_preds = []
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 转换为numpy数组
    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_scores = np.array(all_probs)

    # 类别名称
    class_names = list(LABELS_MAP)

    # 绘制完整评估图
    plot_multiclass_evaluation(
        y_true=y_true,
        y_pred=y_pred,
        y_scores=y_scores,
        classes=class_names,
        train_loss=train_loss_history,
        val_loss=val_loss_history,  # 如果您没有记录验证损失
        train_acc=train_acc_history, # 如果您没有记录训练准确率
        val_acc=val_acc_history,
        figsize=(20, 15),
        save_path="full_evaluation.png",
        combined= False
    )

    # 测试
    model.load_state_dict(torch.load('best_lenet5_model_state.pth'))
    _, test_acc = evaluate(model, test_loader, criterion, DEVICE)
    print(f'Test Accuracy: {test_acc:.4f}')
    visualize_predictions(model, test_loader, class_names)
'''

if __name__ == "__main__":
    main()