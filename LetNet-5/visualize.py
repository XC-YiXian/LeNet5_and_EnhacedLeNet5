# -*- coding: gb2312 -*-
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score
)
from itertools import cycle
from sklearn.preprocessing import label_binarize
from config import DEVICE

def plot_training_curves(train_loss, val_loss, train_acc, val_acc, ax=None):
    """绘制训练和验证的损失及准确率曲线"""
    if ax is None:
        fig, ax = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={'width_ratios': [1, 1]})  # Equal width ratios
    
    # 损失曲线
    ax[0].plot(train_loss, label='Train Loss')
    ax[0].plot(val_loss, label='Validation Loss')
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('Loss')
    ax[0].set_title('Training & Validation Loss')
    ax[0].legend()
    
    # 准确率曲线
    ax[1].plot(train_acc, label='Train Accuracy')
    ax[1].plot(val_acc, label='Validation Accuracy')
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('Accuracy')
    ax[1].set_title('Training & Validation Accuracy')
    ax[1].legend()
    
    return ax

def plot_confusion_matrix(y_true, y_pred, classes, ax=None):
    """绘制混淆矩阵"""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes, ax=ax)
    ax.set_title('Confusion Matrix')
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])  
    return ax

def plot_classification_report(y_true, y_pred, classes, ax=None):
    """绘制分类报告热图(准确率、召回率、F1值)"""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    report = classification_report(y_true, y_pred, target_names=classes, output_dict=True)
    df_report = pd.DataFrame(report).iloc[:-1, :].T
    
    sns.heatmap(df_report, annot=True, cmap='YlOrRd', ax=ax)
    ax.set_title('Classification Report (Precision, Recall, F1)')
    
    return ax

def plot_roc_curves(y_true, y_scores, classes, ax=None):
    """绘制多分类ROC曲线"""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    n_classes = len(classes)
    y_true_bin = label_binarize(y_true, classes=np.arange(n_classes))
    
    # 计算每个类的ROC曲线和AUC
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_scores[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # 计算微平均ROC曲线和AUC
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), y_scores.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
    # 绘制所有ROC曲线
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green', 'red',
                   'purple', 'pink', 'brown', 'gray', 'olive'])
    
    lines = []
    labels = []
    
    for i, color in zip(range(n_classes), colors):
        l, = ax.plot(fpr[i], tpr[i], color=color, lw=2)
        lines.append(l)
        labels.append('ROC for {0} (AUC = {1:0.2f})'
                     ''.format(classes[i], roc_auc[i]))
    
    ax.plot([0, 1], [0, 1], 'k--', lw=2)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic (ROC)')
    ax.legend(lines, labels, loc='center left',        # 图例内容的对齐基准点（居中）
        bbox_to_anchor=(1.02, 0.5),  # 图例框的位置 (x=1.02 表示图外右侧，y=0.5 垂直居中)
        borderaxespad=0.1,        # 图例框与轴的间距
        frameon=True              # 显示边框
            ) 
    plt.tight_layout(rect=[0, 0, 0.99, 1])  # rect=[left, bottom, right, top]

    return ax

def plot_pr_curves(y_true, y_scores, classes, ax=None):
    """绘制多分类PR曲线"""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    n_classes = len(classes)
    y_true_bin = label_binarize(y_true, classes=np.arange(n_classes))
    
    # 计算PR曲线
    precision = dict()
    recall = dict()
    average_precision = dict()
    
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(y_true_bin[:, i], y_scores[:, i])
        average_precision[i] = average_precision_score(y_true_bin[:, i], y_scores[:, i])
    
    # 微平均PR曲线
    precision["micro"], recall["micro"], _ = precision_recall_curve(
        y_true_bin.ravel(), y_scores.ravel())
    average_precision["micro"] = average_precision_score(y_true_bin, y_scores, average="micro")
    
    # 绘制PR曲线
    colors = cycle(['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal',
                    'red', 'green', 'blue', 'gray', 'purple'])
    
    lines = []
    labels = []
    
    for i, color in zip(range(n_classes), colors):
        l, = ax.plot(recall[i], precision[i], color=color, lw=2)
        lines.append(l)
        labels.append('PR for {0} (AP = {1:0.2f})'
                      ''.format(classes[i], average_precision[i]))
    
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_ylim([0.0, 1.05])
    ax.set_xlim([0.0, 1.0])
    ax.set_title('Precision-Recall Curve')
    ax.legend(lines, labels, loc='center left',        # 图例内容的对齐基准点（居中）
        bbox_to_anchor=(1.02, 0.5),  # 图例框的位置 (x=1.02 表示图外右侧，y=0.5 垂直居中)
        borderaxespad=0.1,        # 图例框与轴的间距
        frameon=True              # 显示边框)
    )
    plt.tight_layout(rect=[0, 0, 0.99, 1])  # rect=[left, bottom, right, top]
    
    return ax

def visualize_predictions(model, test_loader, class_names, num_samples=5):
    """
    可视化预测结果，确保正确和错误的样本来自不同真实类别
    参数:
        model: 训练好的模型
        test_loader: 测试集数据加载器
        class_names: 类别名称列表（按类别索引排序）
        num_samples: 每种情况显示的样本数（最终显示2*num_samples张图）
    """
    # 创建画布（2行，每行num_samples列）
    fig, axes = plt.subplots(2, num_samples, figsize=(15, 6))
    fig.suptitle('Analysis of model prediction results', fontsize=16)
    
    # 初始化存储
    correct_samples = []  # 存储（图像, 真实标签, 预测标签）
    wrong_samples = []
    used_true_labels = []  # 记录已使用的真实标签
    
    # 遍历测试集
    for i in range(len(test_loader.dataset)):
        image, true_label = test_loader.dataset[i]
        
        # 如果该真实标签已使用，跳过
        if true_label in used_true_labels:
            continue
            
        # 预测
        image_tensor = image.unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            pred_label = model(image_tensor).argmax().item()
        
        # 分类存储
        if true_label == pred_label:
            if len(correct_samples) < num_samples:
                correct_samples.append((image, true_label, pred_label))
                used_true_labels.append(true_label)
        else:
            if len(wrong_samples) < num_samples:
                wrong_samples.append((image, true_label, pred_label))
                used_true_labels.append(true_label)
        
        # 提前终止条件
        if len(correct_samples) >= num_samples and len(wrong_samples) >= num_samples:
            break
        
    # 绘制正确预测样本（第一行）
    for i, (image, true_label, pred_label) in enumerate(correct_samples):
        axes[0, i].imshow(image.squeeze(), cmap='gray')
        axes[0, i].set_title(
            f"True: {class_names[true_label]}\nPred: {class_names[pred_label]}",
            color='black'
        )
        axes[0, i].axis('off')
    
    # 绘制错误预测样本（第二行）
    for i, (image, true_label, pred_label) in enumerate(wrong_samples):
        axes[1, i].imshow(image.squeeze(), cmap='gray')
        axes[1, i].set_title(
            f"True: {class_names[true_label]}\nPred: {class_names[pred_label]}",
            color='red'
        )
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.show()


def plot_class_distribution(y_train, y_test, classes, figsize=(12, 5)):
    """
    绘制训练集和测试集的类别分布直方图
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # 训练集分布
    train_counts = np.bincount(y_train)
    ax1.bar(range(len(classes)), train_counts, color='skyblue')
    ax1.set_title('Training Set Class Distribution')
    ax1.set_xlabel('Class')
    ax1.set_ylabel('Count')
    ax1.set_xticks(range(len(classes)))
    ax1.set_xticklabels(classes, rotation=45)
    
    # 添加数值标签
    for i, count in enumerate(train_counts):
        ax1.text(i, count + 50, str(count), ha='center', va='bottom')
    
    # 测试集分布
    test_counts = np.bincount(y_test)
    ax2.bar(range(len(classes)), test_counts, color='salmon')
    ax2.set_title('Test Set Class Distribution')
    ax2.set_xlabel('Class')
    ax2.set_ylabel('Count')
    ax2.set_xticks(range(len(classes)))
    ax2.set_xticklabels(classes, rotation=45)
    
    # 添加数值标签
    for i, count in enumerate(test_counts):
        ax2.text(i, count + 20, str(count), ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()

def visualize_augmentation(dataset, index=0, num_examples=1):
    plt.figure(figsize=(10, 3*num_examples))
    
    for i in range(num_examples):
        idx = index + i
        # 获取原始图像（不增强）
        original_img = dataset.basic_transform(dataset.images[idx]).numpy().squeeze()
        
        # 获取增强后的图像
        dataset.augment = True
        augmented_img = dataset[idx][0].numpy().squeeze()
        dataset.augment = False
        
        # 显示原始图像
        plt.subplot(num_examples, 2, 2*i+1)
        plt.imshow(original_img, cmap='gray')
        plt.title(f'Original Image {idx}')
        plt.axis('off')
        
        # 显示增强后的图像
        plt.subplot(num_examples, 2, 2*i+2)
        plt.imshow(augmented_img, cmap='gray')
        plt.title(f'Augmented Image {idx}')
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def plot_comparison(history, models, test_loader, class_names, focus_class='Shirt'):
    """绘制两个模型的对比曲线（仅针对Shirt类别）"""
    # 获取Shirt类别的索引
    try:
        focus_idx = class_names.index(focus_class)
    except ValueError:
        raise ValueError(f"'{focus_class}' not found in class_names. Available classes: {class_names}")

    
    # 准备测试集预测结果（仅保留Shirt类别的数据）
    y_trues, y_scores_focus = {}, {}
    for name, model in models.items():
        model.eval()
        all_scores, all_labels = [], []
        with torch.no_grad():
            for images, labels in test_loader:
                outputs = model(images.to(DEVICE))
                all_scores.extend(torch.softmax(outputs, dim=1).cpu().numpy())
                all_labels.extend(labels.numpy())
        
        # 转换为numpy数组并筛选目标类别
        y_true = np.array(all_labels)
        y_scores = np.array(all_scores)
        
        # 二值化标签（目标类别 vs 非目标类别）
        y_trues[name] = (y_true == focus_idx).astype(int)
        y_scores_focus[name] = y_scores[:, focus_idx]  

    # 1. 训练曲线对比（保持不变）
    plt.figure(figsize=(12, 5))
    colors = {'LeNet5': 'blue', 'EnhancedLeNet5': 'red'}
    
    # Loss对比
    plt.subplot(1, 2, 1)
    for name in models:
        plt.plot(history[name]['train_loss'], '--', color=colors[name], label=f'{name} Train')
        plt.plot(history[name]['val_loss'], '-', color=colors[name], label=f'{name} Val')
    plt.title('Loss Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Accuracy对比
    plt.subplot(1, 2, 2)
    for name in models:
        plt.plot(history[name]['train_acc'], '--', color=colors[name], label=f'{name} Train')
        plt.plot(history[name]['val_acc'], '-', color=colors[name], label=f'{name} Val')
    plt.title('Accuracy Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 2. Shirt类别的ROC曲线
    plt.figure(figsize=(8, 6))
    for name in models:
        fpr, tpr, _ = roc_curve(y_trues[name], y_scores_focus[name])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=colors[name], 
                label=f'{name} (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title('ROC Curve Comparison (Shirt Class)')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.show()

    # 3. Shirt类别的PR曲线
    plt.figure(figsize=(8, 6))
    for name in models:
        precision, recall, _ = precision_recall_curve(y_trues[name], y_scores_focus[name])
        ap = average_precision_score(y_trues[name], y_scores_focus[name])
        plt.plot(recall, precision, color=colors[name], 
                label=f'{name} (AP = {ap:.2f})')
    plt.title('Precision-Recall Curve (Shirt Class)')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()
    plt.show()

def plot_multiclass_evaluation(y_true, y_pred, y_scores, classes,
                             train_loss=None, val_loss=None,
                             train_acc=None, val_acc=None,
                             figsize=(20, 15), save_path=None, combined=True):
    """
    多分类模型评估综合可视化(主函数)
    
    参数:
        y_true: 真实标签
        y_pred: 预测标签
        y_scores: 预测概率 (shape: [n_samples, n_classes])
        classes: 类别名称列表
        train_loss: 训练损失历史(可选)
        val_loss: 验证损失历史(可选)
        train_acc: 训练准确率历史(可选)
        val_acc: 验证准确率历史(可选)
        figsize: 图像大小
        save_path: 保存路径(可选)
    """
    if not combined:
        # 独立显示每个图表
        plot_confusion_matrix(y_true, y_pred, classes)
        plot_classification_report(y_true, y_pred, classes)
        plot_roc_curves(y_true, y_scores, classes)
        plot_pr_curves(y_true, y_scores, classes)
        if train_loss is not None:
            plot_training_curves(train_loss, val_loss, train_acc, val_acc)
        plt.show()  # 统一触发显示
        return  # 提前退出，不执行后续合并逻辑

    fig = plt.figure(figsize=figsize)
    
    # 根据是否提供训练历史数据决定布局
    if train_loss is not None and val_loss is not None and train_acc is not None and val_acc is not None:
        # 有训练历史数据 - 3行3列布局
        gs = fig.add_gridspec(3, 3)
        
        # 1. 训练曲线
        ax1 = fig.add_subplot(gs[0, :2])  # 训练损失和准确率(左2列)
        ax2 = fig.add_subplot(gs[0, 2])   # 验证准确率(右1列)
        plot_training_curves(train_loss, val_loss, train_acc, val_acc, ax=[ax1, ax2])
        
        # 2. 混淆矩阵
        ax3 = fig.add_subplot(gs[1, 0])
        plot_confusion_matrix(y_true, y_pred, classes, ax=ax3)
        
        # 3. 分类报告热图
        ax4 = fig.add_subplot(gs[1, 1])
        plot_classification_report(y_true, y_pred, classes, ax=ax4)
        
        # 4. ROC曲线
        ax5 = fig.add_subplot(gs[1, 2])
        plot_roc_curves(y_true, y_scores, classes, ax=ax5)
        
        # 5. PR曲线
        ax6 = fig.add_subplot(gs[2, :])
        plot_pr_curves(y_true, y_scores, classes, ax=ax6)
        
    else:
        # 没有训练历史数据 - 2行3列布局
        gs = fig.add_gridspec(2, 3)
        
        # 1. 混淆矩阵
        ax1 = fig.add_subplot(gs[0, 0])
        plot_confusion_matrix(y_true, y_pred, classes, ax=ax1)
        
        # 2. 分类报告热图
        ax2 = fig.add_subplot(gs[0, 1])
        plot_classification_report(y_true, y_pred, classes, ax=ax2)
        
        # 3. ROC曲线
        ax3 = fig.add_subplot(gs[0, 2])
        plot_roc_curves(y_true, y_scores, classes, ax=ax3)
        
        # 4. PR曲线
        ax4 = fig.add_subplot(gs[1, :])
        plot_pr_curves(y_true, y_scores, classes, ax=ax4)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)  # Added dpi for better quality
    plt.show()