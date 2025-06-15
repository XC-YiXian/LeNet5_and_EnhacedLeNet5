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
    """����ѵ������֤����ʧ��׼ȷ������"""
    if ax is None:
        fig, ax = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={'width_ratios': [1, 1]})  # Equal width ratios
    
    # ��ʧ����
    ax[0].plot(train_loss, label='Train Loss')
    ax[0].plot(val_loss, label='Validation Loss')
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('Loss')
    ax[0].set_title('Training & Validation Loss')
    ax[0].legend()
    
    # ׼ȷ������
    ax[1].plot(train_acc, label='Train Accuracy')
    ax[1].plot(val_acc, label='Validation Accuracy')
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('Accuracy')
    ax[1].set_title('Training & Validation Accuracy')
    ax[1].legend()
    
    return ax

def plot_confusion_matrix(y_true, y_pred, classes, ax=None):
    """���ƻ�������"""
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
    """���Ʒ��౨����ͼ(׼ȷ�ʡ��ٻ��ʡ�F1ֵ)"""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    report = classification_report(y_true, y_pred, target_names=classes, output_dict=True)
    df_report = pd.DataFrame(report).iloc[:-1, :].T
    
    sns.heatmap(df_report, annot=True, cmap='YlOrRd', ax=ax)
    ax.set_title('Classification Report (Precision, Recall, F1)')
    
    return ax

def plot_roc_curves(y_true, y_scores, classes, ax=None):
    """���ƶ����ROC����"""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    n_classes = len(classes)
    y_true_bin = label_binarize(y_true, classes=np.arange(n_classes))
    
    # ����ÿ�����ROC���ߺ�AUC
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_scores[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # ����΢ƽ��ROC���ߺ�AUC
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), y_scores.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
    # ��������ROC����
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
    ax.legend(lines, labels, loc='center left',        # ͼ�����ݵĶ����׼�㣨���У�
        bbox_to_anchor=(1.02, 0.5),  # ͼ�����λ�� (x=1.02 ��ʾͼ���Ҳ࣬y=0.5 ��ֱ����)
        borderaxespad=0.1,        # ͼ��������ļ��
        frameon=True              # ��ʾ�߿�
            ) 
    plt.tight_layout(rect=[0, 0, 0.99, 1])  # rect=[left, bottom, right, top]

    return ax

def plot_pr_curves(y_true, y_scores, classes, ax=None):
    """���ƶ����PR����"""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    n_classes = len(classes)
    y_true_bin = label_binarize(y_true, classes=np.arange(n_classes))
    
    # ����PR����
    precision = dict()
    recall = dict()
    average_precision = dict()
    
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(y_true_bin[:, i], y_scores[:, i])
        average_precision[i] = average_precision_score(y_true_bin[:, i], y_scores[:, i])
    
    # ΢ƽ��PR����
    precision["micro"], recall["micro"], _ = precision_recall_curve(
        y_true_bin.ravel(), y_scores.ravel())
    average_precision["micro"] = average_precision_score(y_true_bin, y_scores, average="micro")
    
    # ����PR����
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
    ax.legend(lines, labels, loc='center left',        # ͼ�����ݵĶ����׼�㣨���У�
        bbox_to_anchor=(1.02, 0.5),  # ͼ�����λ�� (x=1.02 ��ʾͼ���Ҳ࣬y=0.5 ��ֱ����)
        borderaxespad=0.1,        # ͼ��������ļ��
        frameon=True              # ��ʾ�߿�)
    )
    plt.tight_layout(rect=[0, 0, 0.99, 1])  # rect=[left, bottom, right, top]
    
    return ax

def visualize_predictions(model, test_loader, class_names, num_samples=5):
    """
    ���ӻ�Ԥ������ȷ����ȷ�ʹ�����������Բ�ͬ��ʵ���
    ����:
        model: ѵ���õ�ģ��
        test_loader: ���Լ����ݼ�����
        class_names: ��������б��������������
        num_samples: ÿ�������ʾ����������������ʾ2*num_samples��ͼ��
    """
    # ����������2�У�ÿ��num_samples�У�
    fig, axes = plt.subplots(2, num_samples, figsize=(15, 6))
    fig.suptitle('Analysis of model prediction results', fontsize=16)
    
    # ��ʼ���洢
    correct_samples = []  # �洢��ͼ��, ��ʵ��ǩ, Ԥ���ǩ��
    wrong_samples = []
    used_true_labels = []  # ��¼��ʹ�õ���ʵ��ǩ
    
    # �������Լ�
    for i in range(len(test_loader.dataset)):
        image, true_label = test_loader.dataset[i]
        
        # �������ʵ��ǩ��ʹ�ã�����
        if true_label in used_true_labels:
            continue
            
        # Ԥ��
        image_tensor = image.unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            pred_label = model(image_tensor).argmax().item()
        
        # ����洢
        if true_label == pred_label:
            if len(correct_samples) < num_samples:
                correct_samples.append((image, true_label, pred_label))
                used_true_labels.append(true_label)
        else:
            if len(wrong_samples) < num_samples:
                wrong_samples.append((image, true_label, pred_label))
                used_true_labels.append(true_label)
        
        # ��ǰ��ֹ����
        if len(correct_samples) >= num_samples and len(wrong_samples) >= num_samples:
            break
        
    # ������ȷԤ����������һ�У�
    for i, (image, true_label, pred_label) in enumerate(correct_samples):
        axes[0, i].imshow(image.squeeze(), cmap='gray')
        axes[0, i].set_title(
            f"True: {class_names[true_label]}\nPred: {class_names[pred_label]}",
            color='black'
        )
        axes[0, i].axis('off')
    
    # ���ƴ���Ԥ���������ڶ��У�
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
    ����ѵ�����Ͳ��Լ������ֲ�ֱ��ͼ
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # ѵ�����ֲ�
    train_counts = np.bincount(y_train)
    ax1.bar(range(len(classes)), train_counts, color='skyblue')
    ax1.set_title('Training Set Class Distribution')
    ax1.set_xlabel('Class')
    ax1.set_ylabel('Count')
    ax1.set_xticks(range(len(classes)))
    ax1.set_xticklabels(classes, rotation=45)
    
    # �����ֵ��ǩ
    for i, count in enumerate(train_counts):
        ax1.text(i, count + 50, str(count), ha='center', va='bottom')
    
    # ���Լ��ֲ�
    test_counts = np.bincount(y_test)
    ax2.bar(range(len(classes)), test_counts, color='salmon')
    ax2.set_title('Test Set Class Distribution')
    ax2.set_xlabel('Class')
    ax2.set_ylabel('Count')
    ax2.set_xticks(range(len(classes)))
    ax2.set_xticklabels(classes, rotation=45)
    
    # �����ֵ��ǩ
    for i, count in enumerate(test_counts):
        ax2.text(i, count + 20, str(count), ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()

def visualize_augmentation(dataset, index=0, num_examples=1):
    plt.figure(figsize=(10, 3*num_examples))
    
    for i in range(num_examples):
        idx = index + i
        # ��ȡԭʼͼ�񣨲���ǿ��
        original_img = dataset.basic_transform(dataset.images[idx]).numpy().squeeze()
        
        # ��ȡ��ǿ���ͼ��
        dataset.augment = True
        augmented_img = dataset[idx][0].numpy().squeeze()
        dataset.augment = False
        
        # ��ʾԭʼͼ��
        plt.subplot(num_examples, 2, 2*i+1)
        plt.imshow(original_img, cmap='gray')
        plt.title(f'Original Image {idx}')
        plt.axis('off')
        
        # ��ʾ��ǿ���ͼ��
        plt.subplot(num_examples, 2, 2*i+2)
        plt.imshow(augmented_img, cmap='gray')
        plt.title(f'Augmented Image {idx}')
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def plot_comparison(history, models, test_loader, class_names, focus_class='Shirt'):
    """��������ģ�͵ĶԱ����ߣ������Shirt���"""
    # ��ȡShirt��������
    try:
        focus_idx = class_names.index(focus_class)
    except ValueError:
        raise ValueError(f"'{focus_class}' not found in class_names. Available classes: {class_names}")

    
    # ׼�����Լ�Ԥ������������Shirt�������ݣ�
    y_trues, y_scores_focus = {}, {}
    for name, model in models.items():
        model.eval()
        all_scores, all_labels = [], []
        with torch.no_grad():
            for images, labels in test_loader:
                outputs = model(images.to(DEVICE))
                all_scores.extend(torch.softmax(outputs, dim=1).cpu().numpy())
                all_labels.extend(labels.numpy())
        
        # ת��Ϊnumpy���鲢ɸѡĿ�����
        y_true = np.array(all_labels)
        y_scores = np.array(all_scores)
        
        # ��ֵ����ǩ��Ŀ����� vs ��Ŀ�����
        y_trues[name] = (y_true == focus_idx).astype(int)
        y_scores_focus[name] = y_scores[:, focus_idx]  

    # 1. ѵ�����߶Աȣ����ֲ��䣩
    plt.figure(figsize=(12, 5))
    colors = {'LeNet5': 'blue', 'EnhancedLeNet5': 'red'}
    
    # Loss�Ա�
    plt.subplot(1, 2, 1)
    for name in models:
        plt.plot(history[name]['train_loss'], '--', color=colors[name], label=f'{name} Train')
        plt.plot(history[name]['val_loss'], '-', color=colors[name], label=f'{name} Val')
    plt.title('Loss Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Accuracy�Ա�
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

    # 2. Shirt����ROC����
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

    # 3. Shirt����PR����
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
    �����ģ�������ۺϿ��ӻ�(������)
    
    ����:
        y_true: ��ʵ��ǩ
        y_pred: Ԥ���ǩ
        y_scores: Ԥ����� (shape: [n_samples, n_classes])
        classes: ��������б�
        train_loss: ѵ����ʧ��ʷ(��ѡ)
        val_loss: ��֤��ʧ��ʷ(��ѡ)
        train_acc: ѵ��׼ȷ����ʷ(��ѡ)
        val_acc: ��֤׼ȷ����ʷ(��ѡ)
        figsize: ͼ���С
        save_path: ����·��(��ѡ)
    """
    if not combined:
        # ������ʾÿ��ͼ��
        plot_confusion_matrix(y_true, y_pred, classes)
        plot_classification_report(y_true, y_pred, classes)
        plot_roc_curves(y_true, y_scores, classes)
        plot_pr_curves(y_true, y_scores, classes)
        if train_loss is not None:
            plot_training_curves(train_loss, val_loss, train_acc, val_acc)
        plt.show()  # ͳһ������ʾ
        return  # ��ǰ�˳�����ִ�к����ϲ��߼�

    fig = plt.figure(figsize=figsize)
    
    # �����Ƿ��ṩѵ����ʷ���ݾ�������
    if train_loss is not None and val_loss is not None and train_acc is not None and val_acc is not None:
        # ��ѵ����ʷ���� - 3��3�в���
        gs = fig.add_gridspec(3, 3)
        
        # 1. ѵ������
        ax1 = fig.add_subplot(gs[0, :2])  # ѵ����ʧ��׼ȷ��(��2��)
        ax2 = fig.add_subplot(gs[0, 2])   # ��֤׼ȷ��(��1��)
        plot_training_curves(train_loss, val_loss, train_acc, val_acc, ax=[ax1, ax2])
        
        # 2. ��������
        ax3 = fig.add_subplot(gs[1, 0])
        plot_confusion_matrix(y_true, y_pred, classes, ax=ax3)
        
        # 3. ���౨����ͼ
        ax4 = fig.add_subplot(gs[1, 1])
        plot_classification_report(y_true, y_pred, classes, ax=ax4)
        
        # 4. ROC����
        ax5 = fig.add_subplot(gs[1, 2])
        plot_roc_curves(y_true, y_scores, classes, ax=ax5)
        
        # 5. PR����
        ax6 = fig.add_subplot(gs[2, :])
        plot_pr_curves(y_true, y_scores, classes, ax=ax6)
        
    else:
        # û��ѵ����ʷ���� - 2��3�в���
        gs = fig.add_gridspec(2, 3)
        
        # 1. ��������
        ax1 = fig.add_subplot(gs[0, 0])
        plot_confusion_matrix(y_true, y_pred, classes, ax=ax1)
        
        # 2. ���౨����ͼ
        ax2 = fig.add_subplot(gs[0, 1])
        plot_classification_report(y_true, y_pred, classes, ax=ax2)
        
        # 3. ROC����
        ax3 = fig.add_subplot(gs[0, 2])
        plot_roc_curves(y_true, y_scores, classes, ax=ax3)
        
        # 4. PR����
        ax4 = fig.add_subplot(gs[1, :])
        plot_pr_curves(y_true, y_scores, classes, ax=ax4)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)  # Added dpi for better quality
    plt.show()