import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn import metrics
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             cohen_kappa_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score,
                             roc_curve, precision_recall_curve, auc)
import matplotlib

matplotlib.use('TkAgg')  # 用于无GUI环境
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path
from model import *
from utils import *

# 设置随机种子
SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# 可视化配置
VIS_DIR = 'ResGINSynergy'
Path(VIS_DIR).mkdir(parents=True, exist_ok=True)


def plot_training_curve(train_losses, val_aucs, fold):
    """绘制训练曲线"""
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.title(f'Fold {fold} - Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.subplot(1, 2, 2)
    plt.plot(val_aucs, label='Validation AUC', color='darkorange')
    plt.title(f'Fold {fold} - Validation AUC')
    plt.xlabel('Epoch')
    plt.ylabel('AUC')

    plt.tight_layout()
    plt.savefig(os.path.join(VIS_DIR, f'fold{fold}_training.png'))
    plt.close()


def plot_confusion_matrix(y_true, y_pred, fold):
    """绘制混淆矩阵"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'])
    plt.title(f'Fold {fold} - Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(os.path.join(VIS_DIR, f'fold{fold}_cm.png'))
    plt.close()


def plot_roc_pr_curves(y_true, y_score, fold):
    """绘制ROC和PR曲线"""
    # ROC曲线
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    # PR曲线
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    pr_auc = auc(recall, precision)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Fold {fold} - ROC Curve')
    plt.legend(loc="lower right")

    plt.subplot(1, 2, 2)
    plt.plot(recall, precision, color='blue', lw=2,
             label=f'PR (AUC = {pr_auc:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Fold {fold} - Precision-Recall Curve')
    plt.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(os.path.join(VIS_DIR, f'fold{fold}_curves.png'))
    plt.close()


def plot_final_metrics(all_metrics):
    """最终指标对比图"""
    import pandas as pd
    df = pd.DataFrame(all_metrics)

    plt.figure(figsize=(15, 7))
    metrics_order = ['AUC', 'ACC', 'F1', 'Precision', 'Recall', 'BACC']
    labels = ['AUC', 'Accuracy', 'F1-Score', 'Precision', 'Recall', 'Balanced Acc']

    # 计算均值和标准差
    means = df[metrics_order].mean()
    stds = df[metrics_order].std()

    # 绘制柱状图
    x = np.arange(len(means))
    plt.bar(x, means, yerr=stds, align='center', alpha=0.7,
            capsize=10, color=['#1f77b4', '#ff7f0e', '#2ca02c',
                               '#d62728', '#9467bd', '#8c564b'])

    plt.xticks(x, labels)
    plt.ylabel('Score')
    plt.ylim(0.5, 1.0)
    plt.title('Cross-Validation Performance (Mean ± SD)')

    # 添加数值标签
    for i, (mean, std) in enumerate(zip(means, stds)):
        plt.text(i, mean + 0.02, f'{mean:.3f}±{std:.3f}',
                 ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(os.path.join(VIS_DIR, 'final_metrics.png'))
    plt.close()


def train(model, device, loader_train, optimizer, epoch):
    model.train()
    total_loss = 0
    for batch_idx, (data1, data2, y) in enumerate(loader_train):
        data1, data2, y = data1.to(device), data2.to(device), y.to(device)
        optimizer.zero_grad()
        output = model(data1, data2)
        loss = F.cross_entropy(output, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        if batch_idx % LOG_INTERVAL == 0:
            print('Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch,
                                                                           batch_idx * len(data1.x),
                                                                           len(loader_train.dataset),
                                                                           100. * batch_idx / len(loader_train),
                                                                           loss.item()))

    return total_loss / len(loader_train)


def predicting(model, device, loader):
    """预测函数"""
    model.eval()
    probs, labels, preds = [], [], []
    with torch.no_grad():
        for data1, data2, y in loader:
            data1, data2 = data1.to(device), data2.to(device)
            output = model(data1, data2)
            prob = F.softmax(output, dim=1)[:, 1].cpu().numpy()
            pred = output.argmax(dim=1).cpu().numpy()

            probs.extend(prob)
            preds.extend(pred)
            labels.extend(y.numpy())
    return np.array(labels), np.array(probs), np.array(preds)


# 主程序
if __name__ == "__main__":
    # 超参数设置
    TRAIN_BATCH_SIZE = 128
    TEST_BATCH_SIZE = 128
    LR = 0.0005
    LOG_INTERVAL = 20
    NUM_EPOCHS = 1
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Learning rate: {LR}")
    print(f"Epochs: {NUM_EPOCHS}")
    # 数据集准备
    dataset = MyDataset()
    all_metrics = []
    # 5折交叉验证
    n_folds =5
    indices = np.arange(len(dataset))
    np.random.shuffle(indices)
    fold_splits = np.array_split(indices, n_folds)

    for fold in range(n_folds):
        print(f"\n=== Fold {fold + 1}/{n_folds} ===")

        # 划分训练测试集
        test_indices = fold_splits[fold]
        train_indices = np.concatenate([fold_splits[i] for i in range(n_folds) if i != fold])

        train_dataset = dataset.get_data(train_indices)
        test_dataset = dataset.get_data(test_indices)

        train_loader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE,
                                  shuffle=True, collate_fn=collate)
        test_loader = DataLoader(test_dataset, batch_size=TEST_BATCH_SIZE,
                                 shuffle=False, collate_fn=collate)

        # 模型初始化
        model = AttenSyn().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)

        # 训练记录
        best_auc = 0
        train_losses = []
        val_aucs = []
        best_preds = None

        # 训练循环
        for epoch in range(1, NUM_EPOCHS + 1):
            # 训练阶段
            avg_loss = train(model, device, train_loader, optimizer, epoch)
            train_losses.append(avg_loss)
            # 验证阶段
            y_true, y_prob, y_pred = predicting(model, device, test_loader)
            current_auc = roc_auc_score(y_true, y_prob)
            val_aucs.append(current_auc)
            # 保存最佳模型
            if current_auc > best_auc:
                best_auc = current_auc
                best_preds = (y_true, y_prob, y_pred)

                # 保存指标
                metrics_dict = {
                    'AUC': current_auc,
                    'ACC': accuracy_score(y_true, y_pred),
                    'F1': f1_score(y_true, y_pred),
                    'Precision': precision_score(y_true, y_pred),
                    'Recall': recall_score(y_true, y_pred),
                    'BACC': balanced_accuracy_score(y_true, y_pred)
                }
        # 绘制本折结果
        y_true, y_prob, y_pred = best_preds
        plot_training_curve(train_losses, val_aucs, fold + 1)
        plot_confusion_matrix(y_true, y_pred, fold + 1)
        plot_roc_pr_curves(y_true, y_prob, fold + 1)
        # 记录指标
        all_metrics.append(metrics_dict)
        print(f"Fold {fold + 1} Best AUC: {best_auc:.4f}")

    # 绘制最终对比图
    plot_final_metrics(all_metrics)
    print("\nTraining completed. Visualizations saved to:", VIS_DIR)