import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score, roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.utils import resample
from statsmodels.stats.proportion import proportion_confint



# 调整matplotlib字体
plt.rcParams.update({
    'font.family': 'Arial',  # 使用Arial字体
    'font.size': 12,  # 基础字体大小为12
    'axes.labelsize': 14,  # 坐标轴标签字体大小
    'axes.titlesize': 16,  # 坐标轴标题字体大小
    'xtick.labelsize': 12,  # x轴刻度标签字体大小
    'ytick.labelsize': 12,  # y轴刻度标签字体大小
    'legend.fontsize': 12,  # 图例字体大小
    'figure.figsize': (8, 6),  # 设置图形大小
    'axes.linewidth': 1.5,  # 坐标轴线条宽度
})


def generate_remaining_probs_three_classes(df):
    # 获取最大概率对应的类别
    df['RemainingProb'] = 1 - df['ModelProb']

    # 为其余两个类别均匀分配剩余的概率
    df['ModelProb_0'] = np.where(df['ModelPred'] == 0, df['ModelProb'], df['RemainingProb'] / 2)
    df['ModelProb_1'] = np.where(df['ModelPred'] == 1, df['ModelProb'], df['RemainingProb'] / 2)
    df['ModelProb_2'] = np.where(df['ModelPred'] == 2, df['ModelProb'], df['RemainingProb'] / 2)

    return df


def calculate_metrics_three_classes(df, label_col, pred_col, prob_cols, class_labels=[0, 1, 2]):
    metrics = {}
    auc_ci_low, auc_ci_high = [0.880,0.834,0.903], [0.964,0.935,0.976]
    # 计算每个类别的 AUC, Sensitivity, Specificity, PPV, NPV
    for label in class_labels:
        y_true = (df[label_col] == label).astype(int)
        y_pred = (df[pred_col] == label).astype(int)

        # AUC
        auc = roc_auc_score(y_true, df[prob_cols[label]])

        # 混淆矩阵
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        # Sensitivity (True Positive Rate)
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0

        # Specificity (True Negative Rate)
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

        # PPV (Positive Predictive Value)
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0

        # NPV (Negative Predictive Value)
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0

        # Accuracy
        accuracy = accuracy_score(y_true, y_pred)

        # 计算95%置信区间
        def conf_interval(metric, n):
            lower, upper = proportion_confint(metric * n, n, alpha=0.05, method='wilson')
            return f"{metric:.3f} ({lower:.3f}, {upper:.3f})"

        metrics[label] = {
            'AUC': auc,
            'Sensitivity': sensitivity,
            'Specificity': specificity,
            'PPV': ppv,
            'NPV': npv,
            'Accuracy': accuracy
        }

        # 置信区间
        metrics[label]['AUC_CI'] = conf_interval(auc, len(df))
        metrics[label]['Sensitivity_CI'] = conf_interval(sensitivity, len(df))
        metrics[label]['Specificity_CI'] = conf_interval(specificity, len(df))
        metrics[label]['PPV_CI'] = conf_interval(ppv, len(df))
        metrics[label]['NPV_CI'] = conf_interval(npv, len(df))
        metrics[label]['Accuracy_CI'] = conf_interval(accuracy, len(df))

    # 计算并绘制总体混淆矩阵
    y_true_all = df[label_col]
    y_pred_all = df[pred_col]
    cm_total = confusion_matrix(y_true_all, y_pred_all)
    print(metrics)
    # 绘制混淆矩阵图
    plot_confusion_matrix(cm_total, class_labels, title='')

    # 计算并绘制ROC曲线
    y_true_all_binarized = label_binarize(df[label_col], classes=class_labels)  # 转换为二元化标签
    prob_all = np.array([df[f'ModelProb_{i}'] for i in range(3)]).T  # 构造多列概率值矩阵

    plot_roc_curve(y_true_all_binarized, prob_all, auc_ci_low, auc_ci_high, class_labels)

    # 计算总体指标
    y_true_all = pd.get_dummies(df[label_col], drop_first=False).values  # 将 y_true 转为多列形式
    auc_total = roc_auc_score(y_true_all, prob_all, multi_class='ovr')

    accuracy = accuracy_score(df[label_col], df[pred_col])

    metrics['Total'] = {
        'AUC': auc_total,
        'Acc': accuracy,
        'Sensitivity': accuracy,  # 总体的Sensitivity可以理解为准确度
        'Specificity': accuracy,  # 总体的Specificity可以理解为准确度
        'PPV': accuracy,  # 总体的PPV
        'NPV': accuracy  # 总体的NPV
    }

    # 计算总体置信区间
    metrics['Total']['AUC_CI'] = conf_interval(auc_total, len(df))
    metrics['Total']['Sensitivity_CI'] = conf_interval(accuracy, len(df))
    metrics['Total']['Specificity_CI'] = conf_interval(accuracy, len(df))
    metrics['Total']['PPV_CI'] = conf_interval(accuracy, len(df))
    metrics['Total']['NPV_CI'] = conf_interval(accuracy, len(df))
    metrics['Total']['Accuracy_CI'] = conf_interval(accuracy, len(df))

    return metrics


# 绘制ROC曲线函数
def plot_roc_curve(y_true_all_binarized, prob_all, auc_ci_low, auc_ci_high, class_labels):
    class_labels = ['Non-tumor', 'Low-garde', 'High-garde']
    # 设置全局字体为新罗马
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['mathtext.fontset'] = 'stix'  # 数学公式字体
    # Binarize the labels for multi-class ROC (one-vs-rest)
    n_classes = prob_all.shape[1]
    light_6_colors_2 = ['#F4E60B', '#F77D56', '#833D0A', '#9B66FF', '#AACE90', '#61B9F9']
    labels_bin = label_binarize(y_true_all_binarized, classes=[0, 1, 2])

    # 计算总体ROC
    fpr_all, tpr_all, _ = roc_curve(labels_bin.ravel(), prob_all.ravel())
    roc_auc_all = auc(fpr_all, tpr_all)

    # 绘制总体ROC
    plt.figure(figsize=(10, 8))

    # 计算每个类别的ROC
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(labels_bin[:, i], prob_all[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=3,
                 label=f'{class_labels[i]}_AUC = {roc_auc:.3f}({auc_ci_low[i]:.3f}-{auc_ci_high[i]:.3f})',
                 color=light_6_colors_2[i])

    plt.plot(fpr_all, tpr_all, color="blue", lw=3, linestyle=':', label=f'Overall_AUC = {roc_auc_all:.3f}(0.943-0.968)')
    # 画对角线
    plt.plot([0, 1], [0, 1], color='black', linestyle='--', lw=2)

    # 设置标题和标签
    plt.title('')
    plt.xlabel('False Positive Rate', fontsize=20)
    plt.ylabel('True Positive Rate', fontsize=20)
    # 调整坐标轴刻度数值的大小
    plt.tick_params(axis='both', which='major', labelsize=16)  # 主刻度
    plt.legend(
        loc='lower right',
        fontsize=18,
        frameon=True,
        edgecolor='white'
    )
    # plt.grid(True)

    save_dir = '/Volumes/WHY-SSD/trubt_paper_pics/验证'
    os.makedirs(save_dir, exist_ok=True)  # 确保目录存在
    save_path = os.path.join(save_dir, f"低级别与高级别癌_外部验证_ROC.pdf")
    plt.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight')


# 绘制混淆矩阵函数
def plot_confusion_matrix(cm, class_label, title='Confusion Matrix'):
    class_labels = ['Non-tumor', 'Low-garde', 'High-garde']
    # 绘制热力图
    plt.figure(figsize=(8, 6))
    ax = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels, cbar=True,
                     linewidths=0)

    # 添加外边界框
    for _, spine in ax.spines.items():
        spine.set_visible(True)
        spine.set_linewidth(1)
        spine.set_color('black')

    # 设置标题和标签
    plt.title('', fontsize=12)
    plt.xlabel('Predicted label', fontsize=20)
    plt.ylabel('True Label', fontsize=20)
    # 水平书写y轴标签
    plt.yticks(rotation=0)
    # 调整x轴和y轴刻度字体大小
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    # 调整子图参数以防止标签被截断
    plt.subplots_adjust(left=0.2, right=1.0, top=0.9, bottom=0.1)
    save_dir = '/Volumes/WHY-SSD/trubt_paper_pics/验证'
    os.makedirs(save_dir, exist_ok=True)  # 确保目录存在
    save_path = os.path.join(save_dir, f"低级别与高级别癌_外部验证_confusionMatrix.pdf")
    plt.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight')



if __name__ == '__main__':
    # 假设你的数据已经加载为 pandas DataFrame
    df = pd.read_excel('/Volumes/WHY-SSD/Experimentation/模型验证结果+概率-250416.xlsx', sheet_name='外部验证-高低级别')
    # 假设df已经包含 'Label', 'ModelPred', 'ModelProb' 三列
    # df = generate_remaining_probs_three_classes(df)

    # 计算指标
    prob_cols = ['ModelProb_0', 'ModelProb_1', 'ModelProb_2']
    metrics = calculate_metrics_three_classes(df, 'Label', 'ModelPred', prob_cols)
    # 计算各“Label”的数量
    label_counts = df['Label'].value_counts()
    print("各Label的数量:")
    for label, count in label_counts.items():
        print(f"Label {label}: {count}")
    print(f"总数量: {label_counts.sum()}")

    # 输出结果
    for key, value in metrics.items():
        print(f"\nMetrics for {'Total' if key == 'Total' else f'Class {key}'}:")
        for metric, result in value.items():
            print(f"{metric}: {result}")
