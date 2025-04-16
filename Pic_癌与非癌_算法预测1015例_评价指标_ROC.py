import os

import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc, confusion_matrix
import matplotlib.pyplot as plt


# 加载数据
def load_data(file_path):
    df = pd.read_excel(file_path)  # 假设数据存储在Excel文件中
    return df['Label'], df['ModelProb0'], df['ModelProb1'], df['pred']


# 计算ROC曲线和AUC
def compute_roc(label, prob, pos_label):
    fpr, tpr, _ = roc_curve(label, prob, pos_label=pos_label)
    roc_auc = auc(fpr, tpr)
    return fpr, tpr, roc_auc


# 绘制ROC曲线
def plot_roc(fpr, tpr, roc_auc, label, low, upper, color):
    plt.plot(fpr, tpr, lw=3, label=f'{label}_AUC = {roc_auc:.3f}({low:.3f}-{upper:.3f})', color=color)

# 计算AUC及其95%置信区间
def compute_auc_ci(label, prob, pos_label, n_bootstraps=1000, alpha=0.95):
    from sklearn.utils import resample
    # 计算原始AUC
    roc_auc = auc(*roc_curve(label, prob, pos_label=pos_label)[:2])
    bootstrapped_aucs = []
    # Bootstrap重采样
    for _ in range(n_bootstraps):
        label_resampled, prob_resampled = resample(label, prob)
        if len(np.unique(label_resampled)) < 2:  # 确保样本包含两类
            continue
        auc_resampled = auc(*roc_curve(label_resampled, prob_resampled, pos_label=pos_label)[:2])
        bootstrapped_aucs.append(auc_resampled)
    # 计算置信区间
    sorted_aucs = np.sort(bootstrapped_aucs)
    ci_lower = sorted_aucs[int((1 - alpha) / 2 * len(sorted_aucs))]
    ci_upper = sorted_aucs[int((1 + alpha) / 2 * len(sorted_aucs))]
    return roc_auc, ci_lower, ci_upper

def plot_confusion_matrix(labels, predictions, class_names, font_size=12):
    import textwrap
    import seaborn as sns
    """
    绘制混淆矩阵
    :param labels: 真实标签
    :param predictions: 预测标签
    :param class_names: 类别名称列表
    :param font_size: 字体大小
    """
    # 设置字体为Times New Roman
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    # 计算混淆矩阵
    cm = confusion_matrix(labels, predictions)
    # 使用textwrap来处理类别名称的换行
    wrapped_labels = [textwrap.fill(label, width=15) for label in class_names]
    # 绘制热力图
    plt.figure(figsize=(8, 6))
    ax = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names, cbar=True,
                     linewidths=0)

    # 添加外边界框
    for _, spine in ax.spines.items():
        spine.set_visible(True)
        spine.set_linewidth(1)
        spine.set_color('black')

    # 设置标题和标签
    plt.title('', fontsize=font_size)
    plt.xlabel('Predicted label', fontsize=20)
    plt.ylabel('True Label', fontsize=20)
    # 水平书写y轴标签
    plt.yticks(rotation=0)
    # 调整x轴和y轴刻度字体大小
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    # 调整子图参数以防止标签被截断
    plt.subplots_adjust(left=0.25, right=1, top=0.9, bottom=0.1)
    save_dir = 'H:\\trubt_paper_pics\\人工判读与算法判读对比\\1015例'
    os.makedirs(save_dir, exist_ok=True)  # 确保目录存在
    save_path = os.path.join(save_dir, f"非癌与癌_confusionMatrix.pdf")
    plt.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight')


# 计算混淆矩阵指标及其95%置信区间
def compute_metrics_ci(label, pred, pos_label=1, n_bootstraps=1000, alpha=0.95):
    from sklearn.utils import resample
    """
    计算Sensitivity, Specificity, PPV, NPV, ACC及其95% CI。
    参数：
    - label: 真实标签
    - pred: 预测标签
    - pos_label: 正类标签（0或1）
    - n_bootstraps: Bootstrap重采样次数
    - alpha: 置信水平（默认95%）
    返回：字典，包含各指标及其CI
    """
    # 如果正类为0，反转标签
    if pos_label == 0:
        label = 1 - label
        pred = 1 - pred

    # 计算原始混淆矩阵和指标
    cm = confusion_matrix(label, pred)
    tn, fp, fn, tp = cm.ravel()
    total = tp + tn + fp + fn

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    accuracy = (tp + tn) / total if total > 0 else 0

    # Bootstrap计算置信区间
    sensitivities, specificities, ppvs, npvs, accuracies = [], [], [], [], []
    for _ in range(n_bootstraps):
        label_resampled, pred_resampled = resample(label, pred)
        cm_resampled = confusion_matrix(label_resampled, pred_resampled)
        tn_r, fp_r, fn_r, tp_r = cm_resampled.ravel()
        total_r = tp_r + tn_r + fp_r + fn_r

        se = tp_r / (tp_r + fn_r) if (tp_r + fn_r) > 0 else 0
        sp = tn_r / (tn_r + fp_r) if (tn_r + fp_r) > 0 else 0
        ppv_r = tp_r / (tp_r + fp_r) if (tp_r + fp_r) > 0 else 0
        npv_r = tn_r / (tn_r + fn_r) if (tn_r + fn_r) > 0 else 0
        acc_r = (tp_r + tn_r) / total_r if total_r > 0 else 0

        sensitivities.append(se)
        specificities.append(sp)
        ppvs.append(ppv_r)
        npvs.append(npv_r)
        accuracies.append(acc_r)

    # 计算95% CI
    metrics = [sensitivities, specificities, ppvs, npvs, accuracies]
    ci_lowers = [np.percentile(m, (1 - alpha) / 2 * 100) for m in metrics]
    ci_uppers = [np.percentile(m, (1 + alpha) / 2 * 100) for m in metrics]

    return {
        'Sensitivity': (sensitivity, ci_lowers[0], ci_uppers[0]),
        'Specificity': (specificity, ci_lowers[1], ci_uppers[1]),
        'PPV': (ppv, ci_lowers[2], ci_uppers[2]),
        'NPV': (npv, ci_lowers[3], ci_uppers[3]),
        'Accuracy': (accuracy, ci_lowers[4], ci_uppers[4])
    }

# 主函数
def main(file_path):
    # 获取数据

    label, prob0, prob1, pred = load_data(file_path)

    # 计算正类为1的AUC及其95% CI
    auc1, ci_lower1, ci_upper1 = compute_auc_ci(label, prob1, pos_label=1)
    print(f"正类为1的AUC: {auc1:.2f}, 95% CI: [{ci_lower1:.2f}, {ci_upper1:.2f}]")

    # 计算正类为0的AUC及其95% CI
    auc0, ci_lower0, ci_upper0 = compute_auc_ci(label, prob0, pos_label=0)
    print(f"正类为0的AUC: {auc0:.2f}, 95% CI: [{ci_lower0:.2f}, {ci_upper0:.2f}]")

    # 总体AUC（以正类为1）
    print(f"总体AUC（正类为1）: {auc1:.3f}, 95% CI: [{ci_lower1:.3f}, {ci_upper1:.3f}]")

    # 计算正类为1的指标
    metrics_class1 = compute_metrics_ci(label, pred, pos_label=1)
    print("\n正类为1的指标:")
    for metric, (value, ci_lower, ci_upper) in metrics_class1.items():
        print(f"{metric}: {value:.3f}, 95% CI: [{ci_lower:.3f}, {ci_upper:.3f}]")

    # 计算正类为0的指标
    metrics_class0 = compute_metrics_ci(label, pred, pos_label=0)
    print("\n正类为0的指标:")
    for metric, (value, ci_lower, ci_upper) in metrics_class0.items():
        print(f"{metric}: {value:.3f}, 95% CI: [{ci_lower:.3f}, {ci_upper:.3f}]")

    plot_confusion_matrix(label, pred, ['Tumor', 'Non-tumor'], font_size=12)

    # 计算正类为1的ROC
    fpr1, tpr1, roc_auc1 = compute_roc(label, prob1, pos_label=1)

    # 计算正类为0的ROC
    fpr0, tpr0, roc_auc0 = compute_roc(label, prob0, pos_label=0)

    # 绘制ROC曲线
    plt.figure(figsize=(8, 6))
    plot_roc(fpr1, tpr1, roc_auc1, 'Tumor', ci_lower1, ci_upper1, '#F77D56')
    plot_roc(fpr0, tpr0, roc_auc0, 'Non-tumor', ci_lower0, ci_upper0, '#F4E60B')
    plt.plot([0, 1], [0, 1], 'k--')  # 绘制对角线
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
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

    save_dir = 'H:\\trubt_paper_pics\\人工判读与算法判读对比\\1015例'
    os.makedirs(save_dir, exist_ok=True)  # 确保目录存在
    save_path = os.path.join(save_dir, f"非癌与癌_ROC.pdf")
    plt.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight')


# 示例调用
if __name__ == "__main__":
    file_path = '/Users/wanghongyi/Documents/a_6________写作/turbt_论文/turbt_论文/Experimentation/data_012_0123_模型判读_01_1015例.xlsx'  # 替换为您的文件路径
    main(file_path)