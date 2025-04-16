import os

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.preprocessing import label_binarize


def load_data(file_path):
    """加载xlsx文件的数据，sheet='sheet1'"""
    data = pd.read_excel(file_path, sheet_name='1015-浸润非浸润')
    labels = data['Label'].values
    predictions = data['Pred'].values
    probabilities = data[['Prob_Cls0', 'Prob_Cls1', 'Prob_Cls2', 'Prob_Cls3']].values
    return labels, predictions, probabilities


def merge_labels_and_predictions(labels, predictions, probabilities):
    """将0和1合并为新的0，将2变为1，将3变为2，并更新概率"""
    # 合并标签
    labels = np.where(np.isin(labels, [0, 1]), 0, labels)  # 将0和1合并为0
    labels = np.where(labels == 2, 1, labels)  # 将2变为1
    labels = np.where(labels == 3, 2, labels)  # 将3变为2

    # 合并预测
    predictions = np.where(np.isin(predictions, [0, 1]), 0, predictions)  # 将0和1合并为0
    predictions = np.where(predictions == 2, 1, predictions)  # 将2变为1
    predictions = np.where(predictions == 3, 2, predictions)  # 将3变为2

    # 合并概率：Prob_Cls0和Prob_Cls1合并为Prob_Cls0（新类别0）
    probabilities[:, 0] = probabilities[:, 0] + probabilities[:, 1]

    # 合并概率：Prob_Cls2变为Prob_Cls1，Prob_Cls3变为Prob_Cls2
    probabilities[:, 1] = probabilities[:, 2]
    probabilities[:, 2] = probabilities[:, 3]

    # 删除原来的Prob_Cls3列，因为它们已经合并到新的类别中
    probabilities = np.delete(probabilities, [3], axis=1)  # 删除第1列和第3列（Prob_Cls1 和 Prob_Cls3）

    return labels, predictions, probabilities


def calculate_metrics(labels, predictions, probabilities, n_bootstrap=1000):
    """计算AUC、Sensitivity、Specificity、PPV、NPV，并输出每个类别的95%置信区间"""

    def calculate_confusion_matrix_metrics(cm):
        """从混淆矩阵计算敏感度、特异性、阳性预测值、阴性预测值"""
        TN, FP, FN, TP = cm.ravel()

        # Sensitivity (True Positive Rate)
        sensitivity = TP / (TP + FN)

        # Specificity (True Negative Rate)
        specificity = TN / (TN + FP)

        # PPV (Positive Predictive Value)
        ppv = TP / (TP + FP)

        # NPV (Negative Predictive Value)
        npv = TN / (TN + FN)

        # Accuracy
        acc = (TP + TN) / (TP + TN + FP + FN)

        return sensitivity, specificity, ppv, npv, acc

    metrics = {'AUC': [], 'Sensitivity': [], 'Specificity': [], 'PPV': [], 'NPV': [], 'ACC': []}
    ci_metrics = {'AUC_CI': [], 'Sensitivity_CI': [], 'Specificity_CI': [], 'PPV_CI': [], 'NPV_CI': [], 'ACC_CI': []}

    # Binarize the labels for multi-class ROC (one-vs-rest)
    labels_bin = label_binarize(labels, classes=[0, 1, 2])

    for i in range(3):
        # 计算每个类别的AUC
        fpr, tpr, thresholds = roc_curve(labels_bin[:, i], probabilities[:, i])
        roc_auc = auc(fpr, tpr)
        metrics['AUC'].append(roc_auc)

        # 使用混淆矩阵计算 Sensitivity, Specificity, PPV, NPV
        cm = confusion_matrix(labels_bin[:, i], predictions == i)
        sensitivity, specificity, ppv, npv, acc = calculate_confusion_matrix_metrics(cm)

        metrics['Sensitivity'].append(sensitivity)
        metrics['Specificity'].append(specificity)
        metrics['PPV'].append(ppv)
        metrics['NPV'].append(npv)
        metrics['ACC'].append(acc)

        # 计算AUC的95% CI（基于近似公式）
        ci_auc = 1.96 * np.sqrt((roc_auc * (1 - roc_auc)) / len(labels))
        ci_metrics['AUC_CI'].append((roc_auc - ci_auc, roc_auc + ci_auc))

        # 计算Sensitivity, Specificity, PPV, NPV的95% CI（基于bootstrap方法）
        sensitivities, specificities, ppvs, npvs, accs = [], [], [], [], []
        for _ in range(n_bootstrap):
            # 使用bootstrap方法进行抽样
            indices = np.random.choice(len(labels), len(labels), replace=True)
            cm_bootstrap = confusion_matrix(labels_bin[:, i][indices], predictions[indices] == i)
            sensitivity_b, specificity_b, ppv_b, npv_b, acc_b = calculate_confusion_matrix_metrics(cm_bootstrap)

            sensitivities.append(sensitivity_b)
            specificities.append(specificity_b)
            ppvs.append(ppv_b)
            npvs.append(npv_b)
            accs.append(acc_b)

        # 计算bootstrap的置信区间
        ci_sensitivity = np.percentile(sensitivities, [2.5, 97.5])
        ci_specificity = np.percentile(specificities, [2.5, 97.5])
        ci_ppv = np.percentile(ppvs, [2.5, 97.5])
        ci_npvs = np.percentile(npvs, [2.5, 97.5])
        ci_accs = np.percentile(accs, [2.5, 97.5])

        ci_metrics['Sensitivity_CI'].append(ci_sensitivity)
        ci_metrics['Specificity_CI'].append(ci_specificity)
        ci_metrics['PPV_CI'].append(ci_ppv)
        ci_metrics['NPV_CI'].append(ci_npvs)
        ci_metrics['ACC_CI'].append(ci_accs)

    # 计算总体指标和置信区间
    overall_auc = np.mean(metrics['AUC'])
    overall_sensitivity = np.mean(metrics['Sensitivity'])
    overall_specificity = np.mean(metrics['Specificity'])
    overall_ppv = np.mean(metrics['PPV'])
    overall_npvs = np.mean(metrics['NPV'])
    overall_accs = np.mean(metrics['ACC'])

    # 计算总体AUC的置信区间（基于AUC的置信区间的平均值）
    ci_overall_auc = 1.96 * np.sqrt((overall_auc * (1 - overall_auc)) / len(labels))

    # 计算总体的敏感度、特异性、PPV、NPV的置信区间（基于bootstrap方法）
    ci_overall_sensitivity = np.percentile(overall_sensitivity, [2.5, 97.5])
    ci_overall_specificity = np.percentile(overall_specificity, [2.5, 97.5])
    ci_overall_ppv = np.percentile(overall_ppv, [2.5, 97.5])
    ci_overall_npvs = np.percentile(overall_npvs, [2.5, 97.5])
    ci_overall_accs = np.percentile(overall_accs, [2.5, 97.5])

    # 打印总体指标和置信区间
    print(
        f"Overall AUC: {overall_auc:.4f}, 95% CI: ({overall_auc - ci_overall_auc:.4f}, {overall_auc + ci_overall_auc:.4f})")
    print(f"Overall Sensitivity: {overall_sensitivity:.4f}, 95% CI: {ci_overall_sensitivity}")
    print(f"Overall Specificity: {overall_specificity:.4f}, 95% CI: {ci_overall_specificity}")
    print(f"Overall PPV: {overall_ppv:.4f}, 95% CI: {ci_overall_ppv}")
    print(f"Overall NPV: {overall_npvs:.4f}, 95% CI: {ci_overall_npvs}")
    print(f"Overall ACC: {overall_accs:.4f}, 95% CI: {ci_overall_accs}")

    return metrics, ci_metrics


def plot_roc_curve(labels, probabilities, class_names, auc_ci_low, auc_ci_high, font_size=12):
    """绘制每个类别的ROC曲线及总体ROC"""
    # 设置全局字体为新罗马
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['mathtext.fontset'] = 'stix'  # 数学公式字体
    # Binarize the labels for multi-class ROC (one-vs-rest)
    n_classes = probabilities.shape[1]
    light_6_colors_2 = ['#F4E60B', '#F77D56', '#833D0A', '#9B66FF', '#AACE90', '#61B9F9']
    labels_bin = label_binarize(labels, classes=[0, 1, 2])

    # 计算总体ROC
    fpr_all, tpr_all, _ = roc_curve(labels_bin.ravel(), probabilities.ravel())
    roc_auc_all = auc(fpr_all, tpr_all)

    # 绘制总体ROC
    plt.figure(figsize=(10, 8))

    # 计算每个类别的ROC
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(labels_bin[:, i], probabilities[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=3, label=f'{class_names[i]}_AUC = {roc_auc:.3f}({auc_ci_low[i]:.3f}-{auc_ci_high[i]:.3f})', color=light_6_colors_2[i])


    plt.plot(fpr_all, tpr_all, color="blue", lw=3, linestyle=':', label=f'Overall_AUC = {roc_auc_all:.3f}(0.920-0.950)')
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

    save_dir = 'H:\\trubt_paper_pics\\人工判读与算法判读对比\\1015例'
    os.makedirs(save_dir, exist_ok=True)  # 确保目录存在
    save_path = os.path.join(save_dir, f"非浸润与浸润癌_ROC.pdf")
    plt.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight')

def plot_confusion_matrix(labels, predictions, class_names, font_size=12):
    import textwrap
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
    save_path = os.path.join(save_dir, f"非浸润与浸润癌_confusionMatrix.pdf")
    plt.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight')

# def plot_confusion_matrix(labels, predictions, class_names, font_size=12):
#     """
#     绘制混淆矩阵
#     :param labels: 真实标签
#     :param predictions: 预测标签
#     :param class_names: 类别名称列表
#     :param font_size: 字体大小
#     """
#     # 计算混淆矩阵
#     cm = confusion_matrix(labels, predictions)
#
#     # 绘制热力图
#     plt.figure(figsize=(8, 6))
#     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names, cbar=False)
#
#     # 设置标题和标签
#     plt.title('', fontsize=font_size)
#     plt.xlabel('Predicted Label', fontsize=font_size)
#     plt.ylabel('True Label', fontsize=font_size)
#
#     # 显示混淆矩阵图
#     plt.show()

def main(file_path):
    # 加载数据
    labels, predictions, probabilities = load_data(file_path)
    labels, predictions, probabilities = merge_labels_and_predictions(labels, predictions, probabilities)
    # 绘制混淆矩阵
    class_names = ['Noncancerous', 'Non-invasive', 'Invasive']
    plot_confusion_matrix(labels, predictions, class_names, font_size=12)
    # 计算各类指标
    metrics, ci_metrics = calculate_metrics(labels, predictions, probabilities)
    auc_ci_low = []
    auc_ci_high = []
    # 打印每个类别的AUC、Sensitivity、Specificity、PPV、NPV及其置信区间
    for i in range(3):
        print(f"Class {i}:")
        print(f"AUC: {metrics['AUC'][i]:.4f}, 95% CI: {ci_metrics['AUC_CI'][i]}")
        print(f"Sensitivity: {metrics['Sensitivity'][i]:.4f} {ci_metrics['Sensitivity_CI'][i]}")
        print(f"Specificity: {metrics['Specificity'][i]:.4f} {ci_metrics['Specificity_CI'][i]}")
        print(f"PPV: {metrics['PPV'][i]:.4f} {ci_metrics['PPV_CI'][i]}")
        print(f"NPV: {metrics['NPV'][i]:.4f} {ci_metrics['NPV_CI'][i]}")
        print(f"ACC: {metrics['ACC'][i]:.4f} {ci_metrics['ACC_CI'][i]}")
        auc_ci_low.append(ci_metrics['AUC_CI'][i][0])
        auc_ci_high.append(ci_metrics['AUC_CI'][i][1])
    # 绘制ROC曲线
    plot_roc_curve(labels, probabilities, class_names, auc_ci_low, auc_ci_high, font_size=12)




if __name__ == '__main__':

    # 调用主程序
    file_path = 'F:\\文档\\a_6________写作\\turbt_论文\\Experimentation\\1015例模型预测结果+概率-250302.xlsx'
    main(file_path)


