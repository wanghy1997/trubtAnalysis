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
    save_dir = '/Volumes/WHY-SSD/trubt_paper_pics/demo'
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


# 假设原始数据在名为 "原始数据" 的列中
def transform(value):
    if value in [0, 1]:
        return 0
    elif value == 2:
        return 1
    elif value == 3:
        return 2
    else:
        return None  # 或者 return value，看你是否希望保留未知数值


def transform_(value):
    if value in [0]:
        return 0
    else:
        return 1


def process_dataframe(df):
    """
    根据指定规则处理 DataFrame 并新增列：
    - Label0_0：Label0 为 0 或 1 时为 0，否则为 None。
    - Pred0_0：
        - 如果 Pred 为 0 或 2，则为 0，ModelProb_0 赋值给 ModelProb_0_0。
        - 否则为 1，ModelProb_1 赋值给 ModelProb_0_1。
    """

    # Label0_0 列处理
    df["Label_tumorOrNo0"] = df["Label0"].apply(lambda x: 0 if x in [0, 1] else 1)

    # Pred0_0 及 ModelProb 列处理
    def process_pred(row):
        if row["Pred0"] in [0, 1]:
            return pd.Series([0, row["ModelProb_0"], 1.0-row["ModelProb_0"]])
        else:
            return pd.Series([1, 1.0-row["ModelProb_1"], row["ModelProb_1"]])

    df[["Pred_tumorOrNo0", "Pred_tumorOrNo0_0", "Pred_tumorOrNo0_1"]] = df.apply(process_pred, axis=1)

    return df


def process_dataframe1(df):
    """
    根据指定规则处理 DataFrame 并新增列：
    - Label0_0：Label0 为 0 或 1 时为 0，否则为 None。
    - Pred0_0：
        - 如果 Pred 为 0 或 2，则为 0，ModelProb_0 赋值给 ModelProb_0_0。
        - 否则为 1，ModelProb_1 赋值给 ModelProb_0_1。
    """

    # Label0_0 列处理
    df["Label_tumorOrNo1"] = df["Label"].apply(lambda x: 0 if x in [0] else 1)

    # Pred0_0 及 ModelProb 列处理
    def process_pred(row):
        if row["Pred1"] in [0]:
            return pd.Series([0, row["Prob_Cls00"], 1.0-row["Prob_Cls00"]])
        else:
            return pd.Series([1, 1.0-(row["Prob_Cls10"]+row["Prob_Cls20"]), row["Prob_Cls10"]+row["Prob_Cls20"]])

    df[["Pred_tumorOrNo1", "Pred_tumorOrNo1_0", "Pred_tumorOrNo1_1"]] = df.apply(process_pred, axis=1)

    return df


def process_dataframe3(df):
    """
    根据指定规则处理 DataFrame 并新增列：
    - Label0_0：Label0 为 0 或 1 时为 0，否则为 None。
    - Pred0_0：
        - 如果 Pred 为 0 或 2，则为 0，ModelProb_0 赋值给 ModelProb_0_0。
        - 否则为 1，ModelProb_1 赋值给 ModelProb_0_1。
    """

    # Label0_0 列处理
    df["Label_tumorOrNo1"] = df["Label"].apply(lambda x: 0 if x in [0] else 1)

    # Pred0_0 及 ModelProb 列处理
    def process_pred(row):
        if row["Pred1"] in [0]:
            return pd.Series([0, row["Prob_Cls00"], 1.0-row["Prob_Cls00"]])
        else:
            return pd.Series([1, 1.0-(row["Prob_Cls10"]+row["Prob_Cls20"]), row["Prob_Cls10"]+row["Prob_Cls20"]])

    df[["Pred_tumorOrNo1", "Pred_tumorOrNo1_0", "Pred_tumorOrNo1_1"]] = df.apply(process_pred, axis=1)

    return df


def process_pred_tumor_flags(df):
    """
    处理预测标记列，根据 Pred_tumorOrNo0 和 Pred_tumorOrNo1 的值生成 pred、prob0 和 prob1。
    """

    def compute_pred_and_probs(row):
        if row["Pred_tumorOrNo0"] == 0 and row["Pred_tumorOrNo1"] == 0:
            pred = 0
            prob0 = row["Pred_tumorOrNo0_0"]
            prob1 = row["Pred_tumorOrNo0_1"]
        else:
            pred = 1
            prob0 = row["Pred_tumorOrNo1_0"]
            prob1 = row["Pred_tumorOrNo1_1"]
        return pd.Series([pred, prob0, prob1], index=["pred", "prob0", "prob1"])

    df[["pred", "prob0", "prob1"]] = df.apply(compute_pred_and_probs, axis=1)
    return df


# 主函数
def main(file_path):
    # 获取数据

    df = pd.read_excel(file_path)  # 假设数据存储在Excel文件中


    # # 应用转换逻辑并新增一列
    # df["Pred0_0"] = df["Pred0_0"].apply(transform_)
    # df["Label0_0"] = df["Label0_0"].apply(transform_)


    # 调用处理方法
    process_pred_tumor_flags(df)

    # 保存修改后的表格
    df.to_excel('/Users/wanghongyi/Documents/a_6________写作/turbt_论文/Experimentation/内部验证-癌与非癌-整合后结2.xlsx', index=False)



# 示例调用
if __name__ == "__main__":
    file_path = '/Users/wanghongyi/Documents/a_6________写作/turbt_论文/Experimentation/内部验证-癌与非癌-整合后结果.xlsx'  # 替换为您的文件路径
    main(file_path)