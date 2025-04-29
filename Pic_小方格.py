import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D
from sklearn.metrics import accuracy_score, f1_score, recall_score, confusion_matrix
from scipy.stats import bootstrap


def evaluate_doctor_predictions_to_excel(mapped_labels, mapped_doctor_results, excel_path=''):
    classes = [0, 1, 2]
    n_bootstraps = 1000
    alpha = 0.05

    # Function to compute metrics for a specific class
    def compute_class_metrics(y_true, y_pred, cls):
        cm = confusion_matrix(y_true, y_pred, labels=classes)
        TP = cm[cls, cls]
        FP = cm[:, cls].sum() - TP
        FN = cm[cls, :].sum() - TP
        TN = cm.sum() - TP - FP - FN

        recall = TP / (TP + FN) if TP + FN > 0 else 0
        specificity = TN / (TN + FP) if TN + FP > 0 else 0
        ppv = TP / (TP + FP) if TP + FP > 0 else 0
        npv = TN / (TN + FN) if TN + FN > 0 else 0
        f1 = 2 * ppv * recall / (ppv + recall) if ppv + recall > 0 else 0
        acc = np.mean(y_true == y_pred)

        return {
            'accuracy': acc,
            'recall': recall,
            'specificity': specificity,
            'ppv': ppv,
            'npv': npv,
            'f1': f1
        }

    # Function to calculate bootstrap CI for a given class
    def bootstrap_class_ci(y_true, y_pred, cls):
        from sklearn.utils import resample
        stats = {'accuracy': [], 'recall': [], 'specificity': [], 'ppv': [], 'npv': [], 'f1': []}
        n = len(y_true)

        for _ in range(n_bootstraps):
            idx = resample(np.arange(n), replace=True)
            metrics = compute_class_metrics(y_true[idx], y_pred[idx], cls)
            for key, value in metrics.items():
                stats[key].append(value)

        ci = {}
        for key in stats:
            values = np.array(stats[key])
            ci[key] = (np.percentile(values, 100 * alpha / 2),
                       np.percentile(values, 100 * (1 - alpha / 2)))
        return ci

    # Structure to hold results for all classes
    results_by_class = {cls: {} for cls in classes}

    # Compute per-doctor metrics for each class
    for doctor, y_pred in mapped_doctor_results.items():
        for cls in classes:
            metrics = compute_class_metrics(mapped_labels, y_pred, cls)
            ci = bootstrap_class_ci(mapped_labels, y_pred, cls)

            entry = {}

            for metric in metrics:
                entry[f"{metric}_value"] = metrics[metric]
            for metric in ci:
                entry[f"{metric}_ci_low"] = ci[metric][0]
                entry[f"{metric}_ci_high"] = ci[metric][1]

            results_by_class[cls][doctor] = entry

    # Export to Excel
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        for cls in classes:
            df = pd.DataFrame.from_dict(results_by_class[cls], orient='index')
            df.index.name = 'Doctor'
            df.reset_index(inplace=True)
            df.to_excel(writer, sheet_name=f'Class {cls}', index=False)

    print(f"✅ Metrics written to: {excel_path}")


def map_case_0123(label):
    """情况1：将0合并为新0，1/2合并为新1，3/4合并为新2，5/6合并为新3"""
    if label in [0, 1, 2]:
        return 0
    elif label in [3, 4]:
        return 1
    elif label in [5, 6]:
        return 2
    else:
        return label  # 保证对于未知标签的情况，返回原标签

def map_case_01(label):
    """情况2：将0/1/2合并为新0，3/4/5/6合并为新1"""
    if label in [0, 1, 2]:
        return 0
    elif label in [3, 4, 5, 6]:
        return 1
    else:
        return label  # 保证对于未知标签的情况，返回原标签

def map_case_012(label):
    """情况3：将0/1/2合并为新0，3/5合并为新1，4/6合并为新2"""
    if label in [0, 1, 2]:
        return 0
    elif label in [3, 5]:
        return 1
    elif label in [4, 6]:
        return 2
    else:
        return label  # 保证对于未知标签的情况，返回原标签

def apply_mapping_to_labels(labels, mapping_function):
    """将标签映射应用于标签数组"""
    return np.array([mapping_function(label) for label in labels])

def apply_mapping_to_doctors_predictions(doctor_results, mapping_function):
    """将标签映射应用于医生的所有预测"""
    for doctor, predictions in doctor_results.items():
        doctor_results[doctor] = apply_mapping_to_labels(predictions, mapping_function)
    return doctor_results

def plot_doctor_predictions(map_case, remark):
    # 定义颜色映射
    color_map = {
        0: '#D3D3D3',  # 最浅
        1: '#A9A9A9',
        2: '#696969',
        3: '#000000'  # 最深
    }
    # 读取 Excel 数据
    file_path = 'F:\\文档\\a_6________写作\\turbt_论文\\Experimentation\\data_0123_人工与辅助判读_0227用.xlsx'
    save_dir = 'F:\\文档\\a_6________写作\\turbt_论文\\Experimentation\\人工判读与算法判读对比'

    # 读取Excel文件
    data = pd.read_excel(file_path, sheet_name='AI_AD01')

    # 自动加载医生标识符，从第7列到第27列
    doctors = data.columns[6:24].tolist()

    # 通过 'Label' 获取真实标签
    labels = data['Label'].values
    # 根据医生结果列来处理医生判读数据
    doctor_results = {doctor: data[doctor].values for doctor in doctors}

    # 映射标签
    mapped_labels = apply_mapping_to_labels(labels, map_case)  # 使用情况1的映射
    mapped_doctor_results = apply_mapping_to_doctors_predictions(doctor_results, map_case)  # 使用情况1的映射
    print('mapped_labels', mapped_labels)
    print('mapped_doctor_results', mapped_doctor_results)

    # 创建图形
    fig, ax = plt.subplots(figsize=(15, 10))

    # 设置可调节的上下左右距离
    vertical_gap = 2  # 上下格子之间的距离（可调整）
    horizontal_gap = 2  # 左右格子之间的距离（可调整）
    gap_size = 5  # 格子之间的距离（可调整）
    # 确保格子是正方形
    cell_size = 20  # 格子的大小，正方形的边长

    # 设置绘图的纵横比为相等，确保格子是正方形
    ax.set_aspect('equal')

    # 绘制格子
    for i, doctor in enumerate(doctors):

        for j in range(len(labels)):  # 遍历每一条记录

            doctor_prediction = mapped_doctor_results[doctor][j]
            true_label = mapped_labels[j]

            # 计算每个小格子的位置
            x_pos = j * (cell_size + gap_size)
            y_pos = (len(doctors) - i - 1) * (cell_size + gap_size)  # 计算格子在Y轴上的位置

            # 判断是否正确
            if doctor_prediction == true_label:
                # 正确判读，使用带框的白色格子
                ax.add_patch(patches.Rectangle((x_pos, y_pos), cell_size, cell_size, linewidth=2, edgecolor='white', facecolor='#8A2BE2'))
            else:
                # 错误判读，使用红色格子
                ax.add_patch(patches.Rectangle((x_pos, y_pos), cell_size, cell_size, linewidth=0, edgecolor='white', facecolor='#F4B183'))

        # 在每一行的最左侧添加医生标识符
        ax.text(-horizontal_gap, (len(doctors) - i - 1) * (cell_size + gap_size) + cell_size / 2,
                doctor, va='center', ha='right', fontsize=12, color='black')

    # 设置坐标轴
    ax.set_xlim(0, len(labels) * (cell_size + gap_size))
    ax.set_ylim(0, len(doctors) * (cell_size + gap_size))

    # 隐藏坐标轴
    ax.axis('off')
    ax.set_title(f'{remark}', fontsize=16)
    # 设置x轴和y轴的显示
    ax.set_xticks(np.arange(0, len(labels) * (cell_size + gap_size), cell_size + gap_size))
    ax.set_yticks(np.arange(0, len(doctors) * (cell_size + gap_size), cell_size + gap_size))
    ax.set_xticklabels(np.arange(1, len(labels) + 1))
    ax.set_yticklabels([])  # 不显示y轴标签，因为已通过文本添加医生标识符

    # 保存输出结果
    plt.savefig(f'{save_dir}/doctor_predictions_comparison_with_gaps.png')
    plt.show()


def metric_ci(mapped_labels, mapped_doctor_results, excel_path='/Volumes/WHY-SSD/trubt_paper_pics/癌与非癌.xlsx'):
    import numpy as np
    import pandas as pd
    from statsmodels.stats.proportion import proportion_confint

    def calculate_metrics_and_ci(y_true, y_pred, pos_label, n_bootstrap=1000):
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        y_true_pos = (y_true == pos_label)
        y_pred_pos = (y_pred == pos_label)

        TP = np.sum(y_true_pos & y_pred_pos)
        TN = np.sum(~y_true_pos & ~y_pred_pos)
        FP = np.sum(~y_true_pos & y_pred_pos)
        FN = np.sum(y_true_pos & ~y_pred_pos)

        n = len(y_true)
        acc = (TP + TN) / n
        recall = TP / (TP + FN) if (TP + FN) != 0 else np.nan
        specificity = TN / (TN + FP) if (TN + FP) != 0 else np.nan
        ppv = TP / (TP + FP) if (TP + FP) != 0 else np.nan
        npv = TN / (TN + FN) if (TN + FN) != 0 else np.nan
        precision = ppv
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else np.nan

        ci_acc = proportion_confint(TP + TN, n, method='wilson')
        ci_recall = proportion_confint(TP, TP + FN, method='wilson') if (TP + FN) > 0 else (np.nan, np.nan)
        ci_specificity = proportion_confint(TN, TN + FP, method='wilson') if (TN + FP) > 0 else (np.nan, np.nan)
        ci_ppv = proportion_confint(TP, TP + FP, method='wilson') if (TP + FP) > 0 else (np.nan, np.nan)
        ci_npv = proportion_confint(TN, TN + FN, method='wilson') if (TN + FN) > 0 else (np.nan, np.nan)

        f1_scores = []
        for _ in range(n_bootstrap):
            indices = np.random.choice(n, size=n, replace=True)
            y_true_bs = y_true[indices]
            y_pred_bs = y_pred[indices]

            tp_bs = np.sum((y_true_bs == pos_label) & (y_pred_bs == pos_label))
            fn_bs = np.sum((y_true_bs == pos_label) & (y_pred_bs != pos_label))
            fp_bs = np.sum((y_true_bs != pos_label) & (y_pred_bs == pos_label))

            precision_bs = tp_bs / (tp_bs + fp_bs) if (tp_bs + fp_bs) > 0 else np.nan
            recall_bs = tp_bs / (tp_bs + fn_bs) if (tp_bs + fn_bs) > 0 else np.nan
            f1_bs = 2 * (precision_bs * recall_bs) / (precision_bs + recall_bs) if (
                                                                                               precision_bs + recall_bs) > 0 else np.nan
            f1_scores.append(f1_bs)

        f1_scores_valid = [x for x in f1_scores if not np.isnan(x)]
        ci_f1 = (np.percentile(f1_scores_valid, 2.5), np.percentile(f1_scores_valid, 97.5)) if f1_scores_valid else (
            np.nan, np.nan)

        metrics = {
            'acc': acc, 'f1': f1, 'recall': recall,
            'specificity': specificity, 'ppv': ppv, 'npv': npv
        }

        cis = {
            'acc': ci_acc, 'recall': ci_recall, 'specificity': ci_specificity,
            'ppv': ci_ppv, 'npv': ci_npv, 'f1': ci_f1
        }

        return metrics, cis

    # 转换输入为 numpy 格式
    for key in mapped_doctor_results:
        mapped_doctor_results[key] = np.array(mapped_doctor_results[key])
    mapped_labels = np.array(mapped_labels)

    # 汇总结果：按 Positive=0 / Positive=1 存入不同 sheet
    all_results = {0: {}, 1: {}}
    for doctor in mapped_doctor_results:
        y_true = mapped_labels
        y_pred = mapped_doctor_results[doctor]
        for pos_label in [0, 1]:
            metrics, cis = calculate_metrics_and_ci(y_true, y_pred, pos_label)
            row = {}
            for metric in metrics:
                row[f"{metric.upper()}_Value"] = metrics[metric]
                row[f"{metric.upper()}_CI_Low"] = cis[metric][0]
                row[f"{metric.upper()}_CI_High"] = cis[metric][1]
            all_results[pos_label][doctor] = row

    # 写入 Excel，医生为行，每个 positive 值一张 sheet
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        for pos_label in [0, 1]:
            df = pd.DataFrame.from_dict(all_results[pos_label], orient='index')
            df.index.name = 'Doctor'
            df.reset_index(inplace=True)
            df.to_excel(writer, sheet_name=f'Positive={pos_label}', index=False)

    print(f"📄 Results saved to: {excel_path}")


def plot_doctor_predictions_0123(map_case, color_map):

    # 读取 Excel 数据
    file_path = '/Volumes/WHY-SSD/Experimentation/data_0123_人工与辅助判读_0227用.xlsx'

    # 读取Excel文件
    data = pd.read_excel(file_path, sheet_name='AI_AD03')

    # 自动加载医生标识符，从第7列到第27列
    doctors = data.columns[6:24].tolist()

    # 通过 'Label' 获取真实标签
    labels = data['Label'].values
    # 根据医生结果列来处理医生判读数据
    doctor_results = {doctor: data[doctor].values for doctor in doctors}

    # 映射标签
    mapped_labels = apply_mapping_to_labels(labels, map_case)
    mapped_doctor_results = apply_mapping_to_doctors_predictions(doctor_results, map_case)
    print('mapped_labels', mapped_labels)
    print('mapped_doctor_results', mapped_doctor_results)
    evaluate_doctor_predictions_to_excel(mapped_labels, mapped_doctor_results, excel_path='/Volumes/WHY-SSD/trubt_paper_pics/浸润与非浸润.xlsx')
    doctors = list(mapped_doctor_results.keys())
    labels = mapped_labels
    # 创建图形
    fig, ax = plt.subplots(figsize=(15, 10))

    # 设置格子尺寸和间距
    cell_size = 20  # 格子大小
    gap_size = 5  # 格子间距
    horizontal_gap = 2
    vertical_gap = 2

    ax.set_aspect('equal')

    # 绘制标签行
    for j, label in enumerate(labels):
        x_pos = j * (cell_size + gap_size)
        y_pos = (len(doctors)) * (cell_size + gap_size)  # 标签行位置在顶部
        ax.add_patch(patches.Rectangle((x_pos, y_pos), cell_size, cell_size, linewidth=1, edgecolor='white',
                                       facecolor=color_map[label]))
    # 添加Label标识文本
    ax.text(-horizontal_gap, (len(doctors)) * (cell_size + gap_size) + cell_size / 2,
            'GT', va='center', ha='right', fontsize=12, color='black')
    # 绘制医生预测行
    for i, doctor in enumerate(doctors):
        for j in range(len(labels)):
            doctor_prediction = mapped_doctor_results[doctor][j]
            x_pos = j * (cell_size + gap_size)
            y_pos = (len(doctors) - i - 1) * (cell_size + gap_size)  # Y轴位置

            ax.add_patch(patches.Rectangle((x_pos, y_pos), cell_size, cell_size, linewidth=1, edgecolor='white',
                                           facecolor=color_map[doctor_prediction]))

        # 添加医生标识
        ax.text(-horizontal_gap, (len(doctors) - i - 1) * (cell_size + gap_size) + cell_size / 2,
                doctor, va='center', ha='right', fontsize=12, color='black')

    # 设置坐标轴
    ax.set_xlim(0, len(labels) * (cell_size + gap_size))
    ax.set_ylim(0, (len(doctors) + 1) * (cell_size + gap_size))

    # 隐藏坐标轴
    ax.axis('off')
    ax.set_title('Subtyping results(invasive and non-invasive urothelial (bladder) carcinoma)', fontsize=16)

    legend_elements = [
        Line2D([0], [0], marker='s', color='w', label='Non-tumor', markersize=20, markerfacecolor=color_map[0],
               markeredgecolor='white'),
        Line2D([0], [0], marker='s', color='w', label='Non-invasive carcinoma', markersize=20, markerfacecolor=color_map[1],
               markeredgecolor='white'),
        Line2D([0], [0], marker='s', color='w', label='Invasive carcinoma', markersize=20, markerfacecolor=color_map[2],
               markeredgecolor='white'),
    ]
    ax.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.1), fontsize=12, title="", ncol=4, frameon=False)

    # 显示图像
    save_dir = '/Volumes/WHY-SSD/trubt_paper_pics/人工判读与算法辅助对比'
    os.makedirs(save_dir, exist_ok=True)  # 确保目录存在
    save_path = os.path.join(save_dir, f"人工与辅助判读对比的小方格_浸润与非浸润.pdf")
    plt.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight')
    # plt.show()


def plot_doctor_predictions_012(map_case, color_map):

    # 读取 Excel 数据
    file_path = '/Volumes/WHY-SSD/Experimentation/data_0123_人工与辅助判读_0227用.xlsx'


    # 读取Excel文件
    data = pd.read_excel(file_path, sheet_name='AI_AD03')

    # 自动加载医生标识符，从第7列到第27列
    doctors = data.columns[6:24].tolist()

    # 通过 'Label' 获取真实标签
    labels = data['Label'].values
    # 根据医生结果列来处理医生判读数据
    doctor_results = {doctor: data[doctor].values for doctor in doctors}

    # 映射标签
    mapped_labels = apply_mapping_to_labels(labels, map_case)  # 使用情况1的映射
    mapped_doctor_results = apply_mapping_to_doctors_predictions(doctor_results, map_case)  # 使用情况1的映射
    print('mapped_labels', mapped_labels)
    print('mapped_doctor_results', mapped_doctor_results)
    evaluate_doctor_predictions_to_excel(mapped_labels, mapped_doctor_results, excel_path='/Volumes/WHY-SSD/trubt_paper_pics/高低级别.xlsx')
    doctors = list(mapped_doctor_results.keys())
    labels = mapped_labels
    fig, ax = plt.subplots(figsize=(15, 10))

    # 设置格子尺寸和间距
    cell_size = 20  # 格子大小
    gap_size = 5  # 格子间距
    horizontal_gap = 2
    vertical_gap = 2

    ax.set_aspect('equal')

    # 绘制标签行
    for j, label in enumerate(labels):
        x_pos = j * (cell_size + gap_size)
        y_pos = (len(doctors)) * (cell_size + gap_size)  # 标签行位置在顶部
        ax.add_patch(patches.Rectangle((x_pos, y_pos), cell_size, cell_size, linewidth=1, edgecolor='white',
                                       facecolor=color_map[label]))
    # 添加Label标识文本
    ax.text(-horizontal_gap, (len(doctors)) * (cell_size + gap_size) + cell_size / 2,
            'GT', va='center', ha='right', fontsize=12, color='black')
    # 绘制医生预测行
    for i, doctor in enumerate(doctors):
        for j in range(len(labels)):
            doctor_prediction = mapped_doctor_results[doctor][j]
            x_pos = j * (cell_size + gap_size)
            y_pos = (len(doctors) - i - 1) * (cell_size + gap_size)  # Y轴位置

            ax.add_patch(patches.Rectangle((x_pos, y_pos), cell_size, cell_size, linewidth=1, edgecolor='white',
                                           facecolor=color_map[doctor_prediction]))

        # 添加医生标识
        ax.text(-horizontal_gap, (len(doctors) - i - 1) * (cell_size + gap_size) + cell_size / 2,
                doctor, va='center', ha='right', fontsize=12, color='black')

    # 设置坐标轴
    ax.set_xlim(0, len(labels) * (cell_size + gap_size))
    ax.set_ylim(0, (len(doctors) + 1) * (cell_size + gap_size))

    # 隐藏坐标轴
    ax.axis('off')
    ax.set_title('Subtyping results(Low-grade and High-grade)', fontsize=16)

    legend_elements = [
        Line2D([0], [0], marker='s', color='w', label='Non-tumor', markersize=20, markerfacecolor=color_map[0],
               markeredgecolor='white'),
        Line2D([0], [0], marker='s', color='w', label='Low-grade', markersize=20, markerfacecolor=color_map[1],
               markeredgecolor='white'),
        Line2D([0], [0], marker='s', color='w', label='High-grade', markersize=20, markerfacecolor=color_map[2],
               markeredgecolor='white'),
    ]
    ax.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.1), fontsize=12, title="", ncol=4, frameon=False)

    # 显示图像
    save_dir = '/Volumes/WHY-SSD/trubt_paper_pics/人工判读与算法辅助对比'
    os.makedirs(save_dir, exist_ok=True)  # 确保目录存在
    save_path = os.path.join(save_dir, f"人工与辅助判读对比的小方格_高低级别.pdf")
    plt.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight')


def plot_doctor_predictions_01(map_case, color_map):

    # 读取 Excel 数据
    file_path = '/Volumes/WHY-SSD/Experimentation/data_0123_人工与辅助判读_0227用.xlsx'


    # 读取Excel文件
    data = pd.read_excel(file_path, sheet_name='AI_AD03')

    # 自动加载医生标识符，从第7列到第27列
    doctors = data.columns[6:24].tolist()

    # 通过 'Label' 获取真实标签
    labels = data['Label'].values
    # 根据医生结果列来处理医生判读数据
    doctor_results = {doctor: data[doctor].values for doctor in doctors}

    # 映射标签
    mapped_labels = apply_mapping_to_labels(labels, map_case)  # 使用情况1的映射
    mapped_doctor_results = apply_mapping_to_doctors_predictions(doctor_results, map_case)  # 使用情况1的映射
    print('mapped_labels', mapped_labels)
    print('mapped_doctor_results', mapped_doctor_results)
    metric_ci(mapped_labels, mapped_doctor_results)
    doctors = list(mapped_doctor_results.keys())
    labels = mapped_labels
    # 创建图形
    fig, ax = plt.subplots(figsize=(15, 10))

    # 设置格子尺寸和间距
    cell_size = 20  # 格子大小
    gap_size = 5  # 格子间距
    horizontal_gap = 2
    vertical_gap = 2

    ax.set_aspect('equal')

    # 绘制标签行
    for j, label in enumerate(labels):
        x_pos = j * (cell_size + gap_size)
        y_pos = (len(doctors)) * (cell_size + gap_size)  # 标签行位置在顶部
        ax.add_patch(patches.Rectangle((x_pos, y_pos), cell_size, cell_size, linewidth=1, edgecolor='white',
                                       facecolor=color_map[label]))
    # 添加Label标识文本
    ax.text(-horizontal_gap, (len(doctors)) * (cell_size + gap_size) + cell_size / 2,
            'GT', va='center', ha='right', fontsize=12, color='black')
    # 绘制医生预测行
    for i, doctor in enumerate(doctors):
        for j in range(len(labels)):
            doctor_prediction = mapped_doctor_results[doctor][j]
            x_pos = j * (cell_size + gap_size)
            y_pos = (len(doctors) - i - 1) * (cell_size + gap_size)  # Y轴位置

            ax.add_patch(patches.Rectangle((x_pos, y_pos), cell_size, cell_size, linewidth=1, edgecolor='white',
                                           facecolor=color_map[doctor_prediction]))

        # 添加医生标识
        ax.text(-horizontal_gap, (len(doctors) - i - 1) * (cell_size + gap_size) + cell_size / 2,
                doctor, va='center', ha='right', fontsize=12, color='black')

    # 设置坐标轴
    ax.set_xlim(0, len(labels) * (cell_size + gap_size))
    ax.set_ylim(0, (len(doctors) + 1) * (cell_size + gap_size))

    # 隐藏坐标轴
    ax.axis('off')
    ax.set_title('Subtyping results(Non-tumor and tumor)', fontsize=16)

    legend_elements = [
        Line2D([0], [0], marker='s', color='w', label='Non-tumor', markersize=20, markerfacecolor=color_map[0],
               markeredgecolor='white'),
        Line2D([0], [0], marker='s', color='w', label='tumor', markersize=20, markerfacecolor=color_map[1],
               markeredgecolor='white'),
    ]
    ax.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.1), fontsize=12, title="", ncol=4, frameon=False)

    # 显示图像
    save_dir = '/Volumes/WHY-SSD/trubt_paper_pics/人工判读与算法辅助对比'
    os.makedirs(save_dir, exist_ok=True)  # 确保目录存在
    save_path = os.path.join(save_dir, f"人工与辅助判读对比的小方格_癌与非癌.pdf")
    plt.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight')



# 计算混淆矩阵指标及其95%置信区间
def compute_metrics_ci(label, pred, pos_label=1, n_bootstraps=1000, alpha=0.95):
    from sklearn.metrics import roc_curve, auc, confusion_matrix
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


if __name__ == '__main__':
    # 定义颜色映射
    color_map_0123 = {
        0: '#B3CCAF',  # 最浅
        1: '#59AA87',
        2: '#137D74',
        3: '#164E5F'  # 最深
    }
    color_map_012 = {
        0: '#B3CCAF',  # 最浅
        1: '#59AA87',
        2: '#137D74',
    }
    color_map_01 = {
        0: '#B3CCAF',  # 最浅
        1: '#137D74',
        2: '#137D74',
    }
    # 调用函数
    plot_doctor_predictions_0123(map_case_0123, color_map_0123)
    plot_doctor_predictions_012(map_case_012, color_map_012)
    plot_doctor_predictions_01(map_case_01, color_map_01)
    # plot_doctor_predictions_everylabel(map_case_012, color_map)
    # plot_doctor_predictions_everylabel(map_case_0123,  color_map)
