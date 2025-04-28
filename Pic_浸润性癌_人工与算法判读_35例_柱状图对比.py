import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, recall_score
import matplotlib.lines as mlines
import os
import matplotlib.patches as mpatches
"""
非癌，非浸润和浸润 0123
不同级别医生的各项指标，以及算法判读的各项指标，
"""

# 读取 Excel 数据（修改为你的本地文件路径）
file_path = '/Volumes/WHY-SSD/Experimentation/data_0123_人工与辅助判读_0227用.xlsx'
save_dir = '/Volumes/WHY-SSD/trubt_paper_pics/人工判读与算法判读对比/35例'

# 确保保存目录存在
os.makedirs(save_dir, exist_ok=True)

data = pd.read_excel(file_path, sheet_name='combine')

# 医生的标识符
junior_doctors = ['J-1', 'J-2', 'J-3', 'J-4']
intermediate_doctors = ['I-1', 'I-2', 'I-3', 'I-4']
senior_doctors = ['S-1', 'S-2', 'S-3', 'S-4']
all_doctors = junior_doctors + intermediate_doctors + senior_doctors # 所有医生

# 算法的指标（非浸润 & 浸润）
algorithm_metrics = {
    'Non-invasive carcinoma': {'Accuracy': 0.8286, 'F1 Score': 0.7368, 'Recall': 0.7000, 'Specificity': 0.8000, 'PPV': 0.8000, 'NPV': 0.8000},
    'Invasive carcinoma': {'Accuracy': 0.8571, 'F1 Score': 0.8000, 'Recall': 0.8000, 'Specificity': 0.8000, 'PPV': 0.8000, 'NPV': 0.8000}
}

# 计算 Specificity, PPV, NPV，针对任意正类标签
def calculate_additional_metrics(true_labels, pred_labels, pos_label):
    """计算 Specificity, PPV, NPV，针对 pos_label 作为正类 """
    tp = ((true_labels == pos_label) & (pred_labels == pos_label)).sum()  # True Positive
    tn = ((true_labels != pos_label) & (pred_labels != pos_label)).sum()  # True Negative
    fp = ((true_labels != pos_label) & (pred_labels == pos_label)).sum()  # False Positive
    fn = ((true_labels == pos_label) & (pred_labels != pos_label)).sum()  # False Negative

    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0  # Precision
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0

    return specificity, ppv, npv

# 类别合并映射
def merge_labels(labels):
    """合并原始标签，仅保留类别 2（非浸润）和类别 3（浸润）"""
    labels = np.array(labels)
    labels_new = np.full_like(labels, -1)  # 初始化 -1 避免错误
    labels_new[(labels == 0) | (labels == 1) | (labels == 2)] = 0  # 新的类别 0（正常）
    labels_new[(labels == 3) | (labels == 4)] = 2  # 非浸润
    labels_new[(labels == 5) | (labels == 6)] = 3  # 浸润

    return labels_new

# 处理真实标签
true_labels = merge_labels(data['Label'].values)

# 重新初始化结果存储
results = {
    'Doctor': [],
    'Category': [],
    'Accuracy': [],
    'F1 Score': [],
    'Recall': [],
    'Specificity': [],
    'PPV': [],
    'NPV': []
}

# 添加算法的数据
for category in ['Non-invasive carcinoma', 'Invasive carcinoma']:
    results['Doctor'].append('Ours')
    results['Category'].append(category)
    results['Accuracy'].append(algorithm_metrics[category]['Accuracy'])
    results['F1 Score'].append(algorithm_metrics[category]['F1 Score'])
    results['Recall'].append(algorithm_metrics[category]['Recall'])
    results['Specificity'].append(algorithm_metrics[category]['Specificity'])
    results['PPV'].append(algorithm_metrics[category]['PPV'])
    results['NPV'].append(algorithm_metrics[category]['NPV'])



# 计算每位医生的性能（修改部分）
for doctor in all_doctors:
    predicted_labels = merge_labels(data[doctor].values)

    for category, pos_label in [('Non-invasive carcinoma', 2), ('Invasive carcinoma', 3)]:
        # 将标签转换为二分类：pos_label 为正类（1），其他为负类（0）
        true_binary = (true_labels == pos_label).astype(int)
        pred_binary = (predicted_labels == pos_label).astype(int)

        # 计算指标
        accuracy = accuracy_score(true_binary, pred_binary)
        f1 = f1_score(true_binary, pred_binary, zero_division=0)
        recall = recall_score(true_binary, pred_binary, zero_division=0)
        specificity, ppv, npv = calculate_additional_metrics(true_labels, predicted_labels, pos_label)

        # 存储结果
        results['Doctor'].append(doctor)
        results['Category'].append(category)
        results['Accuracy'].append(accuracy)
        results['F1 Score'].append(f1)
        results['Recall'].append(recall)
        results['Specificity'].append(specificity)
        results['PPV'].append(ppv)
        results['NPV'].append(npv)


# # 计算每位医生的性能
# for doctor in all_doctors:
#     predicted_labels = merge_labels(data[doctor].values)
#
#     for category, category_label in [('Non-invasive carcinoma', 2), ('Invasive carcinoma', 3)]:
#         mask = true_labels == category_label
#         if np.sum(mask) == 0:
#             continue
#
#         accuracy = accuracy_score(true_labels[mask], predicted_labels[mask])
#         f1 = f1_score(true_labels[mask], predicted_labels[mask], average='weighted', zero_division=0)
#         recall = recall_score(true_labels[mask], predicted_labels[mask], average='weighted', zero_division=0)
#         specificity, ppv, npv = calculate_additional_metrics(true_labels[mask], predicted_labels[mask])
#
#         results['Doctor'].append(doctor)
#         results['Category'].append(category)
#         results['Accuracy'].append(accuracy)
#         results['F1 Score'].append(f1)
#         results['Recall'].append(recall)
#         results['Specificity'].append(specificity)
#         results['PPV'].append(ppv)
#         results['NPV'].append(npv)


# 转换为 DataFrame
results_df = pd.DataFrame(results)

# 设定颜色
# colors = ['#D9534F', '#9DC4C4', '#D1DAC5', '#90A7C4']  # 算法：红色，初级：天蓝色，中级：淡绿色，高级：灰蓝色

# 设置全局字体
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12  # 默认字号


colors = ['#D9534F', '#D1DAC5', '#cce4fc', '#60acf4', '#fcfcec', '#f4d44c', '#f1e9e7', '#e0788c']

# Assuming results_df is a DataFrame with 'Doctor', 'Category', 'Accuracy' columns
# Example: results_df = pd.DataFrame({'Doctor': ['Algorithm', 'Junior1', ...], 'Category': ['High-garde', 'Low-garde', ...], 'Accuracy': [0.9, 0.85, ...]})

# Pivot the DataFrame to have 'Non-invasive carcinoma', 'Invasive carcinoma' as columns
pivot_df = results_df.pivot(index='Doctor', columns='Category', values='Accuracy')
# desired_order = ['Ours'] + [d for d in pivot_df.index if d != 'Ours']
# pivot_df = pivot_df.reindex(desired_order)
#
# doctors = pivot_df.index  # Now 'Ours' is guaranteed to be first
# 构建 desired_order，确保 'Ours' 在前，后面是 all_doctors 的顺序
desired_order = ['Ours'] + all_doctors

# 重新索引 pivot_df
pivot_df = pivot_df.reindex(desired_order)

doctors = pivot_df.index  # Now 'Ours' is guaranteed to be first

# Define group indices based on doctor categories
# Order: Algorithm (1), Junior (4), Intermediate (4), Senior (4)
group_indices = [0] + [1] * 4 + [2] * 4 + [3] * 4  # Total 13 doctors

# Set bar width to ensure a small gap
bar_width = 0.25

# Set x positions for each doctor
x = np.arange(len(doctors))

# Create figure and axis
fig, ax = plt.subplots(figsize=(15, 6))


# Plot bars for each doctor
for i in range(len(doctors)):
    group = group_indices[i]
    color_high = colors[group * 2 + 1]  # Color for 'High-garde'
    color_low = colors[group * 2]  # Color for 'Low-garde'
    high_val = pivot_df['Invasive carcinoma'].iloc[i]
    low_val = pivot_df['Non-invasive carcinoma'].iloc[i]
    # Plot 'Low-garde' bar (dashed outline for Junior, Intermediate, Senior)
    hatch = '//' if group in [1, 2, 3] else None  # Apply hatching for Junior (1), Intermediate (2), Senior (3)
    ax.bar(x[i] + bar_width / 2, low_val, width=bar_width, color=color_low, hatch=hatch)

    # Plot 'High-garde' bar (solid)
    ax.bar(x[i] - bar_width / 2, high_val, width=bar_width, color=color_high)
    # Add accuracy values on top of bars
    ax.text(x[i] + bar_width / 2, low_val + 0.01, f'{low_val:.2f}', ha='center', va='bottom', fontsize=10)
    ax.text(x[i] - bar_width / 2, high_val + 0.01, f'{high_val:.2f}', ha='center', va='bottom', fontsize=10)


# Add separation lines between doctor categories
ax.axvline(x=0.5, color='black', linestyle='--')  # After Algorithm
ax.axvline(x=4.5, color='gray', linestyle='--')  # After Junior
ax.axvline(x=8.5, color='gray', linestyle='--')  # After Intermediate

# Set x-ticks and labels
ax.set_xticks(x)
ax.set_xticklabels(doctors, rotation=45)

# Set labels and limits
ax.set_ylabel('Accuracy')
ax.set_ylim([0, 1])
ax.set_title('Accuracy for Non-invasive carcinoma and Invasive carcinoma')

# Remove right and top spines for cleaner look
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

# # Add legend for High-grade and Non-invasive carcinoma
# ax.legend(['Non-invasive carcinoma', 'Invasive carcinoma'], loc='upper center', bbox_to_anchor=(0.5, -0.08), ncol=2, frameon=False,
#           fontsize=12)
# Add legend for doctor categories and hatch patterns
legend_patches = [
    mpatches.Patch(facecolor=colors[1], label='Algorithm (Invasive carcinoma)'),
    mpatches.Patch(facecolor=colors[0], label='Algorithm (Non-invasive carcinoma)'),
    mpatches.Patch(facecolor=colors[3], label='Junior (Invasive carcinoma)'),
    mpatches.Patch(facecolor=colors[2], hatch='//', label='Junior (Non-invasive carcinoma)'),
    mpatches.Patch(facecolor=colors[5], label='Intermediate (Invasive carcinoma)'),
    mpatches.Patch(facecolor=colors[4], hatch='//', label='Intermediate (Non-invasive carcinoma)'),
    mpatches.Patch(facecolor=colors[7], label='Senior (Invasive carcinoma)'),
    mpatches.Patch(facecolor=colors[6], hatch='//', label='Senior (Non-invasive carcinoma)')
]
ax.legend(handles=legend_patches, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=4, frameon=False, fontsize=10)


# Adjust layout and save the plot
plt.tight_layout()
os.makedirs(save_dir, exist_ok=True)  # 确保目录存在
save_path = os.path.join(save_dir, f"Non-invasive carcinoma and Invasive carcinoma_35.pdf")
plt.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight')
# plt.show()
plt.close(fig)


# # 绘制 Non-muscle-invasive 和 Muscle-invasive 的柱状图并保存
# for category_name in ['Non-invasive carcinoma', 'Invasive carcinoma']:
#     category_df = results_df[results_df['Category'] == category_name]
#
#     for metric in ['Accuracy']:
#         fig, ax = plt.subplots(figsize=(12, 6))
#
#         # X 轴坐标
#         x = np.arange(len(category_df['Doctor']))
#         bar_width = 0.5  # 设置柱子的宽度
#
#         # 绘制柱状图
#         ax.bar(x[0], category_df[metric].iloc[0], color=colors[0], label='Ours', width=bar_width)  # 算法
#         ax.bar(x[1:5], category_df[metric].iloc[1:5], color=colors[1], label='Junior Doctors', width=bar_width)
#         ax.bar(x[5:9], category_df[metric].iloc[5:9], color=colors[2], label='Intermediate Doctors', width=bar_width)
#         ax.bar(x[9:], category_df[metric].iloc[9:], color=colors[3], label='Senior Doctors', width=bar_width)
#
#         # 计算并绘制每个级别的平均值虚线
#         avg_junior = np.mean(category_df[metric].iloc[1:5])
#         avg_intermediate = np.mean(category_df[metric].iloc[5:9])
#         avg_senior = np.mean(category_df[metric].iloc[9:])
#
#         # 添加分隔线
#         ax.axvline(x=0.5, color='black', linestyle='--')  # 算法和初级医生之间
#         ax.axvline(x=4.5, color='gray', linestyle='--')  # 初级和中级医生之间
#         ax.axvline(x=8.5, color='gray', linestyle='--')  # 中级和高级医生之间
#
#         # 绘制平均值虚线
#         ax.plot([0, len(category_df['Doctor'])], [avg_junior, avg_junior], color=colors[1], linestyle='--', linewidth=2)
#         ax.plot([0, len(category_df['Doctor'])], [avg_intermediate, avg_intermediate], color=colors[2], linestyle='--', linewidth=2)
#         ax.plot([0, len(category_df['Doctor'])], [avg_senior, avg_senior], color=colors[3], linestyle='--', linewidth=2)
#
#         # 设置 X 轴
#         ax.set_xlim([-0.5, len(category_df['Doctor']) - 0.5])
#         ax.set_xticks(x)
#         ax.set_xticklabels(category_df['Doctor'], rotation=45)
#         ax.set_ylabel(metric)
#         ax.set_ylim([0, 1])
#         ax.set_title(f"{category_name}")
#
#         # 删除右侧和上方的边框
#         ax.spines['right'].set_color('none')
#         ax.spines['top'].set_color('none')
#
#         # **创建虚线图例**
#         legend_lines = [
#             mlines.Line2D([], [], color=colors[1], linestyle='--', linewidth=2, label='Average for Junior'),
#             mlines.Line2D([], [], color=colors[2], linestyle='--', linewidth=2, label='Average for Intermediate'),
#             mlines.Line2D([], [], color=colors[3], linestyle='--', linewidth=2, label='Average for Senior'),
#         ]
#
#         ax.legend(handles=legend_lines, loc='upper center', bbox_to_anchor=(0.5, -0.08), ncol=3, frameon=False, fontsize=12, handletextpad=2, columnspacing=5)
#
#         plt.tight_layout()
#         plt.show()
#         save_path = os.path.join(save_dir, f"{metric}_for_{category_name}_人工判读润色.pdf")
#         plt.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight')
#         plt.close(fig)
#
# print(f"所有图像已保存至: {save_dir}")

# 以表格形式打印数据
from tabulate import tabulate
# 分别创建两个表格：Non-invasive carcinoma 和 Invasive carcinoma
results_df_non_invasive = results_df[results_df["Category"] == "Non-invasive carcinoma"]
results_df_invasive = results_df[results_df["Category"] == "Invasive carcinoma"]
# 转换 DataFrame 为表格格式
table_str_1 = tabulate(results_df_non_invasive, headers='keys', tablefmt='grid', showindex=False)
table_str_2 = tabulate(results_df_invasive, headers='keys', tablefmt='grid', showindex=False)

# 打印表格
print(table_str_1)
print(table_str_2)

"""
+----------+------------------------+------------+------------+----------+---------------+-------+-------+
| Doctor   | Category               |   Accuracy |   F1 Score |   Recall |   Specificity |   PPV |   NPV |
+==========+========================+============+============+==========+===============+=======+=======+
| Ours     | Non-invasive carcinoma |        0.6 |   0.7368   |      0.7 |           0.8 |   0.8 |   0.8 |
+----------+------------------------+------------+------------+----------+---------------+-------+-------+
| J-1      | Non-invasive carcinoma |        0.3 |   0.461538 |      0.3 |           0   |   0   |   0   |
+----------+------------------------+------------+------------+----------+---------------+-------+-------+
| J-2      | Non-invasive carcinoma |        0.7 |   0.823529 |      0.7 |           0   |   0   |   0   |
+----------+------------------------+------------+------------+----------+---------------+-------+-------+
| J-3      | Non-invasive carcinoma |        0.4 |   0.571429 |      0.4 |           0   |   0   |   0   |
+----------+------------------------+------------+------------+----------+---------------+-------+-------+
| J-4      | Non-invasive carcinoma |        0.7 |   0.823529 |      0.7 |           0   |   0   |   0   |
+----------+------------------------+------------+------------+----------+---------------+-------+-------+
| I-1      | Non-invasive carcinoma |        0.8 |   0.888889 |      0.8 |           0   |   0   |   0   |
+----------+------------------------+------------+------------+----------+---------------+-------+-------+
| I-2      | Non-invasive carcinoma |        0.8 |   0.888889 |      0.8 |           0   |   0   |   0   |
+----------+------------------------+------------+------------+----------+---------------+-------+-------+
| I-3      | Non-invasive carcinoma |        0.9 |   0.947368 |      0.9 |           0   |   0   |   0   |
+----------+------------------------+------------+------------+----------+---------------+-------+-------+
| I-4      | Non-invasive carcinoma |        0.7 |   0.823529 |      0.7 |           0   |   0   |   0   |
+----------+------------------------+------------+------------+----------+---------------+-------+-------+
+----------+--------------------+------------+------------+----------+---------------+-------+-------+
| Doctor   | Category           |   Accuracy |   F1 Score |   Recall |   Specificity |   PPV |   NPV |
+==========+====================+============+============+==========+===============+=======+=======+
| Ours     | Invasive carcinoma |        0.8 |   0.8      |      0.8 |           0.8 |   0.8 |   0.8 |
+----------+--------------------+------------+------------+----------+---------------+-------+-------+
| J-1      | Invasive carcinoma |        0.3 |   0.461538 |      0.3 |           0   |   0   |   0   |
+----------+--------------------+------------+------------+----------+---------------+-------+-------+
| J-2      | Invasive carcinoma |        0.6 |   0.75     |      0.6 |           0   |   0   |   0   |
+----------+--------------------+------------+------------+----------+---------------+-------+-------+
| J-3      | Invasive carcinoma |        0.7 |   0.823529 |      0.7 |           0   |   0   |   0   |
+----------+--------------------+------------+------------+----------+---------------+-------+-------+
| J-4      | Invasive carcinoma |        0.3 |   0.461538 |      0.3 |           0   |   0   |   0   |
+----------+--------------------+------------+------------+----------+---------------+-------+-------+
| I-1      | Invasive carcinoma |        0.6 |   0.75     |      0.6 |           0   |   0   |   0   |
+----------+--------------------+------------+------------+----------+---------------+-------+-------+
| I-2      | Invasive carcinoma |        0.4 |   0.571429 |      0.4 |           0   |   0   |   0   |
+----------+--------------------+------------+------------+----------+---------------+-------+-------+
| I-3      | Invasive carcinoma |        1   |   1        |      1   |           0   |   0   |   0   |
+----------+--------------------+------------+------------+----------+---------------+-------+-------+
| I-4      | Invasive carcinoma |        0.6 |   0.75     |      0.6 |           0   |   0   |   0   |
+----------+--------------------+------------+------------+----------+---------------+-------+-------+

"""