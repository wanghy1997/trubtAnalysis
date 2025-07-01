import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, recall_score
import matplotlib.lines as mlines
import os
import matplotlib.patches as mpatches
"""
非癌，非癌和癌 0123
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
all_doctors = junior_doctors + intermediate_doctors + senior_doctors  # 所有医生

# 算法的指标（非癌 & 癌）
algorithm_metrics = {
    'Non-tumor': {'Accuracy': 0.9714, 'F1 Score': 0.7368, 'Recall': 0.7000, 'Specificity': 0.8000, 'PPV': 0.8000, 'NPV': 0.8000},
    'Tumor': {'Accuracy': 0.9714, 'F1 Score': 0.8000, 'Recall': 0.8000, 'Specificity': 0.8000, 'PPV': 0.8000, 'NPV': 0.8000}
}



def drawAHistogram(save_dir, results_df):
    """
        绘制柱状图，展示不同医生在非浸润性癌和浸润性癌的准确率。
    :param save_dir: 生成的 pdf 要保存的路径，例如  '/Volumes/WHY-SSD/trubt_paper_pics/浸润性癌'，生成的.pdf 文件会保存到‘浸润性癌’文件夹里。
    :param results_df: 包含医生、类别和准确率的 DataFrame，格式应为三列：Doctor, Category, Accuracy。
    :return:
    """
    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.lines as mlines
    import matplotlib.patches as mpatches

    # 手动设置颜色（分别为：ours, junior, intermediate, senior）
    color_inv = {
        'Ours': '#d1dac5',  # 浅灰绿
        'Junior': '#61acf4',  # 浅蓝
        'Intermediate': '#f4d44c',  # 浅黄
        'Senior': '#e0788c'  # 浅红
    }
    color_non = {
        'Ours': '#d95350',  # 红
        'Junior': '#cce4fc',  # 同色斜线填充
        'Intermediate': '#fdfcec',
        'Senior': '#f1e2de'
    }

    # 医生顺序（与数据匹配）
    ordered_doctors = [
        'Ours',
        'J-1', 'J-2', 'J-3', 'J-4',
        'I-1', 'I-2', 'I-3', 'I-4',
        'S-1', 'S-2', 'S-3', 'S-4'
    ]

    # 医生级别映射（用于决定颜色）
    level_map = {
        'Ours': 'Ours',
        **{f'J-{i}': 'Junior' for i in range(1, 5)},
        **{f'I-{i}': 'Intermediate' for i in range(1, 5)},
        **{f'S-{i}': 'Senior' for i in range(1, 5)},
    }

    # print("【results_df.head()】")
    # print(results_df.head(20))  # 显示前20行
    #
    # print("\n【results_df['Doctor'].unique()】")
    # print(results_df['Doctor'].unique())  # 显示所有医生名称
    #
    # print("\n【results_df.columns】")
    # print(results_df.columns)  # 确认是否真的有 'Doctor'、'Category'、'Accuracy' 三列

    # 转成宽格式
    pivot_df = results_df.pivot(index='Doctor', columns='Category', values='Accuracy').reset_index()
    pivot_df.columns = ['Doctor', 'Non-tumor', 'Tumor']
    pivot_df = pivot_df.set_index('Doctor').loc[ordered_doctors].reset_index()
    pivot_df['Level'] = pivot_df['Doctor'].map(level_map)

    # 开始绘图
    fig, ax = plt.subplots(figsize=(20, 6))
    bar_width = 0.35
    x = np.arange(len(pivot_df))

    # 🎯 先绘制 invasive（纯色） → 左边
    for idx, row in pivot_df.iterrows():
        level = row['Level']
        bar = ax.bar(x[idx] - bar_width / 2, row['Tumor'], width=bar_width,
                     color=color_inv[level])
        ax.text(x[idx] - bar_width / 2, row['Tumor'] + 0.01, f"{row['Tumor']:.2f}",
                ha='center', va='bottom', fontsize=10)

    # 🎯 再绘制 non-invasive（斜线） → 右边
    for idx, row in pivot_df.iterrows():
        level = row['Level']
        bar = ax.bar(x[idx] + bar_width / 2, row['Non-tumor'], width=bar_width,
                     color=color_non[level], hatch='/', linewidth=1.5)
        ax.text(x[idx] + bar_width / 2, row['Non-tumor'] + 0.01, f"{row['Non-tumor']:.2f}",
                ha='center', va='bottom', fontsize=10)



    # 设置x轴
    ax.set_xticks(x)
    ax.set_xticklabels(pivot_df['Doctor'], rotation=45, ha='right', fontsize=10)
    ax.set_ylabel('Accuracy')
    ax.set_ylim([0, 1.05])
    ax.set_title('Accuracy for Non-tumor carcinoma and Tumor carcinoma', fontsize=14)

    # 分组分隔线
    for bound in [0.5, 4.5, 8.5]:
        ax.axvline(x=bound, color='gray' if bound > 0.5 else 'black', linestyle='--', linewidth=1.5)

    # 图例（六类）
    legend_handles = [
        mpatches.Patch(facecolor=color_inv['Ours'], edgecolor='black', label='Algorithm (Tumor carcinoma)'),
        mpatches.Patch(facecolor='white', edgecolor=color_non['Ours'], hatch='///',
                       label='Algorithm (Non-tumor carcinoma)'),
        mpatches.Patch(facecolor=color_inv['Junior'], edgecolor='black', label='Junior (Tumor carcinoma)'),
        mpatches.Patch(facecolor='white', edgecolor=color_non['Junior'], hatch='///', label='Junior (Non-tumor carcinoma)'),
        mpatches.Patch(facecolor=color_inv['Intermediate'], edgecolor='black', label='Intermediate (Tumor carcinoma)'),
        mpatches.Patch(facecolor='white', edgecolor=color_non['Intermediate'], hatch='///', label='Intermediate (Non-tumor carcinoma)'),
        mpatches.Patch(facecolor=color_inv['Senior'], edgecolor='black', label='Senior (Tumor carcinoma)'),
        mpatches.Patch(facecolor='white', edgecolor=color_non['Senior'], hatch='///', label='Senior (Non-tumor carcinoma)'),
    ]

    ax.legend(handles=legend_handles, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=4,
              frameon=False, fontsize=10, columnspacing=1.5, handletextpad=1)

    # 去除顶部和右侧边框
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    save_path = os.path.join(save_dir, "file.pdf")
    plt.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight')
    plt.show()

    print(f"生成的柱状图 pdf 格式的文件，已保存至: {save_path}")


# 计算每位医生的性能，并增加 Specificity, PPV, NPV 指标
def calculate_additional_metrics(true_labels, pred_labels):
    """计算 Specificity, PPV, NPV """
    tp = ((true_labels == 1) & (pred_labels == 1)).sum()  # True Positive
    tn = ((true_labels == 0) & (pred_labels == 0)).sum()  # True Negative
    fp = ((true_labels == 0) & (pred_labels == 1)).sum()  # False Positive
    fn = ((true_labels == 1) & (pred_labels == 0)).sum()  # False Negative

    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0  # Precision
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0

    return specificity, ppv, npv

# 类别合并映射
def merge_labels(labels):
    """合并原始标签，仅保留类别 2（非癌）和类别 3（癌）"""
    labels = np.array(labels)
    labels_new = np.full_like(labels, -1)  # 初始化 -1 避免错误
    labels_new[(labels == 0) | (labels == 1) | (labels == 2)] = 0  # 新的类别 0（正常）
    labels_new[(labels == 3) | (labels == 4) | (labels == 5) | (labels == 6)] = 1  # 非癌
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
for category in ['Non-tumor', 'Tumor']:
    results['Doctor'].append('Ours')
    results['Category'].append(category)
    results['Accuracy'].append(algorithm_metrics[category]['Accuracy'])
    results['F1 Score'].append(algorithm_metrics[category]['F1 Score'])
    results['Recall'].append(algorithm_metrics[category]['Recall'])
    results['Specificity'].append(algorithm_metrics[category]['Specificity'])
    results['PPV'].append(algorithm_metrics[category]['PPV'])
    results['NPV'].append(algorithm_metrics[category]['NPV'])



# 计算每位医生的性能
for doctor in all_doctors:
    predicted_labels = merge_labels(data[doctor].values)

    for category, category_label in [('Non-tumor', 0), ('Tumor', 1)]:
        mask = true_labels == category_label
        if np.sum(mask) == 0:
            continue

        accuracy = accuracy_score(true_labels[mask], predicted_labels[mask])
        f1 = f1_score(true_labels[mask], predicted_labels[mask], average='weighted', zero_division=0)
        recall = recall_score(true_labels[mask], predicted_labels[mask], average='weighted', zero_division=0)
        # specificity, ppv, npv = calculate_additional_metrics(true_labels[mask], predicted_labels[mask])
            # 当前类别作为正类
        tp = ((true_labels == category_label) & (predicted_labels == category_label)).sum()
        fp = ((true_labels != category_label) & (predicted_labels == category_label)).sum()
        tn = ((true_labels != category_label) & (predicted_labels != category_label)).sum()
        fn = ((true_labels == category_label) & (predicted_labels != category_label)).sum()

        # 计算各项指标
        # accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) != 0 else 0
        precision = tp / (tp + fp) if (tp + fp) != 0 else 0
        # recall = tp / (tp + fn) if (tp + fn) != 0 else 0
        # f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) != 0 else 0
        ppv = precision  # PPV就是precision
        npv = tn / (tn + fn) if (tn + fn) != 0 else 0

        results['Doctor'].append(doctor)
        results['Category'].append(category)
        results['Accuracy'].append(accuracy)
        results['F1 Score'].append(f1)
        results['Recall'].append(recall)
        results['Specificity'].append(specificity)
        results['PPV'].append(ppv)
        results['NPV'].append(npv)

# 转换为 DataFrame
results_df = pd.DataFrame(results)

# 设定颜色
colors = ['#D9534F', '#9DC4C4', '#D1DAC5', '#90A7C4']  # 算法：红色，初级：天蓝色，中级：淡绿色，高级：灰蓝色

# 设置全局字体
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12  # 默认字号


drawAHistogram(save_dir=save_dir, results_df=results_df)  # 绘制柱状图并保存
# # 绘制 Non-muscle-invasive 和 Muscle-invasive 的柱状图并保存
# for category_name in ['Non-tumor', 'Tumor']:
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
#         # # **保存文件**
#         os.makedirs(save_dir, exist_ok=True)  # 确保目录存在
#         save_path = os.path.join(save_dir, f"{metric}_for_{category_name}.pdf")
#         plt.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight')
#
#
#
# print(f"所有图像已保存至: {save_dir}")

colors = ['#D9534F', '#D1DAC5', '#cce4fc', '#60acf4', '#fcfcec', '#f4d44c', '#f1e9e7', '#e0788c']

# Assuming results_df is a DataFrame with 'Doctor', 'Category', 'Accuracy' columns
# Example: results_df = pd.DataFrame({'Doctor': ['Algorithm', 'Junior1', ...], 'Category': ['High-garde', 'Low-garde', ...], 'Accuracy': [0.9, 0.85, ...]})


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
    high_val = pivot_df['Tumor'].iloc[i]
    low_val = pivot_df['Non-tumor'].iloc[i]
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
ax.set_title('Accuracy for Non-tumor and Tumor')

# Remove right and top spines for cleaner look
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

# Add legend for High-grade and Low-grade
# ax.legend(['Non-tumor', 'Tumor'], loc='upper center', bbox_to_anchor=(0.5, -0.08), ncol=2, frameon=False,
#           fontsize=12)
legend_patches = [
    mpatches.Patch(facecolor=colors[1], label='Algorithm (Tumor)'),
    mpatches.Patch(facecolor=colors[0], label='Algorithm (Non-tumor)'),
    mpatches.Patch(facecolor=colors[3], label='Junior (Tumor)'),
    mpatches.Patch(facecolor=colors[2], hatch='//', label='Junior (Non-tumor)'),
    mpatches.Patch(facecolor=colors[5], label='Intermediate (Tumor)'),
    mpatches.Patch(facecolor=colors[4], hatch='//', label='Intermediate (Non-tumor)'),
    mpatches.Patch(facecolor=colors[7], label='Senior (Tumor)'),
    mpatches.Patch(facecolor=colors[6], hatch='//', label='Senior (Non-tumor)')
]
ax.legend(handles=legend_patches, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=4, frameon=False, fontsize=10)



# Adjust layout and save the plot
plt.tight_layout()
os.makedirs(save_dir, exist_ok=True)  # 确保目录存在
save_path = os.path.join(save_dir, f"Non-tumor and Tumor_35.pdf")
plt.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight')
# plt.show()
plt.close(fig)




# 以表格形式打印数据
from tabulate import tabulate
# 分别创建两个表格：Non-tumor 和 Tumor
results_df_non_invasive = results_df[results_df["Category"] == "Non-tumor"]
results_df_invasive = results_df[results_df["Category"] == "Tumor"]
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
| Ours     | Non-tumor |     0.9714 |     0.7368 |      0.7 |           0.8 |   0.8 |   0.8 |
+----------+------------------------+------------+------------+----------+---------------+-------+-------+
+----------+--------------------+------------+------------+----------+---------------+-------+-------+
| Doctor   | Category           |   Accuracy |   F1 Score |   Recall |   Specificity |   PPV |   NPV |
+==========+====================+============+============+==========+===============+=======+=======+
| Ours     | Tumor |     0.9714 |        0.8 |      0.8 |           0.8 |   0.8 |   0.8 |
+----------+--------------------+------------+------------+----------+---------------+-------+-------+

"""