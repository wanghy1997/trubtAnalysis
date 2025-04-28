import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D


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
