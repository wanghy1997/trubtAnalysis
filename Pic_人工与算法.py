import pandas as pd
import scipy.stats as st



def map_case_0123(label):
    """将0合并为新0，1/2合并为新1，3/4合并为新2，5/6合并为新3"""
    if label in [0, 1, 2]:
        return 0
    elif label in [3, 4]:
        return 1
    elif label in [5, 6]:
        return 2
    else:
        return label  # 保持未知标签不变

def calculate_metrics(true_labels, pred_labels):
    """计算 ACC, Sensitivity, Specificity, PPV, NPV 以及 95% CI"""
    tp = ((true_labels == 1) & (pred_labels == 1)).sum()  # True Positive
    tn = ((true_labels == 0) & (pred_labels == 0)).sum()  # True Negative
    fp = ((true_labels == 0) & (pred_labels == 1)).sum()  # False Positive
    fn = ((true_labels == 1) & (pred_labels == 0)).sum()  # False Negative

    total = len(true_labels)

    # 计算各项指标
    acc = (tp + tn) / total if total > 0 else 0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0  # Recall
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0  # Precision
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0

    # 计算 95% 置信区间（CI）
    def compute_ci(metric, total_count):
        if total_count > 0:
            ci_low, ci_high = st.t.interval(0.95, df=total_count - 1, loc=metric, scale=st.sem([metric] * total_count))
            return max(0, ci_low), min(1, ci_high)
        else:
            return 0, 0

    acc_ci = compute_ci(acc, total)
    sensitivity_ci = compute_ci(sensitivity, tp + fn)
    specificity_ci = compute_ci(specificity, tn + fp)
    ppv_ci = compute_ci(ppv, tp + fp)
    npv_ci = compute_ci(npv, tn + fn)

    return acc, sensitivity, specificity, ppv, npv, acc_ci, sensitivity_ci, specificity_ci, ppv_ci, npv_ci

def process_and_calculate_metrics(file_path, sheet_name='AI_AD03'):
    """读取Excel数据，进行类别映射，并计算各医生的ACC指标"""
    # 读取Excel文件
    xls = pd.ExcelFile(file_path)
    df = pd.read_excel(xls, sheet_name=sheet_name)

    # 获取医生列（排除非预测列）
    doctor_columns = df.columns[6:14].tolist()
    # 进行映射
    df['Mapped_Label'] = df['Label'].apply(map_case_0123)
    for col in doctor_columns:
        df[f'Mapped_{col}'] = df[col].apply(map_case_0123)

    # 计算每个医生的指标
    metrics_results = []
    for col in doctor_columns:
        acc, sensitivity, specificity, ppv, npv, acc_ci, sensitivity_ci, specificity_ci, ppv_ci, npv_ci = calculate_metrics(
            df['Mapped_Label'], df[f'Mapped_{col}'])
        metrics_results.append(
            [col, acc, acc_ci, sensitivity, sensitivity_ci, specificity, specificity_ci, ppv, ppv_ci, npv, npv_ci])

    # 转换为DataFrame
    metrics_df = pd.DataFrame(metrics_results,
                              columns=['Doctor', 'Accuracy', 'ACC_CI95%', 'Sensitivity', 'Sensitivity_CI95%',
                                       'Specificity', 'Specificity_CI95%', 'PPV', 'PPV_CI95%', 'NPV', 'NPV_CI95%'])

    return metrics_df


if __name__ == '__main__':

    # 运行封装的方法
    file_path = '/Volumes/WHY-SSD/Experimentation/data_0123_人工与辅助判读_0227用.xlsx'
    with pd.option_context('display.max_rows', None,
                           'display.max_columns', None,
                           'display.width', None):
        accuracy_df = process_and_calculate_metrics(file_path)
        print(accuracy_df)
    # 展示结果

