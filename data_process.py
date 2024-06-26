# 加载库
import pandas as pd
import numpy as np
from numpy import mean
from numpy import std
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
from sklearn.metrics import f1_score, classification_report, confusion_matrix, roc_auc_score, roc_curve, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
warnings.filterwarnings('ignore')

sns.set()
sns.set_palette('muted')
SNS_CMAP = 'muted'

colors = sns.palettes.color_palette(SNS_CMAP)

def get_summary_statistics(data):
    """
    这个没用
    :param data:
    :return:
    """
    data.describe().style.background_gradient(cmap=SNS_CMAP)
    pass

def get_data_distribution(data,data_set_name = 'Train_data'):
    # Set the style and size for the plots
    """
    画出数据的分布图，均值
    :param data: 传入的数据，传的是训练数据，包括特征和标签
    :param data_set_name:  用于保存名字
    :return:
    """
    sns.set(style='whitegrid')
    sns.set_palette('mako')
    #plt.figure(figsize=(16, 24))

    sns.boxplot(data=data)
    plt.title(data_set_name)
    # 设置x轴的label的排列方式，rotation表示旋转角度，90度表示竖排
    plt.xticks(rotation=90)
    # 设置保存方式，不然下面的标签显示不出来
    plt.tight_layout()
    plt.savefig("Train_data_distribution.png")
    plt.show()

def get_data_corr(data,data_set_name = 'Train_data'):
    """
    计算特征之间的相关度
    :param data: 传入训练数据，包括特征和label
    :param data_set_name:
    :return:
    """
    # Calculate the correlation matrix for train_data
    #pd数据格式可以包含了计算相关度的方法
    correlation_matrix = data.corr()
    # Plot the correlation heatmap
    plt.figure(figsize=(14, 12))
    # 画热力图
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='flare', cbar=True)
    plt.title(f'Correlation Matrix for {data_set_name}')
    plt.savefig(data_set_name+"_CorrelationMatrix.png",bbox_inches = 'tight')
    plt.show()


def plot_distribution_with_stats(data, data_set_name = 'Train_data'):
    """
    画出训练数据的标签值的分布，即Probability的分布
    :param data: 训练数据，只包含标签那一列
    :param data_set_name:
    :return:
    """
    plt.figure(figsize=(10, 5))
    # 设置颜色风格，画分布图
    sns.histplot(data, kde=True, color='red', bins=30)
    # 计算均值、众数
    mean_value = data.mean()
    median_value = data.median()

    plt.axvline(mean_value, color='r', linestyle='--', label=f'Mean: {mean_value:.2f}')
    plt.axvline(median_value, color='g', linestyle='-', label=f'Median: {median_value:.2f}')
    plt.title(f"{data_set_name} Flood Probability Distribution")
    plt.xlabel('Flood Probability')
    plt.ylabel('Frequency')
    #plt.legend()
    plt.savefig(f"{data_set_name} Flood Probability Distribution",bbox_inches = 'tight')
    plt.show()

def plot_mlp_performance():
    """
    这个没用到，当时说重合了
    :return:
    """
    source_path = './data/pred_mlp.csv'
    save_path = './data/pred_stacking1.csv'
    flood_train = pd.read_csv(os.path.join(source_path))
    mlp_pred = flood_train['FloodProbability']  # Y标签
    #label = flood_train.drop(['FloodProbability'], axis=1)
    label = flood_train['label']  # X特征

    plt.figure(figsize=(10, 5))
    sns.histplot(mlp_pred, kde=True, color='red', bins=30,alpha=0.5)
    mlp_mean_value = mlp_pred.mean()
    mlp_median_value = mlp_pred.median()
    plt.axvline(mlp_mean_value, color='r', linestyle='--', label=f'mlp_Mean: {mlp_mean_value:.2f}')
    plt.axvline(mlp_median_value, color='g', linestyle='-', label=f'mlp_Median: {mlp_median_value:.2f}')
    plt.legend()
    plt.title(f"Prediction Flood Probability Distribution")
    plt.xlabel('Flood Probability')
    plt.ylabel('Frequency')
    plt.savefig(f"Prediction FloodProbability Distribution.png", bbox_inches='tight')
    plt.show()

    plt.figure(figsize=(10, 5))
    sns.histplot(mlp_pred, kde=True, color='g', bins=30,alpha=0.5)
    label_mean_value = label.mean()
    label_median_value = label.median()
    plt.axvline(mlp_mean_value, color='pink', linestyle='--', label=f'label_Mean: {label_mean_value:.2f}')
    plt.axvline(mlp_median_value, color='b', linestyle='-', label=f'label_Median: {label_median_value:.2f}')
    plt.title(f"GT Flood Probability Distribution")
    plt.xlabel('Flood Probability')
    plt.ylabel('Frequency')
    # plt.legend()
    plt.legend()
    plt.savefig(f"GT FloodProbability Distribution.png", bbox_inches='tight')
    plt.show()

def plot_feature_distribution(train_data_drop_id):
    # cont_cols = [f for f in train_data_drop_id.columns if
    #              train_data_drop_id[f].dtype in [float, int] and train_data_drop_id[f].nunique() > 2 and f not in [
    #                  "FloodProbability"]]
    cont_cols = [f for f in train_data_drop_id.columns if
                 train_data_drop_id[f].nunique() > 2 and f not in [
                     "FloodProbability"]]
    cont_cols = cont_cols[:6]
    num_cols = 3
    num_rows = (len(cont_cols) + num_cols - 1) // num_cols  # 确保整数行数

    # Calculate the number of rows needed for the subplots
    # num_rows = (len(cont_cols) + 2) // 4

    # Create subplots for each continuous column
    # fig, axs = plt.subplots(num_rows, 3, figsize=(18, num_rows * 5), constrained_layout=True)
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(18, num_rows * 5), constrained_layout=True)

    # Loop through each continuous column and plot the histograms
    for i, col in enumerate(cont_cols):
        # Determine the range of values to plot
        # max_val = max(train_data_drop_id[col].max())
        # min_val = min(train_data_drop_id[col].min())
        max_val = train_data_drop_id[col].max()
        min_val = train_data_drop_id[col].min()
        range_val = max_val - min_val

        # Determine the bin size and number of bins
        bin_size = range_val / 20
        num_bins_train = round(range_val / bin_size)
        num_bins_test = round(range_val / bin_size)
        num_bins_original = round(range_val / bin_size)

        # Calculate the subplot position
        row = i // num_cols
        col_pos = i % num_cols

        # Plot the histograms
        sns.histplot(train_data_drop_id[col], ax=axs[row][col_pos], color='darkturquoise', kde=True, label='Train',
                     bins=num_bins_train)
        # sns.histplot(test_data_drop_id[col], ax=axs[row][col_pos], color='salmon', kde=True, label='Test',
        #              bins=num_bins_test)
        # sns.histplot(original_data[col], ax=axs[row][col_pos], color='orange', kde=True, label='Original',
        #              bins=num_bins_original)
        axs[row][col_pos].set_title(col)
        axs[row][col_pos].set_xlabel('Value')
        axs[row][col_pos].set_ylabel('Frequency')
        axs[row][col_pos].legend()

    # Remove any empty subplots

    if len(cont_cols) % 3 != 0:
        for col_pos in range(len(cont_cols) % num_cols, num_cols):
            axs[-1][col_pos].remove()

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    plot_mlp_performance()
