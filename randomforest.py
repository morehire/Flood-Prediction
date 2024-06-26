import re
import numpy as np
import pandas as pd
import matplotlib as mlp
import matplotlib.pyplot as plt
import time
import os
import tensorflow as tf
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.optimizers import RMSprop
from keras.callbacks import EarlyStopping
from sklearn import preprocessing
from sklearn.preprocessing import scale
from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import make_scorer, r2_score


def main():
    # ------------------------------ data preparing ------------------------------
    # ===== 读取数据 =====
    source_path = "G:/homework/machine learning/kaggle/input/"
    # save_path = 'G:/homework/machine learning/kaggle/output/pred_mlp.csv'
    flood_train = pd.read_csv(source_path + "train.csv")

    # ===== 数据预处理 =====
    # # 检查缺失值
    # print(flood_train.isna().sum())
    # # 检查异常值
    # print(flood_train.describe())

    # ===== 数据划分 （80% 训练集，10% 验证集，10% 测试集）=====
    Y = flood_train['FloodProbability']  # Y标签
    X = flood_train.drop(['FloodProbability'], axis=1)  # X特征
    X_train, X_temp, Y_train, Y_temp = train_test_split(X, Y, test_size=0.2, random_state=42)
    X_val, X_test, Y_val, Y_test = train_test_split(X_temp, Y_temp, test_size=0.5, random_state=42)

    id_train = X_train['id']
    id_val = X_val['id']
    id_test = X_test['id']
    X_train = X_train.drop(['id'], axis=1)  # 丢弃'id'特征值
    X_val = X_val.drop(['id'], axis=1)
    X_test = X_test.drop(['id'], axis=1)

    # # 特征缩放
    # sc = preprocessing.StandardScaler()
    # X_train_scaled = preprocessing.scale(X_train)
    # scaler = preprocessing.StandardScaler().fit(X_train)
    # X_val_scaled = scaler.transform(X_val)
    # X_test_scaled = scaler.transform(X_test)

    # ------------------------------ model preparing ------------------------------
    line = RandomForestRegressor()

    # ------------------------------- start training -------------------------------
    line.fit(X_train, Y_train)

    # ------------------------------- visualize feature importance -------------------------------
    feature_importances = line.feature_importances_
    # 创建特征名列表
    feature_names = list(X_train.columns)
    # 创建一个DataFrame，包含特征名和其重要性得分
    feature_importances_df = pd.DataFrame({'feature': feature_names, 'importance': feature_importances})
    # 对特征重要性得分进行排序
    feature_importances_df = feature_importances_df.sort_values('importance', ascending=False)

    # 颜色映射
    colors = plt.cm.viridis(np.linspace(0, 1, len(feature_names)))

    # 可视化特征重要性
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(feature_importances_df['feature'], feature_importances_df['importance'], color=colors)
    ax.invert_yaxis()  # 翻转y轴，使得最大的特征在最上面
    ax.set_xlabel('特征重要性', fontsize=12)  # 图形的x标签
    ax.set_title('随机森林特征重要性可视化', fontsize=16)
    for i, v in enumerate(feature_importances_df['importance']):
        ax.text(v + 0.01, i, str(round(v, 3)), va='center', fontname='Times New Roman', fontsize=10)

    # 设置图形样式
    ax.spines['top'].set_visible(False)  # 去掉上边框
    ax.spines['right'].set_visible(False)  # 去掉右边框

    plt.savefig('./feature importance.jpg', dpi=400, bbox_inches='tight')
    plt.show()

    # ------------------------------ start evaluating ------------------------------
    Y_val_pred = line.predict(X_val)
    mse_val = mean_absolute_error(Y_val, Y_val_pred)
    r2_val = r2_score(Y_val, Y_val_pred)
    print("val: mse=",mse_val, "r2=", r2_val)

    # ------------------------------ start testing ------------------------------
    Y_test_pred = line.predict(X_test)
    mse_test = mean_absolute_error(Y_test, Y_test_pred)
    r2_test = r2_score(Y_test, Y_test_pred)
    print("test: mse=", mse_test, "r2=", r2_test)


if __name__ == "__main__":
    main()