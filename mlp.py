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

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import make_scorer, r2_score


def main():
    # ------------------------------ data preparing ------------------------------
    # ===== 读取数据 =====
    source_path = "G:/homework/machine learning/kaggle/input/"
    save_path = "G:/homework/machine learning/kaggle/output/pred_mlp.csv"
    flood_train = pd.read_csv(source_path + "train.csv")
    sample = pd.read_csv(source_path + "sample_submission.csv")

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

    # 特征缩放
    sc = preprocessing.StandardScaler()
    X_train_scaled = preprocessing.scale(X_train)
    scaler = preprocessing.StandardScaler().fit(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    # scaler = preprocessing.StandardScaler()
    # X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    # X_val_scaled = pd.DataFrame(scaler.fit_transform(X_val), columns=X_val.columns)
    # X_test_scaled = pd.DataFrame(scaler.fit_transform(X_test), columns=X_test.columns)

    # ------------------------------ model preparing ------------------------------
    model = Sequential()
    model.add(Dense(64, kernel_initializer='normal', activation='relu',
                    input_shape=(20,)))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1))

    model.compile(
        loss='mse',
        optimizer=RMSprop(learning_rate=0.0005),
        metrics=['mean_absolute_error']
    )

    # ------------------------------- start training -------------------------------
    history = model.fit(
        X_train_scaled, Y_train,
        batch_size=128,
        epochs=100,
        verbose=1,
        validation_split=0.2,
        callbacks=[EarlyStopping(monitor='val_loss', patience=20)]
    )

    plt.plot(history.epoch, history.history.get('loss'), label="Training loss")
    plt.plot(history.epoch, history.history.get('val_loss'), label="Validation loss")
    plt.title('Training and validation loss')
    plt.xlabel("epoch")
    plt.ylabel("MSE")
    plt.legend()
    plt.savefig('./loss.jpg', dpi=400, bbox_inches='tight')
    plt.show()

    plt.plot(history.epoch, history.history.get('mean_absolute_error'), label="Training MAE")
    plt.plot(history.epoch, history.history.get('val_mean_absolute_error'), label="Validation MAE")
    plt.title('Training and validation MAE metric')
    plt.xlabel("epoch")
    plt.ylabel("mean absolute error")
    plt.legend()
    plt.savefig('./mae.jpg', dpi=400, bbox_inches='tight')
    plt.show()

    # ------------------------------ start evaluating ------------------------------
    score = model.evaluate(X_val_scaled, Y_val, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    # -------------------------------- start testing --------------------------------
    Y_pred = model.predict(X_test_scaled)
    Y_pred = Y_pred.squeeze()
    # sample['id'] = id_test
    # sample['FloodProbability'] = Y_pred
    # sample.to_csv(save_path, index=None)
    output_flood_prediction = pd.DataFrame({'id': id_test, 'FloodProbability': Y_pred, 'label': Y_test})
    output_flood_prediction.to_csv(os.path.join(save_path), index=False)

    # metric
    Y_val_pred = model.predict(X_val_scaled)
    print("val:", r2_score(Y_val, Y_val_pred))
    Y_test_pred = model.predict(X_test_scaled)
    print("test:", r2_score(Y_test, Y_test_pred))


# R²评价函数
def R2_score(y_true, y_pred):
    return r2_score(y_true, y_pred)


if __name__ == "__main__":
    main()
