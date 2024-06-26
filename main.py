import re
import numpy as np
import pandas as pd
import matplotlib as mlp
import matplotlib.pyplot as plt
import time
import tqdm
import os
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import make_scorer, r2_score
# 算法辅助 & 数据
import sklearn
from sklearn.model_selection import KFold, cross_validate, cross_val_score, train_test_split  # 交叉验证

# 算法模型
from sklearn.linear_model import ElasticNet, Lasso, BayesianRidge,Ridge,LinearRegression # 线性回归模型
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor # 集成模型
from sklearn.kernel_ridge import KernelRidge  # 核岭回归
from sklearn.pipeline import make_pipeline  # pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler  # 标准化
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone  # 自定义类的API
from sklearn.ensemble import StackingRegressor

# 指标
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

# 数据可视化
from data_process import *

def train_val_model(X_train,Y_train,X_val,Y_val,X_test,Y_test,model_name='LinearRegression'):
    """
    调用各种模型测试

    :param X_train:
    :param Y_train:
    :param X_val:
    :param Y_val:
    :param X_test:
    :param Y_test:
    :param model_name: 模型名称
    :return:
    """

    if model_name == 'LinearRegression':
        model = LinearRegression()
    elif model_name == 'Lasso':
        model = Lasso()
    elif model_name == 'BayesianRidge':
        model = BayesianRidge()
    elif model_name == 'Ridge':
        model = Ridge()
    elif model_name == 'ElasticNet':
        model = ElasticNet()
    elif model_name == 'SVR':
        model = SVR()
    elif model_name == 'RandomForestRegressor':
        model = RandomForestRegressor()
    elif model_name == 'XGBRegressor':
        model = XGBRegressor()
    elif model_name == 'LGBMRegressor':
        model = LGBMRegressor()
    # 训练
    model.fit(X_train, Y_train)
    # val
    val_predict = model.predict(X_val)
    val_mse = mean_absolute_error(Y_val, val_predict)
    val_r2 = r2_score(Y_val, val_predict)
    print(f"{model_name}  val_mse: {val_mse:.4f}, val_r2: {val_r2:.4f}")
    # test
    test_predict = model.predict(X_test)
    test_mse = mean_absolute_error(Y_test, test_predict)
    test_r2 = r2_score(Y_test, test_predict)
    print(f"{model_name}  test_mse: {test_mse:.4f}, test_r2: {test_r2:.4f}")
    res_dic = {'val_mse':val_mse, 'val_r2':val_r2, 'test_mse':test_mse, 'test_r2':test_r2}
    np.save(f"{model_name}_res.npy", res_dic)
    # 返回用于保存各个模型的效果
    return val_mse, val_r2, test_mse, test_r2

def main():
    # ------------------------------ data preparing ------------------------------
    # ===== 读取数据 =====
    source_path = './data/train.csv'
    save_path = './data/pred_stacking1.csv'
    flood_train = pd.read_csv(os.path.join(source_path))

    # ===== 数据预处理 =====
    # 检查缺失值
    print(flood_train.isna().sum())
    # 检查异常值
    print(flood_train.describe())
    # 特征缩放
    scaler = StandardScaler()
    flood_train_scaled = pd.DataFrame(scaler.fit_transform(flood_train), columns=flood_train.columns)

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

    #-------------------------------数据可视化----------------------------------
    pro_X_train = pd.DataFrame(X_train)
    pro_Y_train = pd.DataFrame(Y_train)
    pro_data = pd.concat([pro_X_train, pro_Y_train], axis=1)
    #get_summary_statistics(pro_data)

    #get_data_distribution(pro_data)

    #get_data_corr(pro_data)

    #plot_distribution_with_stats(Y_train)

    plot_feature_distribution(X_train)
    pass

    #---------------------train val test------------------------------
    model_list = ['LinearRegression', 'Lasso', 'BayesianRidge', 'Ridge', 'ElasticNet', 'SVR', 'XGBRegressor',
                  'LGBMRegressor']
    # 保存每个模型的的训练效果
    val_mse_list = []
    val_r2_list = []
    test_mse_list = []
    test_r2_list = []

    for model in model_list:
        print(f"using {model}........\n")
        val_mse, val_r2, test_mse, test_r2 = train_val_model(X_train,Y_train,X_val,Y_val,X_test,Y_test,model_name=model)
        val_mse_list.append(val_mse)
        val_r2_list.append(val_r2)
        test_mse_list.append(test_mse)
        test_r2_list.append(test_r2)

    np.save('contrast_experiment.npy',{'val_mse':val_mse_list,'val_r2':val_r2_list,'test_mse':test_mse_list,'test_r2':test_r2_list})



    # # ------------------------------ base model ------------------------------
    # # ===== model preparing =====
    #
    # ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3))
    # KRR = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)
    # GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
    #                                    max_depth=4, max_features='sqrt',
    #                                    min_samples_leaf=15, min_samples_split=10,
    #                                    loss='huber', random_state=5)
    # lasso = make_pipeline(RobustScaler(), Lasso(alpha=0.0005, random_state=1))
    #
    # # ===== train & evaluate =====
    # score_mse = mse_cv(lasso, 5, X_train, Y_train)
    # score_r2 = r2_cv(lasso, 5, X_train, Y_train)
    # print("Lasso score mse: {:.4f} r2: {:.4f}\n".format(score_mse.mean(), score_r2.mean()))
    #
    # score_mse = mse_cv(ENet, 5, X_train, Y_train)
    # score_r2 = r2_cv(ENet, 5, X_train, Y_train)
    # print("ElasticNet score mse: {:.4f} r2: {:.4f}\n".format(score_mse.mean(), score_r2.mean()))
    #
    # # score_mse = mse_cv(KRR, 5, X_train, Y_train)
    # # score_r2 = r2_cv(KRR, 5, X_train, Y_train)
    # # print("Kernel Ridge score mse: {:.4f} r2: {:.4f}\n".format(score_mse, score_r2))
    # # score = rmsle_cv(KRR, 5, X_train, Y_train)
    # # print("Kernel Ridge score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
    #
    # score_mse = mse_cv(GBoost, 5, X_train, Y_train)
    # score_r2 = r2_cv(GBoost, 5, X_train, Y_train)
    # print("Gradient Boosting score mse: {:.4f} r2: {:.4f}\n".format(score_mse.mean(), score_r2.mean()))
    #
    # # ------------------------------ meta model ------------------------------
    # # ===== model preparing =====
    # stacked_models = StackingAveragedModels(base_models=(ENet, GBoost, KRR), meta_model=lasso)
    #
    # score_mse = mse_cv(stacked_models, 5, X_train, Y_train)
    # score_r2 = r2_cv(stacked_models, 5, X_train, Y_train)
    # print("Stacking Averaged models score mse: {:.4f} r2: {:.4f}\n".format(score_mse.mean(), score_r2.mean()))
    # # score = rmsle_cv(stacked_models, 5, X_train, Y_train)
    # # print("Stacking Averaged models score: {:.4f} ({:.4f})".format(score.mean(), score.std()))
    #
    # # ===== train & evaluate =====
    # stacked_models.fit(X_train, Y_train)
    # stacked_train_pred = stacked_models.predict(X_train)
    # stacked_pred = np.expm1(stacked_models.predict(X_test))
    #
    # mse_train = mean_squared_error(Y_train, stacked_train_pred)
    # r2_train = r2_score(Y_train, stacked_train_pred)
    # mse_test = mean_squared_error(Y_test, stacked_pred)
    # r2_test = r2_score(Y_test, stacked_pred)
    # print("train:", r2_train, mse_train)
    # print("test:", r2_test, mse_test)
    #
    # # ------------------------------ save prediction ------------------------------
    # output_flood_prediction = pd.DataFrame({'id': id_test, 'FloodProbability': stacked_pred})
    # output_flood_prediction.to_csv(os.path.join(save_path), index=False)

#--------------------以下都没用到--------------------------------------------
# Stacking模型
class StackingAveragedModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, base_models, meta_model, n_folds=5):
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_folds = n_folds

    def fit(self, X, y):
        self.base_models_ = [list() for x in self.base_models]
        self.meta_model_ = clone(self.meta_model)
        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=156)
        out_of_fold_predictions = np.zeros((X.shape[0], len(self.base_models)))

        for i, model in enumerate(self.base_models):
            for train_index, holdout_index in kfold.split(X, y):
                instance = clone(model)
                self.base_models_[i].append(instance)
                instance.fit(X[train_index], y[train_index])
                y_pred = instance.predict(X[holdout_index])
                out_of_fold_predictions[holdout_index, i] = y_pred

        self.meta_model_.fit(out_of_fold_predictions, y)
        return self

    def predict(self, X):
        meta_features = np.column_stack([
            np.column_stack([model.predict(X) for model in base_models]).mean(axis=1)
            for base_models in self.base_models_])
        return self.meta_model_.predict(meta_features)


# k折交叉验证
def mse_cv(model, n_folds, X_train, Y_train):
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(X_train)
    mse = -cross_val_score(model, X_train, Y_train, scoring="neg_mean_squared_error", cv=kf)
    return mse


def r2_cv(model, n_folds, X_train, Y_train):
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(X_train)
    r2 = cross_val_score(model, X_train, Y_train, scoring="r2", cv=kf)
    return r2


if __name__ == "__main__":
    main()
