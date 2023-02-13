import numpy as np
import pandas as pd
from keras.datasets import boston_housing
from sklearn import preprocessing
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split



(x_train, y_train), (x_valid, y_valid) = boston_housing.load_data()  # 加载数据
min_max_scaler = preprocessing.MinMaxScaler()

X_train = min_max_scaler.fit_transform(x_train)
y_train = min_max_scaler.fit_transform(y_train.reshape(-1, 1))  # reshape(-1,1)指将它转化为1列，行自动确定
X_test = min_max_scaler.fit_transform(x_valid)
y_test = min_max_scaler.fit_transform(y_valid.reshape(-1, 1))

x_train_single = X_train[:,1]



class LinerRegression():

    def __init__(self,data,labels):

        self.data = data
        self.labels = labels
        num_features = self.data.shape[1]
        self.theta = np.zeros((num_features,1))


    def train(self, alpha, num_iteration = 500):
        cost_history = self.gradient_descent(alpha, num_iteration)
        return self.theta, cost_history

    def gradient_descent(self, alpha, num_iteration):
        cost_history = []
        for i in range(num_iteration):
            self.gradient_step(alpha)
            cost_history.append(self.cost_function(self.data,self.labels))
        return cost_history

    def gradient_step(self, alpha):
        num_example = self.data.shape[0]
        prediction = LinerRegression.hypothesis(self.data,self.theta)
        delta = prediction - self.labels
        theta = self.theta
        theta = theta - alpha*(1/num_example)*(np.dot(delta.T,self.data)).T
        self.theta = theta


    @staticmethod
    def hypothesis(data, theta):
        predictions = np.dot(data,theta)
        return predictions

    def predict(self,data):
        prediction = LinerRegression.hypothesis(data, self.theta)
        return prediction

    def cost_function(self,data,labels):
        num_examples = data.shape[0]
        delta = LinerRegression.hypothesis(self.data,self.theta) - labels
        cost = (1/num_examples)*np.dot(delta.T,delta)
        return cost[0][0]

linear_regression = LinerRegression(X_train,y_train)
(theta,cost_history) = linear_regression.train(0.001,500)
print(cost_history[0])
print(cost_history[-1])
plt.plot(range(500),cost_history)
plt.show()

#模型评估方法
#混淆矩阵
#TP, FP, FN, TN
#precision = TP/TP+FP recall = TP/TP+FN 可以用sklearn现成方法计算
#F1 score调和平均数
#阈值对评估指标的影响 decision_function   precision_recall_curve
#ROC曲线 roc_auc_score






