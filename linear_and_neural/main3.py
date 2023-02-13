import pandas as pd
import numpy as np
import scipy.stats as stats
import sklearn
from keras.datasets import boston_housing

from sklearn.preprocessing import MinMaxScaler
import pandas as pd




def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class MultilayerPerception:
    def __init__(self,data,labels,layers):
        self.data = data
        self.labels = labels
        self.layers = layers
        self.thetas = MultilayerPerception.thetas_init(layers)

    def train(self,max_iterations = 1000, alpha = 0.1):
        unrolled_theta = MultilayerPerception.thetas_unroll(self.thetas)
        MultilayerPerception.gradient_descent(self.data,self.labels,unrolled_theta,self.layers,max_iterations,alpha)

    def thetas_unroll(thetas):
        num_theta_layers = len(thetas)
        unrolled_theta = np.array([])
        for theta_layer_index in range(num_theta_layers):
            np.hstack(unrolled_theta,thetas[theta_layer_index].flatten())
        return unrolled_theta
    @staticmethod
    def theta_roll(unrolled_thetas, layers):
        num_layers = len(layers)
        thetas = {}
        unrolled_shift = 0
        for layer_index in range(num_layers - 1):
            in_count = layers[layer_index]
            out_count = layers[layer_index+1]
            thetas_width = in_count+1
            thetas_height = out_count
            thetas_volume = thetas_height * thetas_width
            start_index = unrolled_shift
            end_index = unrolled_shift + thetas_volume
            layer_theta_unrolled = unrolled_thetas[start_index:end_index]
            thetas[layer_index] = layer_theta_unrolled.reshape((thetas_height,thetas_width))
            unrolled_shift = unrolled_shift + thetas_volume
        return thetas
    @staticmethod
    def thetas_init(layers):
        num_layers = len(layers)
        thetas = {}
        for layer_index in range(num_layers - 1):
            in_count = layers[layer_index]
            out_count = layers[layer_index+1]
            #考虑偏置项
            thetas[layer_index] = np.random.rand(out_count,in_count+1)*0.05
        return thetas

    def gradient_descent(data,labels,unrolled_theta,layers,max_iterations,alpha):
        optimized_theta = unrolled_theta
        cost_history = []

        for i in range(max_iterations):
            cost = MultilayerPerception.cost_function(data,labels,MultilayerPerception.theta_roll(unrolled_theta),layers)

    @staticmethod
    def cost_function(data,labels,thetas,layers):
        num_layers = len(layers)
        num_examples = data.shape[0]
        num_labels = layers[-1]
        #前向传播
        predictions = MultilayerPerception.feedforward_propagation(data,thetas,layers)
        bitwise_labels = np.zeros((num_examples,num_labels))
        for example_index in range(num_examples):
            bitwise_labels[example_index][labels[example_index][0]]

    @staticmethod
    def feedforward_propagation(data,thetas, layers):
        num_layers = len(layers)
        num_examples = data.shape[0]
        in_layer_activation = data

        #逐层计算
        for layer_index in range(num_layers - 1):
            theta = thetas[layer_index]
            out_layer_activation = sigmoid(np.dot(in_layer_activation,theta.T))
            #计算完之后考虑偏置项,为了和下一次迭代相匹配
            out_layer_activation = np.hstack((np.ones((num_examples,1)),out_layer_activation))
            in_layer_activation = out_layer_activation
        #返回输出层结果
        return in_layer_activation[:,1:]




