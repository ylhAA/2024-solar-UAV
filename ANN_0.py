import numpy as np
import paths
from sklearn import preprocessing

"""
神经网络的尝试，总体思路如下：
基于BP-NN demo进行
1 基本文件的读写与格式确定
2 公式推导与代码实现
3 数据生成模块（读写测试后进行）
4 训练与测试集测试
5 带入全系统进行验证
"""


# 给定一个文件路径，输入一维数组用于写入文件数据
def write_data(dir_path, data):
    # 将数据转换为NumPy数组（如果它还不是）
    data = np.array(data)
    data_str = ' '.join(map(str, data)) + '\n'
    with open(dir_path, 'a') as fdata:
        fdata.write(data_str)
    return 0


# 给定一个文件路径，读取文件数据
def read_data(dir_path):
    data_temp = []
    with open(dir_path, 'r') as fdata:
        while True:
            line = fdata.readline()
            if not line:
                break
            data_temp.append([float(i) for i in line.split()])
    return np.array(data_temp)


# 用于在读取数据后来读取训练集和数据集
def random_train_test(data, n_tr):
    # 在行数里里面选择给定数量个标号，不放回（重复）没有比例偏置
    train_index = np.random.choice(data.shape[0], size=n_tr, replace=False, p=None)
    train_data = data[train_index]
    # 被选出的是训练集剩下的是验证集
    test_index = np.delete(np.arange(data.shape[0]), train_index)  # 删除train_index对应索引的行数
    test_data = data[test_index]
    return train_data, test_data


# 用于进行标准化处理 便于机器处理
def min_max_normalization(np_array):
    min_max_scaler = preprocessing.MinMaxScaler()
    ret = min_max_scaler.fit_transform(np_array)
    return ret


# 生成在某一范围内的随机数
def rand(min_val, max_val):
    return min_val + (max_val - min_val) * np.random.rand()


# 改进的生成全零矩阵的函数
def make_matrix_np(m, n, fill=0.0):
    return np.full((m, n), fill)


# relu激活函数的改版 解决死亡神经元的问题，超参数alpha预先设置
def leaky_relu(x, alpha=0.01):
    return max(alpha * x, x)


def leaky_relu_derivative(x, alpha=0.01):
    return alpha if x < 0 else 1


class BPNeuralNetwork:
    def __init__(self):  # 设置在BP神经网络中用到的参数
        self.input_n = 0  # 输入层个数
        self.hidden_n = 0  # 隐藏层个数
        self.output_n = 0  # 输出层个数
        self.input_values = []  # [1.0] * self.input_n
        self.hidden_values = []  # [1.0] * self.hidden_n
        self.output_values = []  # [1.0] * self.output_n
        self.input_weights = []  # 两个权重矩阵
        self.output_weights = []
        self.input_correction = []  # dw1
        self.output_correction = []  # dw2
        self.input_bias = []  # 对应输入层和输出层的偏置量
        self.output_bias = []

    def setup(self, ni, nh, no):  # 参数设置
        self.input_n = ni
        self.hidden_n = nh
        self.output_n = no
        # init
        self.input_values = [1.0] * self.input_n  # 输入层神经元输出（输入特征）
        self.hidden_values = [1.0] * self.hidden_n  # 中间层神经元输出
        self.output_values = [1.0] * self.output_n  # 隐藏层神经元输出（预测结果）
        self.input_weights = make_matrix_np(self.input_n, self.hidden_n)
        self.output_weights = make_matrix_np(self.hidden_n, self.output_n)
        # 初始随机赋值，在范围[-1, +1]内
        for i in range(self.input_n):
            for h in range(self.hidden_n):
                self.input_weights[i][h] = rand(-1, 1)
        for h in range(self.hidden_n):
            for o in range(self.output_n):
                self.output_weights[h][o] = rand(-1, 1)
        self.input_correction = make_matrix_np(self.input_n, self.hidden_n)
        self.output_correction = make_matrix_np(self.hidden_n, self.output_n)
        self.input_bias = [0.0] * self.hidden_n
        self.output_bias = [0.0] * self.output_n

        # 可以用于对输入值的预测，在训练阶段是训练流程的一个部分

    def predict(self, inputs):  # 前向传播（在train中套在反向传播的train前面）
        # 输入层计算
        for i in range(self.input_n):
            self.input_values[i] = inputs[i]
        # 隐藏层计算
        # j循环隐藏层神经元 i循环输入神经元
        for j in range(self.hidden_n):
            total = 0.0
            for i in range(self.input_n):
                total += self.input_values[i] * self.input_weights[i][j]
            self.hidden_values[j] = leaky_relu(total + self.input_bias[j])
        # 输出层计算
        for k in range(self.output_n):
            total = 0.0
            for j in range(self.hidden_n):
                total += self.hidden_values[j] * self.output_weights[j][k]
            self.output_values[k] = leaky_relu(total + self.output_bias[k])
        return self.output_values[:]

    def back_propagate(self, case, label, learn, correct):
        # 前向预测 根据当前的权重值与偏置项的更新
        self.predict(case)

        # 计算输出层的误差 w2
        output_deltas = [0.0] * self.output_n
        for o in range(self.output_n):
            error = label[o] - self.output_values[o]
            output_deltas[o] = leaky_relu_derivative(self.output_values[o]) * error

        # 计算隐藏层的误差 w1
        hidden_deltas = [0.0] * self.hidden_n
        for h in range(self.hidden_n):
            error = 0.0
            for o in range(self.output_n):
                error += output_deltas[o] * self.output_weights[h][o]
            hidden_deltas[h] = leaky_relu_derivative(self.hidden_values[h]) * error

        # 更新隐藏-输出层权重 b2
        for h in range(self.hidden_n):
            for o in range(self.output_n):
                change = output_deltas[o] * self.hidden_values[h]
                self.output_weights[h][o] += learn * change + correct * self.output_correction[h][o]
                self.output_correction[h][o] = change
                self.output_bias[o] += learn * change

        # 更新输入-隐藏层权重 b1
        for i in range(self.input_n):
            for h in range(self.hidden_n):
                change = hidden_deltas[h] * self.input_values[i]
                self.input_weights[i][h] += learn * change + correct * self.input_correction[i][h]
                self.input_correction[i][h] = change
                self.input_bias[h] += learn * change

        # 计算样本的均方误差
        error = 0.0
        for o in range(len(label)):
            error += 0.5 * (label[o] - self.output_values[o]) ** 2
        return error

    def train(self, datas, labels, epochs=5000, learn=0.005, correct=0.01, stop_error=0.001):
        for j in range(epochs):
            error = 0.0
            for i in range(len(datas)):
                label = labels[i]
                data = datas[i]
                error += self.back_propagate(data, label, learn, correct)
            if error <= stop_error:
                return j + 1
            print(j, error)
        return epochs


# #############################################################
# #################### 简单的demo与测试 #########################
# #############################################################
# 用于检验写入和读出
# path = paths.dictionary_file
# data_to_write = np.random.rand(10, 10)
# for _ in range(10):
#     write_data(path, data_to_write[_])
# value = read_data(path)
# 数据集的分配
# train, test = random_train_test(value, 7)
# print(train)
# print(test)
# test = np.random.rand(5, 5)
# print(min_max_normalization(test))

# a = leaky_relu(-0.1)
# print(a)

# 数据的读入与处理
path = paths.dictionary_file
temp_data = read_data(path)
train, test = random_train_test(temp_data, 800)
train_pop = train[:, :30]
train_label = train[:, 30:32]
train_label = train_label
test_pop = test[:, :30]
test_label = test[:, 30:32]
test_label = test_label
print("train_pop shape:", train_pop.shape)
print("train_label shape:", train_label.shape)
# # 网络的初始设置
# network = BPNeuralNetwork()
# network.setup(30, 120, 2)
# network.train(train_pop, train_label, 400, 6e-4, 0.21, 0.001)
#
# # 完成文件写入
# for _ in range(len(network.input_weights)):
#     write_data(paths.input_weight_file, network.input_weights[_])
# for _ in range(len(network.output_weights)):
#     write_data(paths.output_weight_file, network.output_weights[_])
# print(network.input_bias, network.output_bias)
# write_data(paths.input_bias_file, network.input_bias)
# write_data(paths.output_bias_file, network.output_bias)

# 开始进行权重的读取和处理
network = BPNeuralNetwork()
network.setup(30, 120, 2)
network.output_bias = read_data(paths.output_bias_file).flatten()
network.input_bias = read_data(paths.input_bias_file).flatten()
network.output_weights = read_data(paths.output_weight_file)
network.input_weights = read_data(paths.input_weight_file)
# print("output_bias shape:", network.output_bias)
# print("input_bias shape:", network.input_bias)
# print("output_weight shape:", network.output_weights)
# print("input_weight shape:", network.input_weights)

error1 = 0
error2 = 0
for _ in range(len(test_pop)):
    xcg, aoa = network.predict(test_pop[_])
    if xcg != 0:
        error1 += np.fabs(xcg - test_label[_][0]) / xcg
    if aoa != 0:
        error2 += np.fabs(aoa - test_label[_][1]) / aoa
print("xcg relevant error:", error1/len(test_pop))
print("aoa relevant error:", error2/len(test_pop))
