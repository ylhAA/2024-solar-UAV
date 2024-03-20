import numpy as np
import paths
import train_data_generate
import torch
import torch.nn as nn
import torch.optim as optim

"""
尝试使用Kriging模型进行预测
"""


def random_train_test(data, n_tr):
    # 在行数里里面选择给定数量个标号，不放回（重复）没有比例偏置
    train_index = np.random.choice(data.shape[0], size=n_tr, replace=False, p=None)
    train_data = data[train_index]
    # 被选出的是训练集剩下的是验证集
    test_index = np.delete(np.arange(data.shape[0]), train_index)  # 删除train_index对应索引的行数
    test_data = data[test_index]
    return train_data, test_data


class SimpleNeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(SimpleNeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size2, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        return out


# 进行代理模型的训练
def train():
    temp = train_data_generate.read_data(paths.dictionary_file)
    print(np.shape(temp))
    train_para = temp[:, :15]
    train_cmy = temp[:, [15, 16]]
    train_cl = temp[:, [17, 18]]
    train_cdi = temp[:, [19]]
    print("train:", train_para[0], train_cmy[0], train_cl[0], train_cdi[0])
    # #############################################################
    # ######################## CMy 代理模型#########################
    # #############################################################
    # 将numpy数组转换为PyTorch张量
    X_tensor = torch.from_numpy(train_para).float()
    y_tensor = torch.from_numpy(train_cmy).float()
    # 给出标准差 平均值，归一化
    input_mean = X_tensor.mean(dim=0)
    input_std = X_tensor.std(dim=0)
    print(f"input_mean = {input_mean}, input_std = {input_std}\n")
    X_normalized = (X_tensor - input_mean) / input_std
    label1_mean = y_tensor.mean(dim=0)
    label1_std = y_tensor.std(dim=0)
    print(f"cmy_label_mean = {label1_mean}, cmy_label_std = {label1_std}\n")
    y_normalized = (y_tensor - label1_mean) / label1_std
    # 初始化网络
    input_scale = X_tensor.shape[1]
    hidden_scale1 = 80  # 第一个隐藏层的大小
    hidden_scale2 = 30  # 第二个隐藏层的大小
    output_scale = y_tensor.shape[1]
    model = SimpleNeuralNetwork(input_scale, hidden_scale1, hidden_scale2, output_scale)
    # 定义损失函数和优化器
    criterion = nn.MSELoss()  # 使用均方误差损失
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # 使用Adam优化器
    # 训练网络
    num_epochs = 2000  # 训练轮数
    for epoch in range(num_epochs):
        # 前向传播
        outputs = model(X_normalized)
        loss = criterion(outputs, y_normalized)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}')
        # 训练完成后，模型就准备好了，可以用于预测
    # 保存模型
    torch.save(model.state_dict(), paths.cmy_NN)

    # #############################################################
    # ######################## Cl 代理模型##########################
    # #############################################################
    # 将numpy数组转换为PyTorch张量
    X_tensor = torch.from_numpy(train_para).float()
    y_tensor = torch.from_numpy(train_cl).float()
    # 给出标准差 平均值，归一化
    input_mean = X_tensor.mean(dim=0)
    input_std = X_tensor.std(dim=0)
    print(f"input_mean = {input_mean}, input_std = {input_std}\n")
    X_normalized = (X_tensor - input_mean) / input_std
    label2_mean = y_tensor.mean(dim=0)
    label2_std = y_tensor.std(dim=0)
    print(f"cmy_label_mean = {label2_mean}, cmy_label_std = {label2_std}\n")
    y_normalized = (y_tensor - label2_mean) / label2_std
    # 初始化网络
    input_scale = X_tensor.shape[1]
    hidden_scale1 = 80
    hidden_scale2 = 30
    output_scale = y_tensor.shape[1]
    model = SimpleNeuralNetwork(input_scale, hidden_scale1, hidden_scale2, output_scale)
    # 定义损失函数和优化器
    criterion = nn.MSELoss()  # 使用均方误差损失
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # 使用Adam优化器
    # 训练网络
    num_epochs = 2000  # 训练轮数
    for epoch in range(num_epochs):
        # 前向传播
        outputs = model(X_normalized)
        loss = criterion(outputs, y_normalized)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}')
        # 训练完成后，模型就准备好了，可以用于预测
    # 保存模型
    torch.save(model.state_dict(), paths.cl_NN)

    # #############################################################
    # ######################## CDi 代理模型#########################
    # #############################################################
    # 将numpy数组转换为PyTorch张量
    X_tensor = torch.from_numpy(train_para).float()
    y_tensor = torch.from_numpy(train_cdi).float()
    # 给出标准差 平均值，归一化
    input_mean = X_tensor.mean(dim=0)
    input_std = X_tensor.std(dim=0)
    print(f"input_mean = {input_mean}, input_std = {input_std}\n")
    X_normalized = (X_tensor - input_mean) / input_std
    label3_mean = y_tensor.mean(dim=0)
    label3_std = y_tensor.std(dim=0)
    print(f"cmy_label_mean = {label3_mean}, cmy_label_std = {label3_std}\n")
    y_normalized = (y_tensor - label3_mean) / label3_std
    # 初始化网络
    input_scale = X_tensor.shape[1]
    hidden_scale1 = 50
    hidden_scale2 = 15
    output_scale = y_tensor.shape[1]
    model = SimpleNeuralNetwork(input_scale, hidden_scale1, hidden_scale2, output_scale)
    # 定义损失函数和优化器
    criterion = nn.MSELoss()  # 使用均方误差损失
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # 使用Adam优化器
    # 训练网络
    num_epochs = 1500  # 训练轮数
    for epoch in range(num_epochs):
        # 前向传播
        outputs = model(X_normalized)
        loss = criterion(outputs, y_normalized)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}')
    # 训练完成后，模型就准备好了，可以用于预测

    # 保存模型
    torch.save(model.state_dict(), paths.cdi_NN)

    return (input_mean.numpy(), input_std.numpy(), label1_mean.numpy(), label1_std.numpy(),
            label2_mean.numpy(), label2_std.numpy(), label3_mean.numpy(), label3_std.numpy())

