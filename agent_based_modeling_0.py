import numpy as np
import paths
import train_data_generate
import file_paths_generate
import torch
import torch.nn as nn
import torch.optim as optim

"""
这个只是一个初步的尝试，用的是标准差归一化方法
"""


def random_train_test(data, n_tr):
    # 在行数里里面选择给定数量个标号，不放回（重复）没有比例偏置
    train_index = np.random.choice(data.shape[0], size=n_tr, replace=False, p=None)
    train_data = data[train_index]
    # 被选出的是训练集剩下的是验证集
    test_index = np.delete(np.arange(data.shape[0]), train_index)  # 删除train_index对应索引的行数
    test_data = data[test_index]
    return train_data, test_data


# 创建一个简单的神经网络类
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


def model_train():
    # 初步的数据生成
    file_paths_generate.check_and_generate_path()
    temp = train_data_generate.read_data(paths.dictionary_file)
    train_temp, test_temp = random_train_test(temp, n_tr=1600)
    train_pop_arr = train_temp[:, :9]
    train_cmy_arr = train_temp[:, [9, 10]]
    train_cl_arr = train_temp[:, [11, 12]]
    train_cdi_arr = train_temp[:, [13]]
    test_pop_arr = test_temp[:, :9]
    test_cmy_arr = test_temp[:, [9, 10]]
    test_cl_arr = test_temp[:, [11, 12]]
    test_cdi_arr = test_temp[:, [13]]
    # #############################################################
    # ######################## CMy 代理模型#########################
    # #############################################################
    # 将numpy数组转换为PyTorch张量
    X_tensor = torch.from_numpy(train_pop_arr).float()
    y_tensor = torch.from_numpy(train_cmy_arr).float()
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
    # 将测试数据转换为PyTorch张量
    test_X_tensor = torch.from_numpy(test_pop_arr).float()
    test_y_tensor = torch.from_numpy(test_cmy_arr).float()
    X_test_normalized = (test_X_tensor - input_mean) / input_std
    # 将模型设置为评估模式
    model.eval()
    # 用于存储预测结果的列表
    test_predictions = []
    # 迭代测试数据
    with torch.no_grad():  # 不需要计算梯度，也不进行反向传播
        for test_x, test_y in zip(X_test_normalized, test_y_tensor):
            # 进行预测
            test_output = model(test_x.unsqueeze(0))  # 假设模型期望batch_size=1
            # 将预测结果添加到列表中
            test_predictions.append(test_output.numpy())

        # 将预测结果列表转换为numpy数组
    test_predictions = np.concatenate(test_predictions, axis=0)
    predicted = test_predictions * label1_std.numpy() + label1_mean.numpy()
    # 计算测试损失
    test_loss = criterion(torch.from_numpy(predicted), test_y_tensor)
    print(f'CMy Test Loss: {test_loss.item()}')

    # 保存模型
    torch.save(model.state_dict(), paths.cmy_NN)
    # # 加载模型
    # model.load_state_dict(torch.load('.pth'))

    # #############################################################
    # ######################## Cl 代理模型##########################
    # #############################################################
    # 将numpy数组转换为PyTorch张量
    X_tensor = torch.from_numpy(train_pop_arr).float()
    y_tensor = torch.from_numpy(train_cl_arr).float()
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
    # 将测试数据转换为PyTorch张量
    test_X_tensor = torch.from_numpy(test_pop_arr).float()
    test_y_tensor = torch.from_numpy(test_cl_arr).float()
    X_test_normalized = (test_X_tensor - input_mean) / input_std
    # 将模型设置为评估模式
    model.eval()
    # 用于存储预测结果的列表
    test_predictions = []
    # 迭代测试数据
    with torch.no_grad():  # 不需要计算梯度，也不进行反向传播
        for test_x, test_y in zip(X_test_normalized, test_y_tensor):
            # 进行预测
            test_output = model(test_x.unsqueeze(0))  # 假设模型期望batch_size=1
            # 将预测结果添加到列表中
            test_predictions.append(test_output.numpy())

        # 将预测结果列表转换为numpy数组
    test_predictions = np.concatenate(test_predictions, axis=0)
    predicted = test_predictions * label2_std.numpy() + label2_mean.numpy()
    # 计算测试损失
    test_loss = criterion(torch.from_numpy(predicted), test_y_tensor)
    print(f'Cl Test Loss: {test_loss.item()}')
    # 保存模型
    torch.save(model.state_dict(), paths.cl_NN)

    # # 加载模型
    # model.load_state_dict(torch.load('.pth'))
    # #############################################################
    # ######################## CDi 代理模型#########################
    # #############################################################
    # 将numpy数组转换为PyTorch张量
    X_tensor = torch.from_numpy(train_pop_arr).float()
    y_tensor = torch.from_numpy(train_cdi_arr).float()
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
    # 将测试数据转换为PyTorch张量
    test_X_tensor = torch.from_numpy(test_pop_arr).float()
    test_y_tensor = torch.from_numpy(test_cdi_arr).float()
    X_test_normalized = (test_X_tensor - input_mean) / input_std
    # 将模型设置为评估模式
    model.eval()
    # 用于存储预测结果的列表
    test_predictions = []
    # 迭代测试数据
    with torch.no_grad():  # 不需要计算梯度，也不进行反向传播
        for test_x, test_y in zip(X_test_normalized, test_y_tensor):
            # 进行预测
            test_output = model(test_x.unsqueeze(0))  # 假设模型期望batch_size=1
            # 将预测结果添加到列表中
            test_predictions.append(test_output.numpy())

        # 将预测结果列表转换为numpy数组
    test_predictions = np.concatenate(test_predictions, axis=0)
    predicted = test_predictions * label3_std.numpy() + label3_mean.numpy()
    # 计算测试损失
    test_loss = criterion(torch.from_numpy(predicted), test_y_tensor)
    print(f'CDi Test Loss: {test_loss.item()}')
    # 保存模型
    torch.save(model.state_dict(), paths.cdi_NN)

    return (input_mean.numpy(), input_std.numpy(), label1_mean.numpy(), label1_std.numpy(),
            label2_mean.numpy(), label2_std.numpy(), label3_mean.numpy(), label3_std.numpy())


# def Cmy_predict(x):
#     return 0


# # 进行生成
# in_mean, in_std, lab1_mean, lab1_std, lab2_mean, lab2_std, lab3_mean, lab3_std = model_train()
# cmy_predict = SimpleNeuralNetwork(input_size=9, hidden_size1=80, hidden_size2=30, output_size=2)
# cl_predict = SimpleNeuralNetwork(input_size=9, hidden_size1=80, hidden_size2=30, output_size=2)
# cdi_predict = SimpleNeuralNetwork(input_size=9, hidden_size1=50, hidden_size2=15, output_size=1)
#
# cmy_predict.load_state_dict(torch.load(paths.cmy_NN))
# cl_predict.load_state_dict(torch.load(paths.cl_NN))
# cdi_predict.load_state_dict(torch.load(paths.cdi_NN))
#
# tep = train_data_generate.read_data(paths.dictionary_file)
# test_pop = tep[:, :9]
# test_lab = tep[:, [9, 10]]
# test_cl = tep[:, [11, 12]]
# test_cdi = tep[:, [13]]
# test_pop = (test_pop - in_mean) / in_std
# # 设置模型为评估模式
# cl_predict.eval()
# cmy_predict.eval()
# cdi_predict.eval()
# # 将输入特征转换为张量
# test_inputs = torch.tensor(test_pop, dtype=torch.float32)
# # 使用加载的模型进行预测 cmy
# with torch.no_grad():
#     predictions = cmy_predict(test_inputs)
# # 将预测结果转换为NumPy数组
# predictions = predictions.numpy()
# predictions = predictions * lab1_std + lab1_mean
# for num in range(len(predictions)):
#     print(predictions[num][0] - test_lab[num][0])
#     print(predictions[num][1] - test_lab[num][1])
# print("\n\n\n")
#
# # 加载模型进行预测 cl
# with torch.no_grad():
#     predictions = cl_predict(test_inputs)
# predictions = predictions.numpy()
# predictions = predictions * lab2_std + lab2_mean
# for num in range(len(predictions)):
#     print(predictions[num][0] - test_cl[num][0])
#     print(predictions[num][1] - test_cl[num][1])
#
# # 加载模型进行预测 cdi
# with torch.no_grad():
#     predictions = cdi_predict(test_inputs)
# predictions = predictions.numpy()
# predictions = predictions * lab3_std + lab3_mean
# for num in range(len(predictions)):
#     print(predictions[num][0], test_cdi[num][0])
