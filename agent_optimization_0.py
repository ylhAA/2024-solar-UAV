import numpy as np
import torch
import paths
import agent_based_modeling_0
# import train_data_generate
from agent_based_modeling_0 import SimpleNeuralNetwork
from scipy.optimize import minimize


def cl_determine(cmy, cl):
    alpha0 = -5
    alpha1 = 5
    alpha = alpha0 - (alpha1 - alpha0) * cmy[0] / (cmy[1] - cmy[0])
    cl = cl[0] + (alpha - alpha0) * (cl[1] - cl[0]) / (alpha1 - alpha0)
    return cl


def cmy_determine(cmy, cl):
    alpha0 = -5
    alpha1 = 5
    alpha = alpha0 + (alpha1 - alpha0) * (0.32 - cl[0]) / (cl[1] - cl[0])
    cmy = cmy[0] + (alpha - alpha0) * (cmy[1] - cmy[0]) / (alpha1 - alpha0)
    return cmy


# 用来调用神经网络进行预测
def predict_and_scale(network, input_data, input_mean, input_std, mean, std):
    with torch.no_grad():  # 不需要计算梯度，只进行预测
        input_data = (input_data - input_mean) / input_std
        tensor_input = torch.from_numpy(input_data).float()
        output_tensor = network(tensor_input)
        output_numpy = output_tensor.numpy()
        scaled_output = output_numpy * std + mean
    return scaled_output


def constraint(cl_NN, cmy_NN, input_data, input_mean, input_std, mean1, std1, mean2, std2):
    cmy = predict_and_scale(cmy_NN, input_data, input_mean, input_std, mean1, std1)
    cl = predict_and_scale(cl_NN, input_data, input_mean, input_std, mean2, std2)
    value = cmy_determine(cmy, cl)
    return value


def objective(input_data, cdi_NN, input_mean, input_std, mean, std):
    cdi = predict_and_scale(cdi_NN, input_data, input_mean, input_std, mean, std)
    return cdi


# 网络的训练部分
in_mean, in_std, lab1_mean, lab1_std, lab2_mean, lab2_std, lab3_mean, lab3_std = agent_based_modeling_0.model_train()
# # 归一化数据
# in_mean = [9.7625e-04, 4.3476e-04, 6.3180e-04, 6.2379e-05, -4.1467e-04,
#            -1.5718e-04, -3.7592e-04, 3.4132e-04, 5.0596e-01]
# in_std = [0.0234, 0.0229, 0.0230, 0.0229, 0.0231, 0.0230, 0.0232, 0.0231, 0.2891]
# lab1_mean = [0.0459, -0.0317]
# lab1_std = [0.0249, 0.0248]
# lab2_mean = [-0.4085, 0.3789]
# lab2_std = [0.0514, 0.0514]
# lab3_mean = [0.0019]
# lab3_std = [0.0024]

# 网络加载
cmy_predict = SimpleNeuralNetwork(input_size=9, hidden_size1=80, hidden_size2=30, output_size=2)
cl_predict = SimpleNeuralNetwork(input_size=9, hidden_size1=80, hidden_size2=30, output_size=2)
cdi_predict = SimpleNeuralNetwork(input_size=9, hidden_size1=50, hidden_size2=15, output_size=1)
cmy_predict.load_state_dict(torch.load(paths.cmy_NN))
cl_predict.load_state_dict(torch.load(paths.cl_NN))
cdi_predict.load_state_dict(torch.load(paths.cdi_NN))
# # #############################################################
# # #################### 简单的demo与测试 #########################
# # #############################################################
# # 一个计算值的尝试
# # test_x = np.array([0.02, 0.03, 0.03, -0.03, 0.02, 0.04, 0.04, -0.02, 0.2])
# # test_x_tensor = torch.from_numpy(test_x).float()
# # with torch.no_grad():  # 不需要计算梯度，因为只是进行预测
# #     cmy_tensor = cmy_predict(test_x_tensor)
# #     cl_tensor = cl_predict(test_x_tensor)
# #     cdi_tensor = cdi_predict(test_x_tensor)
# #     cmy_predictions = cmy_tensor.numpy()
# #     cl_predictions = cl_tensor.numpy()
# #     cdi_predictions = cdi_tensor.numpy()
# #     cmy_predictions = cmy_predictions * lab1_std + lab1_mean
# #     cl_predictions = cl_predictions * lab2_std + lab2_mean
# #     cdi_predictions = cdi_predictions * lab3_std + lab3_mean
# # print(cmy_predictions, cl_predictions, cdi_predictions)
# # print(cl_determine(cmy_predictions, cl_predictions))
# # print(cmy_determine(cmy_predictions, cl_predictions))
# # test_x = np.array([0.02, 0, 0.03, -0.01, 0.01, 0.03, 0.03, -0.01, 0.5])
# # test_x = np.zeros(9)
# # print(restriction(cl_predict, cmy_predict, test_x, in_mean, in_std, lab1_mean, lab1_std, lab2_mean, lab2_std))
# # tep = train_data_generate.read_data(paths.dictionary_file)
# # test_pop = tep[:, :9]
# # test_cmy = tep[:, [9, 10]]
# # test_cl = tep[:, [11, 12]]
# # test_cdi = tep[:, [13]]
# # for i in range(1000):
# #     print(f"cmy label {test_cmy[i]} predict {predict_and_scale(cmy_predict, test_pop[i], in_mean, in_std,
# #                                                                lab1_mean, lab1_std)} ")
# # for i in range(1000):
# #     print(f"cl label {test_cl[i]} predict {predict_and_scale(cl_predict, test_pop[i], in_mean, in_std,
# #                                                              lab2_mean, lab2_std)} ")
# # for i in range(1000):
# #     print(f"cdi label {test_cdi[i]} predict {predict_and_scale(cdi_predict, test_pop[i], in_mean, in_std,
# #                                                                lab3_mean, lab3_std)} ")

# 优化模型
# 定义决策变量的边界
lower_bounds = [-0.04] * 8 + [0]  # 前8个是-0.04，最后一个是0
upper_bounds = [0.04] * 8 + [1]  # 前8个是0.04，最后一个是1
# 创建边界元组列表
bounds = tuple((np.array([lb]), np.array([ub])) for lb, ub in zip(lower_bounds, upper_bounds))
# 初始化决策变量
guess = np.random.uniform(low=np.min(lower_bounds), high=np.max(upper_bounds),
                          size=len(lower_bounds))

# 约束条件设置为等于0
con = {'type': 'eq', 'fun': lambda x: constraint(cl_predict, cmy_predict, x, in_mean, in_std, lab1_mean, lab1_std,
                                                 lab2_mean, lab2_std)}

# 调用SciPy的minimize函数
res = minimize(objective, guess, method='SLSQP', bounds=bounds, constraints=con,
               args=(cdi_predict, in_mean, in_std, lab3_mean, lab3_std))
# 输出结果
print("Solution: ", res.x)
print("Objective value at solution: ", res.fun)
