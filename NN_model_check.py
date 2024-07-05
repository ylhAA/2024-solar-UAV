# import numpy as np
# import matplotlib.pyplot as plt
# import paths
# import agent_based_modeling_1
# from agent_based_modeling_1 import SimpleNeuralNetwork
# import train_data_generate
# import torch
# from agent_optimization_3 import predict_and_scale


# class Config:
#     def __init__(self, input_mean, input_std, mean1, std1, mean2, std2, mean, std):
#         self.input_mean = input_mean
#         self.input_std = input_std
#         self.mean1 = mean1
#         self.std1 = std1
#         self.mean2 = mean2
#         self.std2 = std2
#         self.mean = mean
#         self.std = std

#
# check = train_data_generate.read_data(".\\train_data\\check_data.txt")
# check_para = check[:, :16]
# check_cmy = check[:, [16, 17]]
# check_cl = check[:, [18, 19]]
# check_cdi = check[:, [20]]
# # # 网格的重新训练
# in_mean, in_std, lab1_mean, lab1_std, lab2_mean, lab2_std, lab3_mean, lab3_std = (agent_based_modeling_1.
#                                                                                   train())
# # 创建 Config 对象
# config = Config(in_mean, in_std, lab1_mean, lab1_std, lab2_mean, lab2_std, lab3_mean, lab3_std)
# # 网络加载
# cmy_predict = SimpleNeuralNetwork(input_size=16, hidden_size1=80, hidden_size2=30, output_size=2)
# cl_predict = SimpleNeuralNetwork(input_size=16, hidden_size1=80, hidden_size2=30, output_size=2)
# cdi_predict = SimpleNeuralNetwork(input_size=16, hidden_size1=50, hidden_size2=15, output_size=1)
#
# cmy_predict.load_state_dict(torch.load(paths.cmy_NN))
# cl_predict.load_state_dict(torch.load(paths.cl_NN))
# cdi_predict.load_state_dict(torch.load(paths.cdi_NN))
#
# cl_record = np.zeros((len(check), 2))
# cmy_record = np.zeros((len(check), 2))
# cdi_record = np.zeros(len(check))
# cmy2 = 0
# cmy1 = 0
# cl1 = 0
# cl2 = 0
# cdi = 0
# MSE = np.zeros(3)
# MAE = np.zeros(3)
# for i in range(len(check)):
#     cmy_record[i] = predict_and_scale(cmy_predict, check_para[i], config.input_mean, config.input_std, config.mean1, config.std1)
#     cl_record[i] = predict_and_scale(cl_predict, check_para[i], config.input_mean, config.input_std, config.mean2, config.std2)
#     cdi_record[i] = predict_and_scale(cdi_predict, check_para[i], config.input_mean, config.input_std, config.mean, config.std)
#     cmy1 += cmy_record[i][0]
#     cmy2 += cmy_record[i][1]
#     cl1 += cl_record[i][0]
#     cl2 += cl_record[i][1]
#     cdi += cdi_record[i]
# cl1 /= float(len(check))
# cl2 /= float(len(check))
# cmy1 /= float(len(check))
# cmy2 /= float(len(check))
# cdi /= float(len(check))
# for i in range(len(check)):
#     MSE[0] += (cmy_record[i][0] - cmy1) ** 2 + (cmy_record[i][1] - cmy2) ** 2
#     MSE[1] += (cl_record[i][0] - cl1) ** 2 + (cl_record[i][1] - cl2) ** 2
#     MSE[2] += (cdi_record[i] - cdi) ** 2
#     MAE[0] += np.fabs(cmy_record[i][0] - cmy1) + np.fabs(cmy_record[i][1] - cmy2)
#     MAE[1] += np.fabs(cl_record[i][0] - cl1) + np.fabs(cl_record[i][1] - cl2)
#     MAE[2] += np.fabs(cdi_record[i] - cdi)
# MSE[0] /= float(2*len(check))
# MSE[1] /= float(2*len(check))
# MSE[2] /= float(len(check))
# MAE /= float(len(check))
# MAE[0] /= 2.0
# MAE[1] /= 2.0
# print(f"MSE CMy {MSE[0]} MAE{MAE[0]}")
# print(f"MSE CL {MSE[1]} MAE{MAE[1]}")
# print(f"MSE Cdi {MSE[2]} MAE{MAE[2]}")

# 这个是模型精度的原始数据，要注意处理！！
# sample_point = [11, 21, 43, 70, 300]
# Mse_data = [[1.099e-4, 2.793e-4, 1.334e-4, 5.155e-4, 1.0289e-4],
#             [1.491e-3, 1.439e-3, 2.496e-3, 3.726e-3, 1.192e-3],
#             [1.089e-5, 1.849e-5, 6.079e-6, 2.019e-5, 1.845e-6]]
# Mae_data = [[8.945e-3, 1.370e-2, 9.502e-3, 2.024e-2, 7.78e-3],
#             [3.448e-2, 3.089e-2, 4.329e-2, 5.530e-2, 2.639e-2],
#             [2.831e-3, 3.576e-3, 2.10e-3, 4.1808e-3, 1.109e-3]]
# plt.scatter(sample_point, Mse_data[1])
# plt.title("MSE of Cmy")
# plt.xlabel("sample points")
# plt.ylabel("Mse value")
# plt.legend()
# plt.show()

