import numpy as np
import torch
import paths
import agent_based_modeling_1
from agent_based_modeling_1 import SimpleNeuralNetwork
import Gemo_Generate
import vsp_4


# 配平点的升力系数
def cl_determine(cmy, cl):
    alpha0 = -5
    alpha1 = 5
    alpha = alpha0 - (alpha1 - alpha0) * cmy[0] / (cmy[1] - cmy[0])
    cl = cl[0] + (alpha - alpha0) * (cl[1] - cl[0]) / (alpha1 - alpha0)
    return cl


# 满足升力系数的力矩系数
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


# 满足设计配平工况，升力系数不达标惩罚~~其实大了不需要加约束，惩罚量1e-1
def constraint(cl_NN, cmy_NN, input_data, input_mean, input_std, mean1, std1, mean2, std2):
    cmy = predict_and_scale(cmy_NN, input_data, input_mean, input_std, mean1, std1)
    cl = predict_and_scale(cl_NN, input_data, input_mean, input_std, mean2, std2)
    alpha = -10 * cmy[0] / (cmy[1] - cmy[0]) - 5
    if alpha > 10 or alpha < -5:
        value = 1e3
        with open(paths.supervise_file, 'a') as f_supervise:
            f_supervise.write(f"配平攻角{alpha} 不合理攻角\n")
    else:
        value = cl_determine(cmy, cl)
        if value <= 0.32:  # 设计升力系数
            value = 10 * (value - 0.32) ** 2
        else:
            pass
    return value


# 优化目标诱导阻力系数要求尽可能的小
def objective(input_data, cdi_NN, input_mean, input_std, mean, std):
    cdi = predict_and_scale(cdi_NN, input_data, input_mean, input_std, mean, std)
    return cdi


class Config:
    def __init__(self, input_mean, input_std, mean1, std1, mean2, std2, mean, std):
        self.input_mean = input_mean
        self.input_std = input_std
        self.mean1 = mean1
        self.std1 = std1
        self.mean2 = mean2
        self.std2 = std2
        self.mean = mean
        self.std = std


def evaluate(input_data, cmy_NN, cl_NN, cdi_NN, con_fig):
    # 计算约束和目标
    constraint_value = constraint(cl_NN, cmy_NN, input_data, con_fig.input_mean, con_fig.input_std, con_fig.mean1,
                                  con_fig.std1, con_fig.mean2, con_fig.std2)
    objective_value = objective(input_data, cdi_NN, con_fig.input_mean, con_fig.input_std, con_fig.mean, con_fig.std)

    value = objective_value + constraint_value
    return value


# #############################################################
# #########################  GWO  #############################
# #############################################################
# 灰狼算法 主成分分析
def GWO_0(pop, eva, iteration, iterations):
    # 参数设置
    population = len(pop)  # 种群数目
    ID = find_leading_wolf(eva)
    # 优化更新
    for i in range(population):
        if i != ID[0] and i != ID[1] and i != ID[2]:
            temp = np.zeros(len(pop[i]))
            a = 2 - 2 * (iterations / iteration)
            for _ in range(3):
                C = 2 * np.random.rand()
                A = a * (2 * np.random.rand() - 1)
                temp += pop[ID[_]] - A * (C * pop[ID[_]] - pop[i])
            pop[i] = temp / 3
    # 返回两个量,一个是用于计算的字典组 另一个是更新的扰动种群
    return pop, ID


# 找到最小的三个值的索引
def find_leading_wolf(arr):
    indexed_arr = list(enumerate(arr))
    sorted_arr = sorted(indexed_arr, key=lambda x: x[1], reverse=False)
    index = [index for index, _ in sorted_arr[:3]]
    return index


# 边界的检查 bound 中第一列是下限第二列是上限
def boundary_check(arr, bound):
    for i in range(len(arr)):
        for j in range(len(arr[0])):
            if arr[i][j] < bound[j][0]:
                arr[i][j] = bound[j][0]
            if arr[i][j] > bound[j][1]:
                arr[i][j] = bound[j][1]
    return arr


def optimize():
    # # 网格初始化训练
    in_mean, in_std, lab1_mean, lab1_std, lab2_mean, lab2_std, lab3_mean, lab3_std = agent_based_modeling_1.train()
    # 创建 Config 对象
    config = Config(in_mean, in_std, lab1_mean, lab1_std, lab2_mean, lab2_std, lab3_mean, lab3_std)
    # 网络加载
    cmy_predict = SimpleNeuralNetwork(input_size=15, hidden_size1=80, hidden_size2=30, output_size=2)
    cl_predict = SimpleNeuralNetwork(input_size=15, hidden_size1=80, hidden_size2=30, output_size=2)
    cdi_predict = SimpleNeuralNetwork(input_size=15, hidden_size1=50, hidden_size2=15, output_size=1)

    cmy_predict.load_state_dict(torch.load(paths.cmy_NN))
    cl_predict.load_state_dict(torch.load(paths.cl_NN))
    cdi_predict.load_state_dict(torch.load(paths.cdi_NN))

    boundary = [
        [0.2, 0.4],
        [0.2, 0.4],
        [0.2, 0.4],
        [0.2, 0.4],
        [-0.4, 0.4],
        [-0.4, 0.4],
        [-0.4, 0.4],
        [0.2, 0.4],
        [0.2, 0.4],
        [0.2, 0.4],
        [0.2, 0.4],
        [-0.4, 0.4],
        [-0.4, 0.4],
        [-0.4, 0.4],
        [0, 1]
    ]

    # 生成30行7列的数组，每列的值在对应的边界范围内
    max_population = 500
    iteration = 100
    people = np.zeros((max_population, 15))  # 初始化一个全零数组
    people_best = np.zeros(15)
    # 遍历每一列，并生成随机值
    for i, (min_val, max_val) in enumerate(boundary):
        people[:, i] = np.random.uniform(low=min_val, high=max_val, size=max_population)
    # print(people)
    # eva = np.zeros(max_population)
    # for i in range(max_population):
    #     eva[i] = evaluate(people[i], cmy_predict, cl_predict , cdi_predict, config)
    # print(eva)
    # # 调用神经网络进行评估
    eva = np.zeros(max_population)
    for iterations in range(iteration):
        people = boundary_check(people, boundary)
        for i in range(max_population):
            eva[i] = evaluate(people[i], cmy_predict, cl_predict, cdi_predict, config)
            # 如果不符合几何条件评估值直接给一个不合理大值
            flag = Gemo_Generate.CST_airfoil_gemo_check(people[i])
            if flag == 1:
                eva[i] = 1e3
            else:
                pass
        print(iterations, eva)
        # 准备更新代理模型
        ID = find_leading_wolf(eva)
        for i in range(len(ID)):
            Gemo_Generate.CST_airfoil_file_generate(people[ID[i]], paths.tip_file, paths.mid_file, paths.root_file)
            vsp_4.create_Geom_3(paths.tip_file, paths.mid_file, paths.root_file)
            # CMy, cl
            x_cg = 0.25859  # 给定设定的重心位置
            # 这个相当于是配平条件的攻角
            a, b = vsp_4.vsp_aero_sweep_1(-5, 5, 2)
            alpha = -10 * a[0] / (a[1] - a[0]) - 5
            # 考虑角度的控制，代理里面也要改过
            if alpha > 10 or alpha < -5:
                pass
            else:
                c = vsp_4.vsp_aero_0(x_cg=x_cg, aoa=alpha)
                print(c)
                print(alpha)
                combined = np.concatenate([people[ID[i]], a, b, c[2]])
                Gemo_Generate.write_data(paths.supervise_file, combined)
                Gemo_Generate.write_data(paths.dictionary_file, combined)

        # # 网格的重新训练
        in_mean, in_std, lab1_mean, lab1_std, lab2_mean, lab2_std, lab3_mean, lab3_std = (agent_based_modeling_1.
                                                                                          train())
        # 创建 Config 对象
        config = Config(in_mean, in_std, lab1_mean, lab1_std, lab2_mean, lab2_std, lab3_mean, lab3_std)
        # 网络加载
        cmy_predict = SimpleNeuralNetwork(input_size=15, hidden_size1=80, hidden_size2=30, output_size=2)
        cl_predict = SimpleNeuralNetwork(input_size=15, hidden_size1=80, hidden_size2=30, output_size=2)
        cdi_predict = SimpleNeuralNetwork(input_size=15, hidden_size1=50, hidden_size2=15, output_size=1)

        cmy_predict.load_state_dict(torch.load(paths.cmy_NN))
        cl_predict.load_state_dict(torch.load(paths.cl_NN))
        cdi_predict.load_state_dict(torch.load(paths.cdi_NN))
        people, ID = GWO_0(people, eva, iterations=iterations, iteration=iteration)
        people_best = people[ID[0]]
        people_best = np.array(people_best)
        print(eva[ID[0]])
    Gemo_Generate.CST_airfoil_file_generate(people_best, paths.tip_file, paths.mid_file, paths.root_file)
    vsp_4.create_Geom_3(paths.tip_file, paths.mid_file, paths.root_file)
    with open(paths.supervise_file, 'a') as file:
        file.write("\n历史最佳个体生成\n")
    return 0


# # #############################################################
# # #################### 简单的demo与测试 #########################
# # #############################################################
x_cg = 0.25859  # 给定设定的重心位置
# 这个相当于是配平条件的攻角
a, b = vsp_4.vsp_aero_sweep_1(-5, 5, 2)
alpha = -10 * a[0] / (a[1] - a[0]) - 5
# 考虑角度的控制，代理里面也要改过
if alpha > 10 or alpha < -5:
    pass
else:
    c = vsp_4.vsp_aero_0(x_cg=x_cg, aoa=alpha)
    print(c)
    print(alpha)
