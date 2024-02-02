import numpy as np
from sklearn.decomposition import PCA


# 这是一个优化方法的尝试
# 导入的是种群数量和扰动最大范围   （注意需要仔细查看）
def generate(population, max_range):
    pop = np.random.rand(population, 30)  # 30是控制参数的数目
    biggest = np.max(pop)
    pop = pop / biggest * max_range
    return pop


# #############################################################
# ################ 扰动数据处理 disturb_modify ##################
# #############################################################
# 输入一个扰动数组,输出用于气动分析的数据 用于优化算法内部和求解器部分的数据协调
def disturb_modify(person):  # 输入一个30维的向量代表一个个体
    # 优化设计基础外形
    root_up = [0.20027, 0.06942, 0.26365, 0.01476, 0.27234, 0.14104, 0.17256, 0.20330]
    root_low = [-0.20027, -0.08095, -0.21768, -0.12756, -0.14138, -0.24818, 0.04641, 0.16772]
    tip_up = [0.15727, 0.04782, 0.19466, -0.10213, 0.22039, -0.03441, 0.06494, 0.06763]
    tip_low = [-0.15727, -0.33244, -0.11350, -0.26489, -0.29933, -0.14698, -0.27400, -0.25044]
    length = len(root_up)
    # 扰动添加 不受到前缘影响
    for i in range(1, length, 1):
        root_up[i] += person[i]
        root_low[i] += person[length - 1 + i]
        tip_up[i] += person[2 * length - 2 + i]
        tip_low[i] += person[3 * length - 2 + i]
    # 受到前缘影响的参数的特殊化处理
    root_up[0] += person[0]
    root_low[0] = -root_up[0]
    tip_up[0] += person[2 * length]
    tip_low[0] = -tip_up[0]

    return root_up, root_low, tip_up, tip_low


# #############################################################
# ############## 数据打包返回求解器 data_package #################
# #############################################################
# 这个函数用来把经过优化算法处理的种群打包成用于求解器处理的格式
def data_package(population, pop):
    # 用来存储数据的字典数组
    data_list = []
    for i in range(population):
        root_up, root_low, tip_up, tip_low = disturb_modify(pop[i])
        data = {
            'root_up': root_up,
            'root_low': root_low,
            'tip_up': tip_up,
            'tip_low': tip_low,
        }
        data_list.append(data)
    return data_list


# #############################################################
# ######################## PSO-PCA  ###########################
# #############################################################
# 粒子群算法-主成分分析 单步更新
def PSO_PCA(pop, pre_pop, evaluate, population, rate):
    # pop种群的扰动完整数据 population 种群个体数目
    # pre_pop 上一次迭代的种群数据 evaluate评估值
    # rate 更新速率
    n_component = 3  # 降阶次数
    inertia_eff = 0.2  # 惯性参数
    guide_eff = 0.5  # 寻优参数
    random_eff = 0.1  # 布朗运动参数
    pca = PCA(n_component)
    pop_pca = pca.fit_transform(pop)
    pre_pca = pca.transform(pre_pop)  # 上一步的粒子位置
    velocity = pop_pca - pre_pca  # 获得速度向量组
    rand = np.random.rand(population, n_component)
    rand = rand - 0.5
    best_ID = np.argmax(evaluate)  # 获得评估值最高的个体ID
    for i in range(population):
        if i != best_ID:
            pop_pca[i] = pop_pca[i] + (pop_pca[best_ID] - pop_pca[i]) * rate * guide_eff + velocity[
                i] * inertia_eff * rate + rand[i] * random_eff * rate
        else:
            pass
    pop_reverse = pca.inverse_transform(pop_pca)
    data = data_package(population, pop_reverse)
    return data


# #############################################################
# ######################## GWO-PCA  ###########################
# #############################################################
# 灰狼算法 主成分分析

# #############################################################
# ######################## GA-PCA  ###########################
# #############################################################
# 遗传算法 主成分分析

# #############################################################
# #################### 简单的demo与测试 #########################
# #############################################################
# # 用于测试数据处理是否正常
# pop = []
# pp = np.ones(30)
# pop.append(pp)
# data_list = data_package(1, pop)
# print(data_list[0]['root_up'])
# print(data_list[0]['root_low'])
# print(data_list[0]['tip_up'])
# print(data_list[0]['tip_low'])

# # 用于测试生成函数
# population = 10
# max_range = 0.1
# pop = generate(population, max_range)
# print(pop)
# print(pop.shape)

# # PCA主成分分析 调用方法
# pop = generate(3, 0.1)
# n_component = 2  # 主成分降阶后次数
# pca = PCA(n_component)
# pop_pca = pca.fit_transform(pop)
# pop_reverse = pca.inverse_transform(pop_pca)
# print(pop_pca)
# print(pop)
# print(pop_reverse)

# # simple PSO单步计算的测试
# pp = generate(3, 0.1)
# pre = generate(3, 0.1)
# evaluate = np.array([1, 2, 3])
# data = PSO_PCA(pp, pre, evaluate, 3, 1)
# print("pp:",pp)
# print ("pre: ",pre)
# print(data)
# 经过测试返回字典数组
