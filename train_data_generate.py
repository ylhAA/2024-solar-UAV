import numpy as np
import paths
import ANN_0
import balancing_3
import CST_Generate

# 用于生成基本数据
max_population = 1000
population = 0
generate_pop = 0
while population <= max_population:
    # 生成在范围内的扰动向量
    temp0 = np.random.rand(30)
    pop = (temp0 - 0.5) * 0.2
    temp1, iteration = CST_Generate.Geom_Generate(pop, paths.tip_file, paths.root_file)
    if 0 < iteration <= 10:
        pop = temp1
    elif iteration > 10:
        continue
    else:
        pass
    if iteration <= 10:
        xcg, aoa = balancing_3.balance(paths.tip_file, paths.root_file)
        temp = np.array([xcg, aoa])
        combine = np.append(pop, temp)
        ANN_0.write_data(paths.dictionary_file, combine)
        population += 1
    else:
        pass

# #############################################################
# #################### 简单的demo与测试 #########################
# #############################################################
# for i in range(10):
#     pop = np.random.rand(30)
#     xcg = np.random.rand()
#     alpha = np.random.rand()
#     temp = np.array([xcg, alpha])
#     combine = np.append(pop, temp)
#     ANN_0.write_data(paths.dictionary_file, combine)

# 用来把读取的数据切分成输入和输出值
# data = ANN_0.read_data(paths.dictionary_file)
# first_30 = data[:, :30]
# # Extract the 31st and 32nd elements from each row
# last_2 = data[:, 30:32]
# print("First 30 numbers from each row:")
# print(first_30)
# print("31st and 32nd elements from each row:")
# print(last_2)
