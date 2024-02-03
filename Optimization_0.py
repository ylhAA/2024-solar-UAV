import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import algorithm_0
import solver


# 这个文件是优化计算的主体
# 输入迭代次数 种群大小 迭代方法就可以进行优化
# 如果要对优化方法的参数进行调整需要在algorithm中进行操作
# 解算过程中的日志控制 中途恢复的尝试也都放在这里
# Optimization 控制文件
def Optimization(iteration, population, method):
    if method == 0:
        PSO_PCA_Optimization(iteration, population)
    else:
        pass
    return 0


# #############################################################
# ################## PSO-PCA-Optimization #####################
# #############################################################
def PSO_PCA_Optimization(iteration, population):
    iterations = 0  # 控制迭代的变量
    rate = 1  # 用来控制更新的距离
    max_range = 0.1  # 扰动范围
    best_evaluate = np.zeros(iteration)
    population_file = "D:\\aircraft design competition\\24solar\\design_model\\whole_wing_optimization\\supervise\\pso_population_record.txt"
    supervise_file = "D:\\aircraft design competition\\24solar\\design_model\\whole_wing_optimization\\supervise\\pso_supervise.txt"
    # 把文件清空
    with open(population_file, 'w') as file:
        pass
    with open(supervise_file, 'w') as file:
        pass
    pop = algorithm_0.generate(population, max_range)
    pre = pop + (np.random.rand(population, 30) * max_range * 0.2 - max_range * 0.1)  # 只是为了处理方便加入的扰动速度
    # 先做第一次计算
    current_time = datetime.now()
    # 写入计算开始时间并记录当前最佳个体
    with open(supervise_file, 'a') as file:
        file.write(f"解算开始：\n iteration : {iterations}\n{current_time}\n")
    data_list = algorithm_0.data_package(population, pop)
    evaluate = solver.solve(data_list, 4)
    best_ID = np.argmax(evaluate)
    best_evaluate[iterations] = evaluate[best_ID]  # 最佳评估值记录
    # 只是因为代码太长了 所以分开为两次write
    with open(supervise_file, 'a') as file:
        file.write(f"最佳个体: 评估值{evaluate[best_ID]}\nroot_up:{data_list[best_ID]['root_up']}\nroot_low:{data_list[best_ID]['root_low']}\n")
        file.write(f"tip_up:{data_list[best_ID]['tip_up']}\ntip_low:{data_list[best_ID]['tip_low']}")
    # 第一次储存种群
    with open(population_file, 'a') as file:
        file.write(f"Iterations: {iterations}\npopulation:\n{pop}\nevaluation\n{evaluate}")
    for iterations in range(iteration):
        # 粒子群算法更新 搜索率线性递减
        data_list, pop_new = algorithm_0.PSO_PCA(pop, pre, evaluate, population, rate * (1 - iterations / iteration))
        # 种群的重新赋值
        pre = pop
        pop = pop_new
        # 进行下一次计算
        current_time = datetime.now()
        with open(supervise_file, 'a') as file:
            file.write(f"本次解算开始时间：\n iterations : {iterations + 1}\n{current_time}\n")
        data_list = algorithm_0.data_package(population, pop)
        evaluate = solver.solve(data_list, 4)
        best_ID = np.argmax(evaluate)
        best_evaluate[iterations] = evaluate[best_ID]  # 最佳评估值记录
        with open(supervise_file, 'a') as file:
            file.write(f"最佳个体: 评估值{evaluate[best_ID]}\nroot_up:{data_list[best_ID]['root_up']}\nroot_low:{data_list[best_ID]['root_low']}\n")
            file.write(f"tip_up:{data_list[best_ID]['tip_up']}\ntip_low:{data_list[best_ID]['tip_low']}")
        # 中间次数的保存
        with open(population_file, 'a') as file:
            file.write(f"Iterations: {iterations + 1}\npopulation:\n{pop}\nevaluation\n{evaluate}")
    # 后处理画图
    index = np.arange(0, iteration)
    plt.figure()
    plt.scatter(index, best_evaluate)
    plt.title('Optimization Trend')
    plt.xlabel('Iterations')
    plt.ylabel('Evaluate')
    plt.show()
    print(index)
    print(best_evaluate)
    return 0


# #############################################################
# #################### 简单的demo与测试 #########################
# #############################################################
PSO_PCA_Optimization(30, 20)
