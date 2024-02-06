import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import algorithm_1
import solver_1
import vsp_3


# 这个文件是优化计算的主体
# 输入迭代次数 种群大小 迭代方法就可以进行优化
# 如果要对优化方法的参数进行调整需要在algorithm中进行操作
# 解算过程中的日志控制 中途恢复的尝试也都放在这里
# Optimization 控制文件
def Optimization(iteration, population, method):
    if method == 0:
        PSO_PCA_Optimization(iteration, population)
    elif method == 1:
        GWO_PCA_Optimization(iteration, population)
    elif method == 2:
        GA_Optimization(iteration, population)
    return 0


# #############################################################
# ################## PSO-PCA-Optimization #####################
# #############################################################
def PSO_PCA_Optimization(iteration, population):
    variance = 1e-4  # 用于控制适应度收敛准则的 方均根收敛界限
    iterations = 0  # 控制迭代的变量
    rate = 0.5  # 用来控制初始搜索范围的系数 表示迭代一定比例后开始进行收缩
    max_range = 0.08  # 扰动范围
    solve_mode = 0  # 求解的方式 用于控制求解函数 详细信息在 solver 和 balancing里面
    best_evaluate = np.zeros(iteration)  # 用来存储每次迭代的最优值

    population_file = ("D:\\aircraft design competition\\24solar\\design_model\\"
                       "whole_wing_optimization\\supervise\\pso_population_record.txt")
    supervise_file = ("D:\\aircraft design competition\\24solar\\design_model"
                      "\\whole_wing_optimization\\supervise\\pso_supervise.txt")
    # 把文件清空
    with open(population_file, 'w'):
        pass
    with open(supervise_file, 'w'):
        pass
    pop = algorithm_1.generate(population, max_range)
    pre = pop + (np.random.rand(population, 30) * max_range * 0.2 - max_range * 0.1)  # 只是为了处理方便加入的初始扰动速度
    p_best = pop  # 给个体的最佳位置赋值
    # 先做第一次计算
    current_time = datetime.now()
    # 写入计算开始时间
    with open(supervise_file, 'a') as file:
        file.write(f"解算开始：\n iteration : {iterations}\n{current_time}\n")

    # 进行首次评估分析 并生成迭代PSO所有参数
    data_list = algorithm_1.data_package(population, pop)
    evaluate = solver_1.solve(data_list, solve_mode)
    p_best_evaluate = evaluate  # 存储个体最佳的值 用来更新比较
    best_ID = np.argmax(evaluate)  # 找出当前最佳值的ID
    best_evaluate[iterations] = evaluate[best_ID]  # 当前最佳评估值记录
    history_best = pop[best_ID]  # 历史最佳值初始
    history_best_value = evaluate[best_ID]  # 历史最佳评估值
    # 记录当前最佳个体 只是因为代码太长了 所以分开为两次write
    with open(supervise_file, 'a') as file:
        file.write(f"最佳个体: 评估值{evaluate[best_ID]}\nroot_up:{data_list[best_ID]['root_up']}"
                   f"\nroot_low:{data_list[best_ID]['root_low']}\n")
        file.write(f"tip_up:{data_list[best_ID]['tip_up']}\ntip_low:{data_list[best_ID]['tip_low']}")
    # 第一次储存种群
    with open(population_file, 'a') as file:
        file.write(f"Iterations: {iterations}\npopulation:\n{pop}\nevaluation\n{evaluate}")

    # 开始迭代
    for iterations in range(iteration):
        # 粒子群算法更新 搜索率线性递减 data_list, pop_new = algorithm_1.PSO_PCA_0(pop, pre, p_best, evaluate, population,
        # rate*0.5+rate * np.sqrt((1 - iterations / iteration)),max_range)
        data_list, pop_new = algorithm_1.PSO_PCA_1(pop, pre, p_best, evaluate,
                                                   population, rate, iteration, max_range)
        # 种群的重新赋值
        pre = pop
        pop = pop_new

        # 进行下一次计算
        current_time = datetime.now()
        with open(supervise_file, 'a') as file:
            file.write(f"本次解算开始时间：\n iterations : {iterations + 1}\n{current_time}\n")

        # 计算处理
        data_list = algorithm_1.data_package(population, pop)
        evaluate = solver_1.solve(data_list, solve_mode)

        # 最佳评估值记录
        best_ID = np.argmax(evaluate)
        best_evaluate[iterations] = evaluate[best_ID]

        # 个体最优位置与评估值更新
        for i in range(population):
            if evaluate[i] > p_best_evaluate[i]:
                p_best_evaluate[i] = evaluate[i]
                p_best[i] = pop[i]
            else:
                pass

        # 历史最优值更新
        if evaluate[best_ID] > history_best_value:
            history_best_value = evaluate[best_ID]
            history_best = pop[best_ID]
        else:
            pass

        # 文件写入
        with open(supervise_file, 'a') as file:
            file.write(
                f"最佳个体: 评估值{evaluate[best_ID]}\nroot_up:{data_list[best_ID]['root_up']}"
                f"\nroot_low:{data_list[best_ID]['root_low']}\n")
            file.write(f"tip_up:{data_list[best_ID]['tip_up']}\ntip_low:{data_list[best_ID]['tip_low']}")
        # 迭代过程的保存
        with open(population_file, 'a') as file:
            file.write(f"Iterations: {iterations + 1}\npopulation:\n{pop}\nevaluation\n{evaluate}")

        # 判定方均根收敛准则
        value, error = iteration_ctrl(evaluate, variance)
        if error != 0:
            with open(supervise_file, 'a') as file:
                file.write(f"\n all population are zero\n Abnormal termination\n")
            break
        if value < variance:
            break
        else:
            pass
        with open(supervise_file, 'a') as file:
            file.write(f"\nvariance value:{value}\n")

    # # 历史最佳个体写入
    root_up, root_low, tip_up, tip_low = algorithm_1.disturb_modify(history_best)
    with open(supervise_file, 'a') as file:
        file.write(
            f"\n历史最佳个体: \nroot_up:{root_up}"
            f"\nroot_low:{root_low}\n")
        file.write(f"tip_up:{tip_up}\ntip_low:{tip_low}")

    # 后处理画图
    index = np.arange(0, iterations+1)
    plt.figure()
    plt.scatter(index, best_evaluate[:iterations+1])
    plt.title('PSO Optimization Trend')
    plt.xlabel('Iterations')
    plt.ylabel('Evaluate')
    plt.savefig("D:\\aircraft design competition\\24solar\\design_model\\whole_wing_optimization"
                "\\supervise\\PSO-PCA_first_try.png")
    plt.show()

    print(index)
    print(best_evaluate)
    print(history_best)

    # 进化完成后输出几何模型
    root_up, root_low, tip_up, tip_low = algorithm_1.disturb_modify(history_best)
    vsp_3.create_Geom_1(root_up, root_low, tip_up, tip_low, len(root_up) - 1)
    return 0


# #############################################################
# ################## GWO-PCA-Optimization #####################
# #############################################################
def GWO_PCA_Optimization(iteration, population):
    return 0


# #############################################################
# ###################### GA-Optimization ######################
# #############################################################
def GA_Optimization(iteration, population):
    return 0


# #############################################################
# ###################### iteration_ctrl #######################
# #############################################################
# 用于评估函数是否满足收敛指标
def iteration_ctrl(evaluate, variance):
    value = 0  # 中间值
    avr = 0  # 均值储存
    length = len(evaluate)  # 适应度数组长度
    error = 0

    # 除零检查
    if length == 0:
        length = 1
    else:
        pass

    for i in range(length):
        avr += evaluate[i]
    avr = avr / length
    # 这个模型接入可能会存在种群全是0的情况 采用除以零保护
    if avr == 0:
        avr = 1
        error = 1
        print("error \nall population unreasonable\n")
    else:
        pass

    for i in range(length):
        value += np.power((evaluate[i] - avr) / avr, 2)  # 标准化然后平方
    value = value / length
    value = np.sqrt(value)
    return value, error

# #############################################################
# #################### 简单的demo与测试 #########################
# #############################################################
# PSO_PCA_Optimization(30, 20)
