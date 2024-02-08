import numpy as np
import balancing_3
import paths
import vsp_4
import steaty_state_evaluation_0
import CST_Generate

"""
总体思路
pop的输入可以获得种群中所有个体的信息将每一个种群输入到CST_Generate中
可以对每个个体实现检查几何特征，满足几何特征的约束。 生成的满足几何特征的pop
需要注意由于导入的方式需要在每一次生成后都调用评估函数进行计算得到一个对应的评估值
根据返回重置次数信息来决定是否对种群进行更新
最终返回种群和评估值
"""


# 注意由于vsp_4中文件导入方式发生了更改在solve中的评估值求解方式发生了改变，需要导入所有的pop并通过约束重新生成计算满足条件的pop
# pop 为直接导入的种群 flag 作为求解方式作为神经网络的拓展 0为配平器 1为神经网络的代理模型

def solve(pop, flag):
    tip = paths.tip_file
    root = paths.root_file
    length = len(pop)
    evaluate_mode = 2  # 评估函数选取
    evaluation = np.zeros(length)
    # 硬算配平
    if flag == 0:
        for i in range(length):
            pop_checked, iteration = CST_Generate.Geom_Generate(pop[i], tip, root)
            # 如果进行了更改则更新种群
            if iteration != 0:
                pop[i] = pop_checked
            else:
                pass
            # 文件路径已经写入 可以读取计算气动数据了
            alpha, Xcg = balancing_3.balance(paths.tip_file, paths.root_file)
            cl, _, cdi, CD_tot = vsp_4.vsp_aero(Xcg, alpha)  # 这里是直接读取几何文件的所以不用再输入
            # 气动解算完成引入评估函数
            if evaluate_mode == 2:
                evaluation[i] = evaluate(cl, CD_tot, evaluate_mode)  # 接入稳态爬升模型
            else:
                evaluation[i] = evaluate(cl, cdi, evaluate_mode)  # 代数评估函数代理

    # 神经网络代理
    elif flag == 1:
        pass
    else:
        pass
    # 把评估值和pop都发回来
    return evaluation, pop


# 评估函数
def evaluate(cl, cdi, flag):
    value = 0
    # 代数简单代理
    if flag == 0:
        # 表明计算不出一个合理的速度 同时平飞的时候速度不能超过拉力模型上限（过高的速度被认为不合理）
        if (steaty_state_evaluation_0.steady_velocity_evaluation(cl[0], 0.5 + cdi[0])[0] or cl[0] > 0.186) == 0:
            value = 0
        else:
            value = 1 / cdi[0]
    # 升阻比代理
    elif flag == 1:
        value = cl[0] / cdi[0]
    # 接入稳态爬升模型
    elif flag == 2:
        # 名字是cdi 实际上输入的是 CDtot
        temp = steaty_state_evaluation_0.steady_velocity_evaluation(cl[0], cdi[0])
        value = temp[0]
    else:
        pass
    return value
