import balancing_1
import numpy as np
import vsp_3


# data 是字典组成的数组 储存翼型信息 flag是计算类型 0-2是调用配平器的尝试 3是直接计算 4是代理模型
def solve(data, flag):
    length = len(data)
    evaluation = np.zeros(length)
    if flag == 0 or flag == 1 or flag == 2:  # 调用配平器求解
        for i in range(len(data)):
            alpha, Xcg, error = balancing_1.balance(flag, data[i]['root_up'], data[i]['root_low'], data[i]['tip_up'],
                                                    data[i]['tip_low'], 7)
            if error == 0:
                cl, _, cdi = vsp_3.vsp_aero(Xcg, alpha)  # 这里是直接读取几何文件的所以不用再输入
                evaluation[i] = evaluate(cl, cdi, 1)  # 代数评估函数代理
            else:  # 计算出错
                evaluation[i] = 0

    elif flag == 3:  # 不经过配平预设攻角进行计算
        Xcg = 0.3
        alpha = 2
        for i in range(len(data)):
            vsp_3.create_Geom_1(data[i]['root_up'], data[i]['root_low'], data[i]['tip_up'],
                                data[i]['tip_low'], 7)
            cl, _, cdi = vsp_3.vsp_aero(Xcg, alpha)
            evaluation[i] = evaluate(cl, cdi, 1)  # 升阻比代理

    elif flag == 4:  # 只是代理验证算法是否收敛 全部加起来取倒数
        for i in range(len(data)):
            for j in range(len(data[i]['root_up'])):
                evaluation[i] += np.fabs(data[i]['root_up'][j])
                evaluation[i] += np.fabs(data[i]['root_low'][j])
                evaluation[i] += np.fabs(data[i]['tip_up'][j])
                evaluation[i] += np.fabs(data[i]['tip_low'][j])
            evaluation[i] = 1/evaluation[i]
    else:
        pass

    return evaluation


# 评估函数
def evaluate(cl, cdi, flag):
    value = 0
    if flag == 0:  # 代数简单代理
        value = 10 / cdi[0] + cl[0] / cdi[0]
    elif flag == 1:  # 升阻比代理
        value = 1 / cdi[0]
    elif flag == 2:  # 准备接入稳态爬升模型 还没有写好
        pass
    else:
        pass
    return value
