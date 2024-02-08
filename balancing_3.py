import paths
import vsp_4
import numpy as np
from datetime import datetime
from sklearn.linear_model import LinearRegression
# import matplotlib.pyplot as plt
"""
适用与dat格式输入的配平组件
不需要预先输入CST参数化翼型
balance函数直接缺省
要求调用后直接输出配平重心和攻角
"""
# # 参考示例 NACA 3412的颠倒 和 NASA SC-0412
# stage = 7
# root_up = [0.20027, 0.06942, 0.26365, 0.01476, 0.27234, 0.14104, 0.17256, 0.20330]
# root_low = [-0.20027, -0.08095, -0.21768, -0.12756, -0.14138, -0.24818, 0.04641, 0.16772]
# tip_up = [0.15727, 0.04782, 0.19466, -0.10213, 0.22039, -0.03441, 0.06494, 0.06763]
# tip_low = [-0.15727, -0.33244, -0.11350, -0.26489, -0.29933, -0.14698, -0.27400, -0.25044]

# stage = 7
# root_up = [0.20027, 0.1942, 0.26365, 0.01576, 0.27234, 0.15104, 0.17256, 0.20330]
# root_low = [-0.20027, -0.08095, -0.21768, -0.12756, -0.14138, -0.26818, 0.04641, 0.16772]
# tip_up = [0.15727, 0.04782, 0.19466, -0.16213, 0.22039, -0.03441, 0.06494, 0.06763]
# tip_low = [-0.15727, -0.3744, -0.11350, -0.26489, -0.29933, -0.14698, -0.23400, -0.25044]


# #############################################################
# ######################## 线性规划 ############################
# #############################################################

def linear_regression_alpha(x, y):
    X = np.array(x).reshape(-1, 1)
    Y = np.array(y).reshape(-1, 1)
    model = LinearRegression()
    model.fit(X, Y)
    intercept = model.intercept_
    slope = model.coef_[0]
    alpha = -intercept / slope
    print("零点位置为:", alpha)
    return alpha


# #############################################################
# #################### 焦点计算cal_xcg #########################
# #############################################################

def cal_xcg():
    cref = 0.304  # 平均弦长
    X_f = np.zeros(4)  # 用来储存几个数据
    # vsp_aero_sweep_0 直接调用vsp3路径完成计算
    Cmy, Cl = vsp_4.vsp_aero_sweep_0(-8, 8)
    X_f[0] = cref * (Cmy[1] - Cmy[0]) / (Cl[0] - Cl[1])
    X_f[1] = cref * (Cmy[1] - Cmy[2]) / (Cl[2] - Cl[1])
    X_f[2] = cref * (Cmy[2] - Cmy[3]) / (Cl[3] - Cl[2])
    X_f[3] = cref * (Cmy[3] - Cmy[4]) / (Cl[4] - Cl[3])
    xf_value = np.mean(X_f)
    return xf_value, Cmy, Cl


# #############################################################
# ################### 配平攻角计算alpha_cal #####################
# #############################################################

def alpha_cal(xf_value, Cmy, Cl):
    cref = 0.304
    xcg = xf_value - 8e-2 * cref
    x = np.array([-8, -4, 0, 4, 8])  # 不能改已经做在vsp3里面了
    y = np.zeros(5)
    y[0] = Cl[0] * xcg + cref * Cmy[0]
    y[1] = Cl[0] * xcg + cref * Cmy[0] - (xf_value - xcg) * (Cl[1] - Cl[0])
    y[2] = Cl[0] * xcg + cref * Cmy[0] - (xf_value - xcg) * (Cl[2] - Cl[0])
    y[3] = Cl[0] * xcg + cref * Cmy[0] - (xf_value - xcg) * (Cl[3] - Cl[0])
    y[4] = Cl[0] * xcg + cref * Cmy[0] - (xf_value - xcg) * (Cl[4] - Cl[0])
    print(x, y)
    aoa = linear_regression_alpha(x, y)
    return xcg, aoa


# #############################################################
# ##################### 配平函数balance ########################
# #############################################################
def balance(tip_file, root_file):
    # 参数与default

    # 开始进行计算
    current_time = datetime.now()
    with open(paths.supervise_file, 'a') as file:
        file.write(f"解算开始时间: {current_time}\n")

    # 首先进行几何生成 保存到model.vsp3
    vsp_4.create_Geom_2(tip_file, root_file)

    # 开始计算焦点
    xf_value, Cmy, Cl = cal_xcg()

    # 计算配平攻角
    xcg, temp = alpha_cal(xf_value, Cmy, Cl)
    aoa = temp[0]
    with open(paths.supervise_file, 'a') as file:
        file.write(f"\nXcg: {xcg}\nalpha: {aoa}\n")

    return xcg, aoa

# #############################################################
# #################### 简单的demo与测试 #########################
# #############################################################
# a = [1, 2, 3]
# b = [2, 4, 5.9]
# linear_regression_alpha(a, b)

# cmy = (0.318104797473, 0.02446148082, -0.267792241554)
# cl = (-0.309001337142, 0.004809050798, 0.31795571797)
# xf = 0.25939731496952706 + 0.08 * 0.304
# alpha_cal(xf, cmy, cl)

# balance(root_up, root_low, tip_up, tip_low, stage)
