import numpy as np
from scipy.optimize import root


# import matplotlib.pyplot as plt
#
# # 已知数据
# P_known = np.array([0, 50, 60, 70, 80, 90])
# V_known = np.array([0, 5, 7.3, 10.5])
# thrust_known = np.array([
#     [0, 5.4330, 5.9330, 6.7990, 7.4080, 7.9560],
#     [0.4000, 3.2980, 3.9540, 4.6620, 5.0300, 5.6390],
#     [-0.9500, 2.7300, 3.2090, 4.0220, 4.4510, 5.0160],
#     [-1.5860, 1.7660, 2.3770, 2.8390, 3.3780, 3.5400]
# ])
#
# # 创建网格
# P_grid, V_grid = np.meshgrid(P_known, V_known)
# xData = np.vstack((P_grid.flatten(), V_grid.flatten())).T
# yData = thrust_known.flatten()
#
#
# # 定义高斯函数
# def gauss2D(x, amplitude, xo, yo, sigma_x, sigma_y, offset):
#     xo = float(xo)
#     yo = float(yo)
#     g = amplitude * np.exp(-((x[:, 0] - xo) ** 2 / (2 * sigma_x ** 2) +
#                              (x[:, 1] - yo) ** 2 / (2 * sigma_y ** 2))) + offset
#     return g.ravel()
#
#
# # 初始参数猜测
# initial_guess = [np.max(yData), np.mean(P_known), np.mean(V_known), np.std(P_known), np.std(V_known), np.min(yData)]
# # 拟合
# beta = curve_fit(gauss2D, xData, yData, p0=initial_guess)[0]
#
#
# # 定义Tpv函数
# def Tpv(p, v):
#     # 使用拟合得到的参数计算推力
#     x = np.array([[p, v]])
#     return gauss2D(x, *beta)


def equations(x, CL, Cdi):
    gamma1, V1 = x
    W = 1.08 * 9.8
    CD0 = 0.02194
    S = 0.743
    P = 88
    rho = 1.225
    coeff = [145.2708, 112.9647, 82.7283, -125.9123, 57.4307, -4.706]
    res = [V1 - np.sqrt((2 * W * np.cos(gamma1)) / (rho * S * CL)),
           coeff[0] * np.exp(-((P - coeff[1]) ** 2 / (2 * coeff[2] ** 2) + (V1 - coeff[3]) ** 2 / (2 * coeff[4] ** 2)))
           + coeff[5] - 0.5 * rho * S * V1 ** 2 * (CD0 + Cdi) - W * np.sin(gamma1)]  # 这里需要注意k cl 这种方式和阻力计算并不匹配
    return res


def ROC_cl(cdi, cl):
    # 初始猜测值
    x0 = np.array([10 / 180 * np.pi, 6])
    # 调用root函数
    solutions = root(equations, x0, args=(cl, cdi))
    # 打印解
    print(solutions.x)
    res = solutions.x[1] * np.sin(solutions.x[0])
    print(f"{res}m/s")
    return res


# # 经过几个计算点的尝试，大致满足要求 这里直接转换为Cdi的问题避免那个k的不准确问题 （实在不明白Openvsp的做法）
# ROC_cl(cdi=0.032 * 0.13**2, cl=0.13)
