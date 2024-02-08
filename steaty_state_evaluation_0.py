import numpy as np
# import matplotlib.pyplot as plt
from scipy.interpolate import RectBivariateSpline


def steady_velocity_evaluation(cl, cd):
    # 计算常量设置
    power = 80  # 功率设置
    mass = 1.0  # 质量1.0kg
    g = 9.8  # ms^-2
    rho = 1.225  # 空气密度
    S = 0.78486  # 主翼面积
    v_ctrl = np.sqrt((2 * g * mass) / np.fabs(rho * S * cl)) - 1e-3  # 平飞配平 初始值预估
    # 从matlab中迁移的函数
    # (TT - (S*cd*rho*v^2)/2)^2/(g^2*mass^2) + (S^2*cl^2*rho^2*v^4)/(4*g^2*mass^2) - 1
    # value = ((Trust_evaluation(power, velocity, 2) - (S * cd * rho * velocity ** 2) / 2) ** 2 /
    #          (g ** 2 * mass ** 2) + (S ** 2 * cl ** 2 * rho ** 2 * velocity ** 4) / (
    #                  4 * g ** 2 * mass ** 2) - 1)
    # 初始化迭代
    velocity_0 = v_ctrl  # 初始猜测值
    tolerance = 1e-4  # 精度要求
    maxIter = 100  # 最大迭代次数
    iteration = 0  # 迭代步数
    delta = 1e10  # 设置一个很大的值如果变化小于这个就停止
    while delta > tolerance and iteration < maxIter:
        # 计算函数值和导数
        # 使用的是12*6.5的那组
        fuc = ((Trust_evaluation(power, velocity_0, 2) - (S * cd * rho * velocity_0 ** 2) / 2) ** 2 /
               (g ** 2 * mass ** 2) + (S ** 2 * cl ** 2 * rho ** 2 * velocity_0 ** 4) / (
                       4 * g ** 2 * mass ** 2) - 1)
        d_fuc = ((Trust_evaluation(power, (velocity_0 + tolerance), 2) - (
                S * cd * rho * (velocity_0 + tolerance) ** 2) / 2) ** 2 /
                 (g ** 2 * mass ** 2) + (S ** 2 * cl ** 2 * rho ** 2 * (velocity_0 + tolerance) ** 4) / (
                         4 * g ** 2 * mass ** 2) - 1)
        d_fuc = (d_fuc - fuc) / tolerance
        velocity_1 = velocity_0 - fuc / d_fuc
        delta = np.abs(velocity_1 - velocity_0)
        iteration += 1
        velocity_0 = velocity_1
    if iteration >= maxIter:
        print('迭代未收敛\n')
        vertical_velocity = 0
        gamma = 0
    elif velocity_0 > 10.5 or velocity_0 < 0:
        print('速度超出合理范围\n')
        vertical_velocity = 0
        velocity_0 = 0
        gamma = 0
    else:
        print('迭代完成，解为 v = \n', velocity_0)
        velocity = velocity_0
        gamma = np.arccos(0.5 * cl * rho * S * velocity_0 ** 2 / (g * mass))
        vertical_velocity = velocity * np.sin(gamma)
        # 弧度制角度值转换
        gamma = np.degrees(gamma)

    if vertical_velocity <= 0:
        evaluation = 0
    else:
        evaluation = vertical_velocity
    return evaluation, velocity_0, gamma


# 拉力模型
def Trust_evaluation(power, velocity, stage):
    if stage == 1:
        x = [50, 60, 70, 80, 90, 100]
        y = [0, 5, 7.1, 10.4]
        Thrust = np.array([[5.208, 2.724, 1.863, 0.725],
                           [5.794, 3.623, 2.451, 1.242],
                           [6.374, 4.121, 3.137, 1.788],
                           [7.41, 4.829, 3.678, 2.21],
                           [7.63, 5.232, 4.106, 2.653],
                           [8.357, 5.786, 4.737, 3.19]])
        interp_func_spline = RectBivariateSpline(x, y, Thrust, kx=2, ky=2)
        value = interp_func_spline(power, velocity)
    elif stage == 2:
        x = [0, 50, 60, 70, 80, 90]
        y = [0, 5, 7.3, 10.5]
        Thrust = np.array([[0, -0.4, -0.95, -1.586],
                           [5.433, 3.298, 2.73, 1.766],
                           [5.933, 3.954, 3.209, 2.377],
                           [6.799, 4.662, 4.022, 2.839],
                           [7.408, 5.03, 4.451, 3.378],
                           [7.956, 5.639, 5.016, 3.54]])
        interp_func_spline = RectBivariateSpline(x, y, Thrust, kx=2, ky=2)
        value = interp_func_spline(power, velocity)
    else:
        value = 0

    return value


# #############################################################
# #################### 简单的demo与测试 #########################
# #############################################################
# 生成绘图所需的数据  用来检验拉力模型
# P_values = np.linspace(50, 90, 50)
# v_values = np.linspace(0, 10, 50)
# tpv_values = np.zeros((50, 50))
# for i in range(50):
#     for j in range(50):
#         tpv_values[i][j] = Trust_evaluation(P_values[i], v_values[j], 1)
#
# plt.imshow(tpv_values, cmap='viridis', extent=[0, 10, 50, 90], aspect='auto', origin='lower')
# plt.colorbar(label='Thrust (N)')
# plt.xlabel('Velocity (m/s)')
# plt.ylabel('Power (W)')
# plt.title('Tpv Interpolation for 4004 15*5')
# plt.show()
#
# for i in range(50):
#     for j in range(50):
#         tpv_values[i][j] = Trust_evaluation(P_values[i], v_values[j], 2)
#
# plt.imshow(tpv_values, cmap='viridis', extent=[0, 10, 50, 90], aspect='auto', origin='lower')
# plt.colorbar(label='Thrust (N)')
# plt.xlabel('Velocity (m/s)')
# plt.ylabel('Power (W)')
# plt.title('Tpv Interpolation for 4004 12*6.5')
# plt.show()

# value = Trust_evaluation(90, 10, 1)
# print(value)
# value = Trust_evaluation(90, 0, 1)
# print(value)
# value = Trust_evaluation(50, 10, 1)
# print(value)
# value = Trust_evaluation(50, 0, 1)
# print(value)


# aa = Trust_evaluation(78, 8, 2)
# print(aa)

# # 检验上升率评估模型
# cd_start = 0.01
# cd_end = 0.20
# cl_start = 0.1
# cl_end = 1
# num = 250
# CD_tot = np.linspace(cd_start, cd_end, num)
# Cl_tot = np.linspace(cl_start, cl_end, num)
# evaluate = np.zeros((num, num))
# velocity_record = np.zeros((num, num))
# gamma_record = np.zeros((num, num))
# for i in range(num):
#     for j in range(num):
#         temp = steady_velocity_evaluation(Cl_tot[i], CD_tot[j])
#         evaluate[i][j] = temp[0]
#         velocity_record[i][j] = temp[1]
#         gamma_record[i][j] = temp[2]
#
#
# plt.imshow(evaluate, cmap='viridis', extent=[cl_start, cl_end, cd_start, cd_end], aspect='auto', origin='lower',
#            interpolation='bicubic')
# plt.colorbar(label='vertical velocity (m/s)')
# plt.xlabel('Cl')
# plt.ylabel('Cd')
# plt.title('Cl and Cd search vertical velocity for steady condition')
# plt.show()
#
# plt.imshow(velocity_record, cmap='viridis', extent=[cl_start, cl_end, cd_start, cd_end], aspect='auto',
# origin='lower', interpolation='bicubic') plt.colorbar(label='velocity (m/s)') plt.xlabel('Cl') plt.ylabel('Cd')
# plt.title('Cl and Cd search velocity for steady condition') plt.show()
#
# plt.imshow(gamma_record, cmap='viridis', extent=[cl_start, cl_end, cd_start, cd_end], aspect='auto', origin='lower',
#            interpolation='bicubic')
# plt.colorbar(label='angle (deg)')
# plt.xlabel('Cl')
# plt.ylabel('Cd')
# plt.title('Cl and Cd search gamma for steady condition')
# plt.show()

# value = Trust_evaluation(80, 20, 1)
# print(value)  #[[2.21]]
#

# degree = 1
# print(np.sin(degree))
# print(np.degrees(degree))
