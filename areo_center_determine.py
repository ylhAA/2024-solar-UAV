import numpy as np
import matplotlib.pyplot as plt
import vsp_4
import paths
from Gemo_Generate import CST_airfoil_file_generate


# test = np.zeros(16)
# for i in range(14):
#     test[i] = 0.1
# test[14] = 0.5
# x_cg = 0.25859
# sweep = np.arange(start=10, stop=21)
# a_center = np.zeros(len(sweep))
# for i in range(len(sweep)):
#     angle = sweep[i] * np.pi / 180
#     sref = ((2 * 0.277 / np.cos(angle) + 0.05 * np.sin(angle)) * 0.05 + 2 * (
#             1.25 + 0.277 * np.tan(angle)) * np.cos(angle)
#             * 0.277 / np.cos(angle))
#     bref = 2 * ((1.25 + 0.277 * np.tan(angle)) * np.cos(angle) + 0.05)
#     cref = sref / bref
#     CST_airfoil_file_generate(test, paths.tip_file, paths.mid_file, paths.root_file)
#     vsp_4.create_Geom_4(paths.tip_file, paths.mid_file, paths.root_file, angle=angle*180/np.pi)
#     cmy, cl = vsp_4.vsp_aero_sweep_1(start=-5, end=5, num=2, sref=sref, bref=bref, cref=cref)
#     a_center[i] = x_cg - (cmy[1] - cmy[0])/(cl[1] - cl[0]) * cref
# plt.plot(sweep, a_center)
# plt.show()
# print(sweep, a_center)

# # 定义x和y数据
# x = np.array([10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
# y = np.array([0.17088883, 0.1824521, 0.19412071, 0.20588436, 0.2177672, 0.22974182,
#               0.24180388, 0.25395514, 0.26619431, 0.27853446, 0.29098629])
#
# for i in range(len(x)):
#     angle = x[i] * np.pi / 180
#     sref = ((2 * 0.277 / np.cos(angle) + 0.05 * np.sin(angle)) * 0.05 + 2 * (
#             1.25 + 0.277 * np.tan(angle)) * np.cos(angle)
#             * 0.277 / np.cos(angle))
#     bref = 2 * ((1.25 + 0.277 * np.tan(angle)) * np.cos(angle) + 0.05)
#     cref = sref / bref
#     y[i] = y[i] - 0.08 * cref
# # 使用numpy的polyfit函数进行线性拟合（1表示一次多项式，即线性）
# coefficients = np.polyfit(x, y, 1)
# # 系数是斜率和截距，即 y = coefficients[0] * x + coefficients[1]
# slope, intercept = coefficients
# # 创建一个x值的范围用于绘制拟合线
# x_fit = np.linspace(min(x), max(x), 100)
# y_fit = slope * x_fit + intercept
# # 绘制原始数据点和拟合线
# plt.scatter(x, y, label='Original data')
# plt.plot(x_fit, y_fit, color='red', label='Fitted line')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.legend()
# plt.show()
# # 打印拟合的斜率和截距
# print(f'Slope: {slope:.4f}')
# print(f'Intercept: {intercept:.4f}')