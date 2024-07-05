import numpy as np
import CST_Generate_2
import vsp_4
# import paths
import math
import matplotlib.pyplot as plt
import paths


# import CST_Generate_1

# arr = np.array([0.23491444598398273, 0.33358884593665983, 0.2410681245948411, 0.2, 0.4, -0.4, -0.4, 0.2,
#                 0.2626700635899731, 0.2, 0.2, -0.35398911012660284, 0.2302333423466283, -0.16407931006779278, 1.0])
# Gemo_Generate.CST_airfoil_file_generate(arr, paths.tip_file,
#                                         paths.mid_file, paths.root_file)
# vsp_4.create_Geom_3(paths.tip_file, paths.mid_file, paths.root_file)

# plt.savefig('root_tip_former_comparison.png')
def writefile(x, y, outputPath):
    with open(outputPath, 'w') as file:
        for i_line in range(len(x)):
            line = f'   {x[i_line]:.5f}   {y[i_line]:.5f}\n'
            file.write(line)
    print("airfoil file output successfully\n")
    return 0


# 输入目标的完整文件路径和文件名 返回翼型的x,y数组
def readFile(inputPath):
    with open(inputPath, 'r', encoding='ANSI') as infile:
        # 读取文件
        x = []
        y = []
        for line in infile:
            data_line = line.strip('\n').split()
            # x, y = int(data_line)
            if data_line:
                try:
                    x.append(float(data_line[0]))
                except ValueError:
                    continue
                try:
                    y.append(float(data_line[1]))
                except ValueError:
                    continue
            else:
                pass
    # 返回原始数据
    return x, y


# # 绘制root与former_root以及tip与former_tip的对比图
# root_x, root_y = readFile(paths.root_file)
# f_root_x, f_root_y = readFile(paths.former_root)
# tip_x, tip_y = readFile(paths.tip_file)
# f_tip_x, f_tip_y = readFile(paths.former_tip)
# # 创建一个包含两个子图的figure
# fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(10, 5))
#
# # 绘制第一个子图：root与former_root
# axs[0].plot(root_x, root_y, 'r-', label='Root')
# axs[0].plot(f_root_x, f_root_y, 'r--', label='Former Root')
# axs[0].set_title('Comparison of Root and Former Root')
# axs[0].set_xlabel('X-axis')
# axs[0].set_ylabel('Y-axis')
# axs[0].legend()
# axs[0].grid(True)
#
# # 绘制第二个子图：tip与former_tip
# axs[1].plot(tip_x, tip_y, 'b-', label='Tip')
# axs[1].plot(f_tip_x, f_tip_y, 'b--', label='Former Tip')
# axs[1].set_title('Comparison of Tip and Former Tip')
# axs[1].set_xlabel('X-axis')
# axs[1].set_ylabel('Y-axis')
# axs[1].legend()
# axs[1].grid(True)
#
# # 调整子图之间的间距
# plt.tight_layout()
#
# # 显示图片
# # plt.show()
# x1, y1 = readFile(".\\airfoil_file\\tip_up.txt")
# x2, y2 = readFile(".\\airfoil_file\\tip_low.txt")
# # print(x1, y1, x2, y2)
# t = 0.12
# y_mid = np.zeros(len(x1))
# x1 = np.array(x1)
# x2 = np.array(x2)
# y1 = np.array(y1)
# y2 = np.array(y2)
# for i in range(len(x1)):
#     y_mid[i] = (y1[i] + y2[i]) / 2
# y_t = np.zeros(len(x1))
# for i in range(len(x1)):
#     y_t[i] = 5 * t * (0.2969 * math.sqrt(x1[i]) - 0.1260 * x1[i] - 0.3516 * x1[i] ** 2 + 0.2843 *
#                       x1[i] ** 3 - 0.1036 * x1[i] ** 4)
# for i in range(len(x1)):
#     y1[i] = y_mid[i] + y_t[i]
#     y2[i] = y_mid[i] - y_t[i]
# temp = CST_Generate_2.Stress_Check(x1, x1, y2, y1)
# print(temp)
# x_reverse = x1[::-1]
# up_reverse = y1[::-1]
# x_combined = np.concatenate((x_reverse, x1))
# y_combined = np.concatenate((up_reverse, y2))
# # writefile(x_combined, y_combined, ".\\airfoil_file\\result_mid.dat")
# plt.axis('equal')
# plt.plot(x_combined, y_combined)
# plt.show()

# x1, y1 = readFile(".\\airfoil_file\\tip_up.txt")
# x2, y2 = readFile(".\\airfoil_file\\tip_low.txt")
# t = 0.12
# y_mid = np.zeros(len(x1))
# x1 = np.array(x1)
# x2 = np.array(x2)
# y1 = np.array(y1)
# y2 = np.array(y2)
# for i in range(len(x1)):
#     y_mid[i] = (y1[i] + y2[i]) / 2
# y_t = np.zeros(len(x1))
# for i in range(len(x1)):
#     y_t[i] = 5 * t * (0.2969 * math.sqrt(x1[i]) - 0.1260 * x1[i] - 0.3516 * x1[i] ** 2 + 0.2843 *
#                       x1[i] ** 3 - 0.1036 * x1[i] ** 4)
# for i in range(len(x1)):
#     y1[i] = y_mid[i] + y_t[i]
#     y2[i] = y_mid[i] - y_t[i]
# x_reverse = x1[::-1]
# up_reverse = y1[::-1]
# tip_x = np.concatenate((x_reverse, x1))
# tip_y = np.concatenate((up_reverse, y2))
# # writefile(tip_x, tip_y, paths.tip_file)
#
# x1, y1 = readFile(".\\airfoil_file\\mid_up.txt")
# x2, y2 = readFile(".\\airfoil_file\\mid_low.txt")
# t = 0.12
# y_mid = np.zeros(len(x1))
# x1 = np.array(x1)
# x2 = np.array(x2)
# y1 = np.array(y1)
# y2 = np.array(y2)
# for i in range(len(x1)):
#     y_mid[i] = (y1[i] + y2[i]) / 2
# y_t = np.zeros(len(x1))
# for i in range(len(x1)):
#     y_t[i] = 5 * t * (0.2969 * math.sqrt(x1[i]) - 0.1260 * x1[i] - 0.3516 * x1[i] ** 2 + 0.2843 *
#                       x1[i] ** 3 - 0.1036 * x1[i] ** 4)
# for i in range(len(x1)):
#     y1[i] = y_mid[i] + y_t[i]
#     y2[i] = y_mid[i] - y_t[i]
# x_reverse = x1[::-1]
# up_reverse = y1[::-1]
# mid_x = np.concatenate((x_reverse, x1))
# mid_y = np.concatenate((up_reverse, y2))
# # writefile(mid_x, mid_y, paths.mid_file)
#
# x1, y1 = readFile(".\\airfoil_file\\root_up.txt")
# x2, y2 = readFile(".\\airfoil_file\\root_low.txt")
# t = 0.12
# y_mid = np.zeros(len(x1))
# x1 = np.array(x1)
# x2 = np.array(x2)
# y1 = np.array(y1)
# y2 = np.array(y2)
# for i in range(len(x1)):
#     y_mid[i] = (y1[i] + y2[i]) / 2
# y_t = np.zeros(len(x1))
# for i in range(len(x1)):
#     y_t[i] = 5 * t * (0.2969 * math.sqrt(x1[i]) - 0.1260 * x1[i] - 0.3516 * x1[i] ** 2 + 0.2843 *
#                       x1[i] ** 3 - 0.1036 * x1[i] ** 4)
# for i in range(len(x1)):
#     y1[i] = y_mid[i] + y_t[i]
#     y2[i] = y_mid[i] - y_t[i]
# x_reverse = x1[::-1]
# up_reverse = y1[::-1]
# root_x = np.concatenate((x_reverse, x1))
# root_y = np.concatenate((up_reverse, y2))
# # writefile(root_x, root_y, paths.root_file)
#
# # 绘制三个翼型
# fig, axs = plt.subplots(3, 1)  # 创建一个3行1列的子图
#
# # 设置大标题
# fig.suptitle('Result Airfoil', fontsize=16)
#
# # 绘制root翼型
# axs[0].plot(root_x, root_y)
# axs[0].set_title('Root Airfoil')
# axs[0].set_xlabel('x')
# axs[0].set_ylabel('y')
# axs[0].set_aspect('equal', adjustable='box')  # 确保x轴和y轴坐标相等
#
# # 绘制mid翼型
# axs[1].plot(mid_x, mid_y)
# axs[1].set_title('Mid Airfoil')
# axs[1].set_xlabel('x')
# axs[1].set_ylabel('y')
# axs[1].set_aspect('equal', adjustable='box')  # 确保x轴和y轴坐标相等
#
# # 绘制tip翼型
# axs[2].plot(tip_x, tip_y)
# axs[2].set_title('Tip Airfoil')
# axs[2].set_xlabel('x')
# axs[2].set_ylabel('y')
# axs[2].set_aspect('equal', adjustable='box')  # 确保x轴和y轴坐标相等
# # 调整子图间距
# plt.tight_layout()
# # 显示图形
# plt.show()

# # 一个计算
# vsp_4.create_Geom_4(paths.tip_file, paths.mid_file, paths.root_file, angle=10.12745129)
# # CMy, cl
# x_cg = 0.0287 + 0.119  # 给定设定的重心位置
# # 这个相当于是配平条件的攻角
# angle = 10 * np.pi / 180
# sref = ((2 * 0.277 / np.cos(angle) + 0.05 * np.sin(angle)) * 0.05 + 2 * (
#         1.25 + 0.277 * np.tan(angle)) * np.cos(angle)
#         * 0.277 / np.cos(angle))
# bref = 2 * ((1.25 + 0.277 * np.tan(angle)) * np.cos(angle) + 0.05)
# cref = sref / bref
# a, b = vsp_4.vsp_aero_sweep_1(-5, 5, 2, sref=sref, bref=bref, cref=cref, xcg=x_cg)
# alpha = -10 * a[0] / (a[1] - a[0]) - 5
# # 考虑角度的控制，代理里面也要改过
# if alpha > 10 or alpha < -5:
#     pass
# else:
#     c = vsp_4.vsp_aero_0(x_cg=x_cg, aoa=alpha, sref=sref, bref=bref, cref=cref)
#     print(c)
#     print(alpha)
