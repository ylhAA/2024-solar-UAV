import numpy as np
from scipy.stats import qmc
# import matplotlib.pyplot as plt
# import os
import paths
import file_paths_generate
import Gemo_Generate
import vsp_4
import CST_Generate_2


# #############################################################
# #################### 简单的demo与测试 #########################
# #############################################################

# 给定一个文件路径，输入一维数组用于写入文件数据
def write_data(dir_path, data):
    # 将数据转换为NumPy数组（如果它还不是）
    data = np.array(data)
    data_str = ' '.join(map(str, data)) + '\n'
    with open(dir_path, 'a') as fdata:
        fdata.write(data_str)
    return 0


# 给定一个文件路径，读取文件数据
def read_data(dir_path):
    data_temp = []
    with open(dir_path, 'r') as fdata:
        while True:
            line = fdata.readline()
            if not line:
                break
            data_temp.append([float(i) for i in line.split()])
    return np.array(data_temp)


# #############################################################
# #################### 简单的demo与测试 #########################
# #############################################################

# 生成文件路径
file_paths_generate.check_and_generate_path()
# 准备开始生成
dim = 15  # 参数空间的大小
max_population = 30
data_path = paths.dictionary_file
lhs = qmc.LatinHypercube(dim)
pop = lhs.random(max_population)
for _ in range(max_population):
    for j in range(0, 4):
        pop[_][j] = pop[_][j] * 0.3 + 0.1
    for j in range(7, 11):
        pop[_][j] = pop[_][j] * 0.3 + 0.1
    for j in range(4, 7):
        pop[_][j] = pop[_][j] * 1 - 0.5
    for j in range(11, 14):
        pop[_][j] = pop[_][j] * 1 - 0.5
for _ in range(max_population):
    with open(paths.supervise_file, 'a') as file:
        file.write(f"{_}\n")
    flag = Gemo_Generate.CST_airfoil_file_generate(pop[_], paths.tip_file, paths.mid_file, paths.root_file)
    if flag == 1:
        pass
    else:
        # 默认后掠角度是10度
        vsp_4.create_Geom_4(paths.tip_file, paths.mid_file, paths.root_file)
        # CMy, cl
        x_cg = 0.148  # 给定设定的重心位置注意后掠角发生变化其设计重心也发生变动
        a, b = vsp_4.vsp_aero_sweep_1(-5, 5, 2)
        alpha = -10 * a[0] / (a[1] - a[0]) - 5
        if 10 > alpha > -5:
            c = vsp_4.vsp_aero_0(x_cg=x_cg, aoa=alpha)
            print(c)
            combined = np.concatenate([pop[_], a, b, c[2]])
            write_data(data_path, combined)
        else:
            pass

# # 生成文件路径 (增加后掠角设计的处理)
# file_paths_generate.check_and_generate_path()
# # 准备开始生成
# dim = 16  # 参数空间的大小
# max_population = 4
# data_path = ".\\train_data\\test_data_4.txt"
# lhs = qmc.LatinHypercube(dim)
# pop = lhs.random(max_population)
# for k in range(max_population):
#     for j in range(0, 4):
#         pop[k][j] = pop[k][j] * 0.3 + 0.1
#     for j in range(7, 11):
#         pop[k][j] = pop[k][j] * 0.3 + 0.1
#     for j in range(4, 7):
#         pop[k][j] = pop[k][j] * 1 - 0.5
#     for j in range(11, 14):
#         pop[k][j] = pop[k][j] * 1 - 0.5
#     pop[k][15] = pop[k][15] * 10 + 10
# for _ in range(max_population):
#     with open(paths.supervise_file, 'a') as file:
#         file.write(f"{_}\n")
#     flag = Gemo_Generate.CST_airfoil_file_generate(pop[_], paths.tip_file, paths.mid_file, paths.root_file)
#     angle = pop[_][15] * np.pi / 180
#     sref = ((2 * 0.277/np.cos(angle) + 0.05 * np.sin(angle)) * 0.05 + 2 * (1.25 + 0.277 * np.tan(angle)) * np.cos(angle)
#             * 0.277 / np.cos(angle))
#     bref = 2 * ((1.25 + 0.277 * np.tan(angle)) * np.cos(angle) + 0.05)
#     cref = sref / bref
#     if flag == 1:
#         pass
#     else:
#         vsp_4.create_Geom_4(paths.tip_file, paths.mid_file, paths.root_file, angle=pop[_][15])
#         # CMy, cl
#         x_cg = 0.0287 + 0.0119 * pop[_][15]  # 给定设定的重心位置
#         a, b = vsp_4.vsp_aero_sweep_1(-5, 5, 2, sref=sref, bref=bref, cref=cref, xcg=x_cg)
#         alpha = -10 * a[0] / (a[1] - a[0]) - 5
#         if 10 > alpha > -5:
#             c = vsp_4.vsp_aero_0(x_cg=x_cg, aoa=alpha, sref=sref, bref=bref, cref=cref)
#             print(c)
#             combined = np.concatenate([pop[_], a, b, c[2]])
#             write_data(data_path, combined)
#         else:
#             pass
#
# pop = read_data(paths.dictionary_file)
# print(pop.shape)
# # 提取前九列
# first_array = pop[:, :9]
#
# # 提取第十列和第十一列
# second_array = pop[:, [9, 10]]
#
# # 提取第十二列和第十三列
# third_array = pop[:, [11, 12]]
#
# # 验证新数组的形状
# print(first_array.shape)  # 输出应该是 (1800, 9)
# print(second_array.shape)  # 输出应该是 (1800, 2)
# print(third_array.shape)  # 输出应该是 (1800, 2)
#
# # 生成文件路径
# # 准备做绘图的一些检验
# file_paths_generate.check_and_generate_path()
# # 准备开始生成
# rate = 0.04  # 参数扰动范围
# dim = 4  # 参数空间的大小
# max_population = 50
# # 设置保存图像的目录
# save_dir = r"E:\_\result\agent-model\picture"
# lhs = qmc.LatinHypercube(dim)
# pop = lhs.random(max_population)
# for _ in range(max_population):
#     for j in range(4):
#         pop[_][j] = pop[_][j] * 2*rate - rate
# for _ in range(max_population):
#     xx, yy, a, b = CST_Generate_2.airfoil_Generate(pop[_], order=3, number=20)
#     plt.plot(xx, yy)
#     # 设置文件名
#     filename = f"airfoil_{_ + 1:03d}.png"  # 03d表示至少3位数，不足的前面补0
#     filepath = os.path.join(save_dir, filename)
#     # 保存图像
#     plt.savefig(filepath)
#     plt.close()  # 关闭当前图形窗口，释放资源
#
#
# file_paths_generate.check_and_generate_path()
# tep = read_data(paths.dictionary_file)
# pop = tep[:, :9]
# cmy = tep[:, [9, 10]]
# print(pop.shape)
# print(cmy.shape)
# alpha = np.zeros(len(pop))
# for _ in range(len(pop)):
#     alpha[_] = -10 * cmy[_][0] / (cmy[_][1] - cmy[_][0]) - 5
# print(alpha)
# print(alpha.shape)
#
# # NACA生成
# file_paths_generate.check_and_generate_path()
# dim = 7
# max_population = 195
# data_path = paths.dictionary_file
# lhs = qmc.LatinHypercube(dim)
# pop = lhs.random(max_population)
# for _ in range(max_population):
#     pop[_][0] = pop[_][0] * 0.06
#     pop[_][3] = - pop[_][3] * 0.06
#     pop[_][1] = pop[_][1] * 0.2 + 0.3
#     pop[_][4] = pop[_][4] * 0.2 + 0.3
#     pop[_][2] = pop[_][2] * 0.05 + 0.1
#     pop[_][5] = pop[_][5] * 0.05 + 0.1
# print(pop)
# for _ in range(max_population):
#     Gemo_Generate.airfoil_file_generate(pop[_], paths.tip_file, paths.mid_file, paths.root_file)
#     vsp_4.create_Geom_3(paths.tip_file, paths.mid_file, paths.root_file)
#     # CMy, cl
#     x_cg = 0.25859  # 给定设定的重心位置
#     a, b = vsp_4.vsp_aero_sweep_1(-5, 5, 2)
#     alpha = -10 * a[0] / (a[1] - a[0]) - 5
#     c = vsp_4.vsp_aero_0(x_cg=x_cg, aoa=alpha)
#     print(c)
#     combined = np.concatenate([pop[_], a, b, c[2]])
#     write_data(data_path, combined)
