import numpy as np
import matplotlib.pyplot as plt
import math
import os
import file_paths_generate
import CST_Generate_2
# import paths
import numpy.linalg as LA
from scipy.stats import qmc

# import paths

"""
这是一个用于翼型几何生成的尝试
初步规划为 
1 NACA系列
2 CST约束改进系列
"""


# 给定一个文件路径，输入一维数组用于写入文件数据
def write_data(dir_path, data):
    # 将数据转换为NumPy数组（如果它还不是）
    data = np.array(data)
    data_str = ' '.join(map(str, data)) + '\n'
    with open(dir_path, 'a') as fdata:
        fdata.write(data_str)
    return 0


def writefile(x, y, outputPath):
    with open(outputPath, 'w') as file:
        for i_line in range(len(x)):
            line = f'   {x[i_line]:.5f}   {y[i_line]:.5f}\n'
            file.write(line)
    print("airfoil file output successfully\n")
    return 0


def picture_generate_1(x, y):
    # 创建一个新的图形窗口，并设置其大小
    plt.figure(figsize=(10, 5))

    # 创建一个 1x2 的子图网格
    plt.subplots_adjust(hspace=0.4)  # 调整子图之间的垂直间距
    # 硬编码控制范围
    x_min = -0.1
    x_max = 1.1
    y_min = -0.1
    y_max = 0.2
    # 激活第一个子图并绘制散点图
    plt.subplot(2, 1, 1)
    plt.scatter(x, y, label='Scatter Plot')
    plt.xlim(x_min, x_max)  # 设置x轴范围
    plt.ylim(y_min, y_max)  # 设置y轴范围
    plt.title('Scatter Plot of Airfoil Coordinates')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.legend()
    plt.gca().set_aspect('equal', adjustable='box')  # 设置子图比例相等

    # 激活第二个子图并绘制连线图
    plt.subplot(2, 1, 2)
    plt.plot(x, y, label='Line Plot')
    plt.xlim(x_min, x_max)  # 设置x轴范围，与第一个子图相同
    plt.ylim(y_min, y_max)  # 设置y轴范围，与第一个子图相同
    plt.title('Line Plot of Airfoil Coordinates')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.legend()
    plt.gca().set_aspect('equal', adjustable='box')  # 设置子图比例相等
    # plt.show()

    return 0


# 生成图像
def picture_generate(x, y):
    # 创建一个新的图形窗口，并设置其大小
    plt.figure(figsize=(10, 5))
    # 创建一个 1x2 的子图网格，并激活第一个子图
    plt.subplot(2, 1, 1)
    # 绘制散点图
    plt.scatter(x, y, label='Scatter Plot')
    # 设置子图的标题和标签
    plt.title('Scatter Plot of Airfoil Coordinates')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.legend()
    # 激活第二个子图
    plt.subplot(2, 1, 2)
    # 绘制连线图
    plt.plot(x, y, label='Line Plot')
    # 设置子图的标题和标签
    plt.title('Line Plot of Airfoil Coordinates')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.legend()
    # 调整子图之间的间距
    plt.tight_layout()
    # 显示图形
    # plt.show()
    return 0


# #############################################################
# ############## CST_Airfoil_Generate by ylh ##################
# #############################################################
# 用于输入CST参数并输出系列坐标点是CST函数的定义
# 类别函数
def class_function(N_1, N_2, x):
    C = np.power(x, N_1) * np.power(1 - x, N_2)
    C = np.array(C)
    return C


# 形状函数
# a 拟合控制系数 B n阶bernstein多项式
def shape_function(coe, x):
    S = np.zeros(len(x))
    for i in range(len(x)):
        for j in range(len(coe)):
            S[i] += coe[j] * np.power(x[i], j) * np.power((1 - x[i]), len(coe) - j)
    return S


def CST_calculate(N_1, N_2, coe, x):
    C = class_function(N_1, N_2, x)
    S = shape_function(coe, x)
    y_cst = C * S
    return y_cst


def CST_airfoil(a, b, number):
    x_sin = np.linspace(0, np.pi / 2, number + 1)
    x = 0.5 * np.sin(2 * x_sin - np.pi / 2) + 0.5
    solve = CST_Generate_2.CST_generator(3, x)
    y_up = solve.generate_curve(a)
    y_low = solve.generate_curve(b)
    y_up = y_up.flatten()
    y_low = y_low.flatten()
    y_mid = (y_low + y_up) / 2
    # picture_generate(x, y_mid)
    flag = 0
    for i in range(len(x) - 2):
        if y_mid[i + 1] > y_mid[i] and y_mid[i - 1] > y_mid[i]:
            flag += 1
        elif y_mid[i + 1] < y_mid[i] and y_mid[i - 1] < y_mid[i]:
            flag += 1
        else:
            pass
    if flag <= 2:
        curve, thick, _ = CST_Generate_2.Stress_Check(x, x, y_low, y_up)
        if curve > 4.88 or thick < 2e-3:
            flag = 3
        else:
            pass
    else:
        pass
    # 满足设计条件进行写入
    x_reverse = x[::-1]
    y_up_reverse = y_up[::-1]
    # 合并上表面（倒置后）和下表面的数组
    x_combined = np.concatenate((x_reverse, x))
    y_combined = np.concatenate((y_up_reverse, y_low))
    # picture_generate_1(x_combined, y_combined)
    return x_combined, y_combined, flag


def CST_airfoil_file_generate(pop, tip_airfoil, mid_airfoil, root_airfoil):
    num = 50
    k = pop[14]
    root_up = pop[:4]
    tep1 = -pop[0]
    tep2 = pop[4:7]
    root_low = np.array([tep1] + tep2.tolist())
    tip_up = pop[7:11]
    tep1 = -pop[7]
    tep2 = pop[11:14]
    tip_low = np.array([tep1] + tep2.tolist())
    root_x, root_y, c = CST_airfoil(root_up, root_low, number=num)
    if c <= 2:
        writefile(root_x, root_y.flatten(), root_airfoil)
        tip_x, tip_y, c = CST_airfoil(tip_up, tip_low, number=num)
        tip_y = -tip_y
        # 顺序反转保证逆时针旋转
        tip_x = tip_x[::-1]
        tip_y = tip_y[::-1]
        # picture_generate(tip_x, tip_y)
        if c <= 2:
            writefile(tip_x, tip_y.flatten(), tip_airfoil)
            mid_x = root_x * k + tip_x * (1 - k)
            mid_y = root_y * k + tip_y * (1 - k)
            writefile(mid_x, mid_y.flatten(), mid_airfoil)
    if c > 2:
        print("geometry unreasonable!\n")
        flag = 1
    else:
        print("geometry successfully established!\n")
        flag = 0
    return flag


def CST_airfoil_gemo_check(pop):
    num = 50
    root_up = pop[:4]
    tep1 = -pop[0]
    tep2 = pop[4:7]
    root_low = np.array([tep1] + tep2.tolist())
    tip_up = pop[7:11]
    tep1 = -pop[7]
    tep2 = pop[11:14]
    tip_low = np.array([tep1] + tep2.tolist())
    root_x, root_y, c = CST_airfoil(root_up, root_low, number=num)
    if c <= 2:
        tip_x, tip_y, c = CST_airfoil(tip_up, tip_low, number=num)
        # picture_generate(tip_x, tip_y)
    if c > 2:
        print("geometry unreasonable!\n")
        flag = 1
    else:
        flag = 0
    return flag


def NACA_airfoil(m, p, t, number):
    # 老套路，生成前后缘加密的翼型的点
    x_sin = np.linspace(0, np.pi / 2, number + 1)
    x = 0.5 * np.sin(2 * x_sin - np.pi / 2) + 0.5
    x = np.array(x)
    y_c = np.zeros(len(x))
    y_t = np.zeros(len(x))
    for i in range(len(x)):
        if x[i] < p:
            y_c[i] = m * x[i] / (p ** 2) * (2 * p - x[i])
        elif x[i] >= p:
            y_c[i] = m * (1 - x[i]) / (1 - p) ** 2 * (1 - 2 * p + x[i])
        else:
            pass
    for i in range(len(x)):
        y_t[i] = 5 * t * (0.2969 * math.sqrt(x[i]) - 0.1260 * x[i] - 0.3516 * x[i] ** 2 + 0.2843 *
                          x[i] ** 3 - 0.1036 * x[i] ** 4)
    kappa = np.zeros(len(x) - 2)
    for i in range(0, len(x) - 2, 1):
        if x[i + 1] < p:
            kappa[i] = 2 * m * (p - x[i + 1]) / p ** 2
        elif x[i + 1] >= p:
            kappa[i] = 2 * m * (p - x[i + 1]) / (1 - p) ** 2
    x_low = np.zeros(len(x))
    x_up = np.zeros(len(x))
    y_up = np.zeros(len(x))
    y_low = np.zeros(len(x))
    x_up[len(x) - 1] = 1
    x_low[len(x) - 1] = 1

    for i in range(len(kappa)):
        x_up[i + 1] = x[i + 1] - y_t[i + 1] * (kappa[i] / np.sqrt(kappa[i] ** 2 + 1))
        x_low[i + 1] = x[i + 1] + y_t[i + 1] * (kappa[i] / np.sqrt(kappa[i] ** 2 + 1))
        y_up[i + 1] = y_c[i + 1] + y_t[i + 1] * (1 / np.sqrt(kappa[i] ** 2 + 1))
        y_low[i + 1] = y_c[i + 1] - y_t[i + 1] * (1 / np.sqrt(kappa[i] ** 2 + 1))

    x_up_reverse = x_up[::-1]
    y_up_reverse = y_up[::-1]
    # 合并上表面（倒置后）和下表面的数组
    x_combined = np.concatenate((x_up_reverse, x_low))
    y_combined = np.concatenate((y_up_reverse, y_low))
    # 显示一下生成的东西
    # picture_generate(x_combined, y_combined)
    # print(f"x = {x_combined}\ny = {y_combined}")
    return x_combined, y_combined


def NACA_airfoil_file_generate(pop, tip_airfoil, mid_airfoil, root_airfoil):
    num = 100
    k = pop[6]
    m1 = pop[0]
    m2 = pop[3]
    m3 = k * m1 + (1 - k) * m2
    p1 = pop[1]
    p2 = pop[4]
    p3 = k * p1 + (1 - k) * p2
    t1 = pop[2]
    t2 = pop[5]
    t3 = k * t1 + (1 - k) * t2
    a, b = NACA_airfoil(m1, p1, t1, number=num)
    writefile(a, b, root_airfoil)
    a, b = NACA_airfoil(m2, p2, t2, number=num)
    writefile(a, b, tip_airfoil)
    a, b = NACA_airfoil(m3, p3, t3, number=num)
    writefile(a, b, mid_airfoil)
    return 0


def airfoil_generate_test():
    # 生成文件路径
    # 准备做绘图的一些检验
    file_paths_generate.check_and_generate_path()
    # 准备开始生成
    # 设置保存图像的目录
    lhs = qmc.LatinHypercube(7)
    pop = lhs.random(100)
    save_dir = r"E:\_\result\agent-model\picture"
    for _ in range(len(pop)):
        for j in range(0, 4):
            pop[_][j] = pop[_][j] * 0.15 + 0.25
        for j in range(4, 7):
            pop[_][j] = pop[_][j] * 0.8 - 0.4
    for i in range(100):
        a = np.array(pop[i][0:4])
        b = np.array([-pop[i][0], pop[i][4], pop[i][5], pop[i][6]])
        xx, yy, flag = CST_airfoil(a, b, number=50)
        if flag <= 2:
            picture_generate_1(xx, yy)
            # 设置文件名
            filename = f"airfoil_{i + 1:03d}.png"  # 03d表示至少3位数，不足的前面补0
            filepath = os.path.join(save_dir, filename)
            # 保存图像
            plt.savefig(filepath)
            plt.close()  # 关闭当前图形窗口，释放资源
    return 0


# 找来的三点曲率计算方法 去掉了法向量输出
# [Cite] Pjer-zhang/PJCurvature
# modified by ylh
def curvature(x, y):
    """
    input  : the coordinate of the three point
    output : the curvature and norm direction
    refer to https://github.com/Pjer-zhang/PJCurvature for detail
    """
    t_a = LA.norm([x[1] - x[0], y[1] - y[0]])
    t_b = LA.norm([x[2] - x[1], y[2] - y[1]])

    M = np.array([
        [1, -t_a, t_a ** 2],
        [1, 0, 0],
        [1, t_b, t_b ** 2]
    ])

    a = np.matmul(LA.inv(M), x)
    b = np.matmul(LA.inv(M), y)
    value = (a[1] ** 2. + b[1] ** 2.) ** 1.5

    # 使用避免除零错误的差分估计法
    if value == 0:
        if x[1] - x[0] != 0 and x[2] - x[1] != 0:
            dy1 = (y[1] - y[0]) / (x[1] - x[0])
            dy2 = (y[2] - y[1]) / (x[2] - x[1])
            if x[2] - x[0] == 0:
                kappa = 0
            else:
                ddy = 2 * (dy2 - dy1) / (x[2] - x[0])
                kappa = np.fabs(ddy) / (np.power(1 + np.power((dy2 + dy1) / 2, 2), 1.5))
        else:
            kappa = 0
    # 使用多项式拟合方法
    else:
        kappa = np.fabs(2 * (a[2] * b[1] - b[2] * a[1])) / value
    return kappa

# #############################################################
# #################### 简单的demo与测试 #########################
# #############################################################
# xx, yy = NACA_airfoil(m=0.05, p=0.4, t=0.12, number=100)
# print(xx, yy)
# airfoil_file_generate(
#     [-0.018446094415062605, 0.2867014564571873, 0.058218320800093516, 0.006971852019190511, -0.03961873883600641,
#      0.01103651764617252, 0.6956134937401263], paths.tip_file, paths.mid_file, paths.root_file)

# arr = [0.179, 0.117, 0.173, 0.203, -0.095, -0.3, 0.167,
#        0.164, 0.3036, 0.1884, 0.2535, -0.0328, -0.0542, -0.068,
#        0.5]
# arr = np.array(arr)
# CST_airfoil_file_generate(arr, paths.tip_file, paths.mid_file, paths.root_file)
# CST_airfoil([0.179, 0.117, 0.173, 0.203], [-0.179, -0.095, -0.3, 0.167], number=50)
# while True:
#     # 初始化一个全零数组
#     arr = np.zeros(15)
#     # 设置第一个范围1-4和8-11的值
#     idx_1 = np.arange(0, 4)
#     idx_2 = np.arange(7, 11)
#     arr[np.concatenate((idx_1, idx_2))] = np.random.uniform(0.2, 0.4, size=len(np.concatenate((idx_1, idx_2))))
#     # 设置第二个范围5-7和12-14的值
#     idx_3 = np.arange(4, 7)
#     idx_4 = np.arange(11, 14)
#     arr[np.concatenate((idx_3, idx_4))] = np.random.uniform(-0.4, 0.4, size=len(np.concatenate((idx_3, idx_4))))
#     # 设置第15个元素的值
#     arr[14] = np.random.uniform(0, 1)
#     print(arr)
#     CST_airfoil_file_generate(arr, paths.tip_file, paths.mid_file, paths.root_file)
# aa = np.random.uniform(low=0.15, high=0.3, size=4)
# bb = -aa
# print(f"aa:{aa}\nbb:{bb}\n")
# test_x, test_y, test_flag = CST_airfoil(aa, bb,
#                                         number=100)
# picture_generate(test_x, -test_y)
# airfoil_generate_test()
