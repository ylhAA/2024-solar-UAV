import numpy as np
import numpy.linalg as LA
import matplotlib.pyplot as plt
import math
# from scipy.stats import qmc
# import paths
# import file_paths_generate
import algorithm_3


# import vsp_4

# #############################################################
# ############## CST_Airfoil_Generate by CJY ##################
# #############################################################
# 用于输入CST参数并输出系列坐标点是CST函数的定义
# 类别函数
def class_function(N_1, N_2, x):
    # N_1 = 0.5
    # N_2 = 1.0
    x_n = len(x)
    C = np.empty((1, x_n))
    C = np.power(x, N_1) * np.power(1 - x, N_2)
    C = np.array(C)
    return C


# Bernstein多项式
def bernstein(n, x):
    # n 是bernstein多项式的阶数
    x_n = len(x)
    B = np.empty((x_n, n + 1))
    for i in range(0, x_n):
        for j in range(0, n + 1):
            B[i, j] = math.factorial(n) / (math.factorial(j) * math.factorial(n - j)) * x[i] ** j * (1 - x[i]) ** (
                    n - j)
    return B


# 形状函数
# a 拟合控制系数 B n阶bernstein多项式
def shape_function(a, B):
    x_n = B.shape[0]
    a_n = B.shape[1]
    for i in range(0, x_n):
        for j in range(0, a_n):
            B[i, j] = B[i, j] * a[j]

    S = np.empty((x_n, 1))
    for i in range(0, x_n):
        S[i, 0] = B[i, :].sum()

    return S


# C类别函数 S 形状函数
def CST_calculate(C, S, x):
    x_n = S.shape[0]
    y_CST = np.empty((x_n, 1))

    for i in range(0, x_n):
        y_CST[i, 0] = C[i] * S[i, 0]

    return y_CST


class CST_generator:
    def __init__(self, Bernstein_order, psi, N_1=0.5, N_2=1.0) -> None:
        # Bernstein_order为Bernstein多项式阶数,
        # 则参数数为2*(Bernstein_order+1)
        # N_1,N_2是形状函数的参数
        self.Bernstein_order = Bernstein_order
        self.psi = psi
        self.C = class_function(N_1, N_2, psi)
        self.B = bernstein(Bernstein_order, psi)

    def generate_curve(self, a_coe):
        """
        a_coe 是中弧线的生成参数 数目比bernstein阶数多1
        """
        S = shape_function(a_coe, self.B)
        y_CST = CST_calculate(self.C, S, self.psi)
        return y_CST


# #############################################################
# ################### airfoil_Generate  #######################
# #############################################################
# 传入中弧线系数用于生成翼型.dat数据 对于翼型厚度使用NACA厚度的处理
# up_cf上表面CST系数 low_cf下表面CST系数 order多项式阶数 number离散点个数
def airfoil_Generate(coe, order, number):
    # 前缘加密方法
    max_thickness = 0.1  # 最大相对厚度
    x_sin = np.linspace(0, np.pi / 2, number + 1)
    x = 0.5 * np.sin(2 * x_sin - np.pi / 2) + 0.5
    # 生成中弧线
    cst_generator = CST_generator(order, x)
    temp_c = cst_generator.generate_curve(coe)
    temp_c = temp_c.flatten()
    y_mid = np.zeros(len(temp_c - 1))
    x_mid = np.zeros(len(temp_c - 1))
    cos_theta = np.zeros(len(temp_c - 1))
    sin_theta = np.zeros(len(temp_c - 1))

    for i in range(len(temp_c) - 1):
        y_mid[i] = temp_c[i] + temp_c[i + 1]
        x_mid[i] = (x[i] + x[i + 1]) / 2
    plt.scatter(x_mid, y_mid)
    plt.show()
    for i in range(len(temp_c) - 1):
        cos_theta[i] = (x[i + 1] - x[i]) / np.sqrt((x[i + 1] - x[i]) ** 2 + (temp_c[i + 1] - temp_c[i]) ** 2)
        sin_theta[i] = (temp_c[i + 1] - temp_c[i]) / np.sqrt((x[i + 1] - x[i]) ** 2 + (temp_c[i + 1] - temp_c[i]) ** 2)
    thickness = np.zeros(len(temp_c) - 1)
    for i in range(len(temp_c) - 1):
        thickness[i] = (0.2969 * np.sqrt(x_mid[i]) - 0.1260 * x_mid[i] - 0.3516 * x_mid[i] ** 2 + 0.2843 *
                        x_mid[i] ** 3 - 0.1015 * x_mid[i] ** 4) / 0.2
    # 上下翼面参数的初始化
    thickness = thickness * max_thickness
    x_up = np.zeros(len(temp_c) - 1)
    x_low = np.zeros(len(temp_c) - 1)
    y_up = np.zeros(len(temp_c) - 1)
    y_low = np.zeros(len(temp_c) - 1)
    for i in range(len(x_up)):
        x_up[i] = x_mid[i] - thickness[i] * sin_theta[i]
        x_low[i] = x_mid[i] + thickness[i] * sin_theta[i]
        y_up[i] = y_mid[i] + thickness[i] * cos_theta[i]
        y_low[i] = y_mid[i] - thickness[i] * cos_theta[i]
    reversed_x_up = np.array(x_up[::-1])
    reversed_x_up = np.concatenate((np.array([1.0]), reversed_x_up, np.array([0.0])))
    x_combined = np.concatenate((reversed_x_up, x_low, [1]))
    reversed_y_up = np.array(y_up[::-1])
    reversed_y_up = np.concatenate((np.array([0.0]), reversed_y_up, np.array([0.0])))
    y_combined = np.concatenate((reversed_y_up, y_low, [0]))
    # 打印新数组以检查结果
    # print(x_combined)
    # print(y_combined)
    # plt.scatter(x_combined, y_combined)
    # plt.show()

    curve, thick, kappa = Stress_Check(x_low, x_up, y_low, y_up)
    # # 创建包含两个子图的图形
    # fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))
    # # 绘制上下表面曲线图
    # ax1.plot(x_up, y_up, label='Upper Surface')
    # ax1.plot(x_low, y_low, label='Lower Surface')
    # ax1.set_xlabel('x')
    # ax1.set_ylabel('y')
    # ax1.set_title('CST Airfoil')
    # ax1.legend()
    # ax1.grid(True)
    #
    # # 绘制上表面曲率分布
    # ax2.plot(x_up[1:-1], kappa, label='Airfoil kappa')
    # ax2.set_xlabel('x')
    # ax2.set_ylabel('kappa')
    # ax2.set_title('kappa distribution')
    # ax2.legend()
    # ax2.grid(True)
    # # 调整子图之间的间距
    # plt.tight_layout()
    # # 显示图形
    # plt.show()

    # 返回了标签值用于判定是否需要重新生成来满足低应力铺板条件
    return x_combined, y_combined, curve, thick


# #############################################################
# ##################### Geom_Generate #########################
# #############################################################
# 用于输入一组数据进行几何生成
# 生成文件后写入翼型文件共与vsp交互的vsp_4 gemo读取
# 返回生成条件是否满足要求
def Geom_Generate(pop, tip_airfoil_path, mid_airfoil_path, root_airfoil_path):
    # 内部参数设置
    bernstein_order = 3
    num = 100
    root, mid, tip = algorithm_3.pop_modify(pop)
    a, b, curve_1, thick_1 = airfoil_Generate(root, bernstein_order, num)
    writefile(a, b, root_airfoil_path)
    a, b, curve_2, thick_2 = airfoil_Generate(tip, bernstein_order, num)
    writefile(a, b, tip_airfoil_path)
    a, b, curve_3, thick_3 = airfoil_Generate(mid, bernstein_order, num)
    writefile(a, b, mid_airfoil_path)
    print("all airfoil generate successfully!")
    curve = [curve_1, curve_2, curve_3]
    thick = [thick_1, thick_2, thick_3]
    return curve, thick


# #############################################################
# ##################### Geom_Generate #########################
# #############################################################
# 传入上下表面的点数据，计算曲率，并返回范围内的最小厚度和最大曲率
def Stress_Check(x_low, x_up, y_low, y_up):
    # 输入标准弦长 前缘和后缘不贴片的距离
    chord = 0.3
    le_ignore_length = 2e-2
    te_ignore_length = 6e-3
    # 用来标记曲率条件
    curvature_max = 0
    # 用来标记厚度条件
    thickness_min = 1
    # deg35 曲率上限
    curve_threshold = 4.88
    # 后的设置下限
    thick_threshold = 2e-3
    # 归一化处理
    le_ignore_length = le_ignore_length / chord
    te_ignore_length = 1 - te_ignore_length / chord
    # 找到对应的索引
    flag1 = 0
    flag2 = 0
    # 正常取到前缘限制的前2个点的序号 否则返回第一个值的序号
    start_id = 0
    # 正常取到后缘限制的第一个点的序号 如果到达倒数第三个点之后则返回倒数第三个点（所求范围一定是扩大的）
    end_id = 0
    n_points = len(x_up)

    for i in range(n_points):
        if x_up[i] > le_ignore_length and flag1 == 0:
            if i <= 1:
                start_id = 0
                flag1 = 1
            else:
                start_id = i - 2
                flag1 = 1
        elif x_up[i] > te_ignore_length and flag2 == 0:
            if i >= n_points - 3:
                end_id = n_points - 3
                flag2 = 1
            else:
                end_id = i
                flag2 = 1
        if flag1 and flag2:
            break
    kappa = np.zeros(n_points - 2)
    for i in range(n_points - 2):
        kappa[i] = curvature([x_up[i], x_up[i + 1], x_up[i + 2]], [y_up[i], y_up[i + 1], y_up[i + 2]])
    # 遍历在索引范围内的kappa 检测其曲率是否超标
    for i in range(start_id, end_id, 1):
        if curvature_max < kappa[i] and kappa[i] > curve_threshold:
            curvature_max = kappa[i]
        low = airfoil_line(x_low, y_low, x_up[i])
        up = y_up[i]
        if up - low < thickness_min and up - low < thick_threshold:
            thickness_min = up - low

    return curvature_max, thickness_min, kappa


def airfoil_line(x, y, point):
    point_id = 0
    if point < x[0] or point > x[-1]:
        value = -1
        print("error\n")
    else:
        for i in range(len(x)):
            if point > x[i]:
                point_id = i - 1
            else:
                pass
        value = y[point_id] + (y[point_id + 1] - y[point_id]) / (x[point_id + 1] - x[point_id]) * (point - x[point_id])
    return value


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
# ##################### read and write  #######################
# #############################################################
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


# #############################################################
# ##################### 写入文件writeFile ######################
# #############################################################
# 用于输入翼型数组并把文件写入outputPath 文件路径+文件名
def writefile(x, y, outputPath):
    with open(outputPath, 'w') as file:
        for i_line in range(len(x)):
            line = f'   {x[i_line]:.5f}\t   {y[i_line]:.5f}\n'
            file.write(line)
    print("airfoil file output successfully\n")
    return 0


# #############################################################
# #################### 简单的demo与测试 #########################
# #############################################################
# para_coe = [0.04, -0.04, -0.04, 0, 0.03, -0.03, 0.03, -0.03, 0.3]
# Geom_Generate(para_coe, paths.tip_file, paths.mid_file, paths.root_file)
# file_paths_generate.check_and_generate_path()
# # 准备开始生成
# dim = 4  # 参数空间的大小
# max_population = 2
# data_path = paths.dictionary_file
# lhs = qmc.LatinHypercube(dim)
# people = lhs.random(max_population)
# for ii in range(max_population):
#     for jj in range(4):
#         people[ii][jj] = people[ii][jj] * 0.08 - 0.04
#
# for ii in range(max_population):
#     print(people[ii])
#     xx, yy, _, _ = airfoil_Generate(people[ii], order=3, number=100)
#     plt.plot(xx, yy)
#     plt.show()
# Geom_Generate(np.zeros(9), paths.tip_file, paths.mid_file, paths.root_file)
aa, bb, _, _ = airfoil_Generate([0.01962847, 0.02835683, 0.02994549, 0.02495018], order=3, number=100)
plt.scatter(aa, bb)
plt.show()
