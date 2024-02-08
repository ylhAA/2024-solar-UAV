import numpy as np
# import matplotlib.pyplot as plt
import numpy.linalg as LA
import math
import algorithm_1


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
    def __init__(self, Bernstein_order, psi_up, psi_low, N_1=0.5, N_2=1.0) -> None:
        # Bernstein_order为Bernstein多项式阶数,
        # 则参数数为2*(Bernstein_order+1)
        #
        # psi是坐标,一般为`np.linspace(0, 1, N)`
        #
        # N_1,N_2是形状函数的参数
        self.Bernstein_order = Bernstein_order

        self.psi_up = psi_up
        self.psi_low = psi_low

        self.C_up = class_function(N_1, N_2, psi_up)
        self.B_up = bernstein(Bernstein_order, psi_up)
        self.C_low = class_function(N_1, N_2, psi_low)
        self.B_low = bernstein(Bernstein_order, psi_low)

    def generate_airfoil(self, a_up, a_low):
        """
        a_up是生成上表面的参数,shape为[Bernstein_order+1, ];

        a_low是生成下表面的参数,shape为[Bernstein_order+1, ];

         """
        S_up = shape_function(a_up, self.B_up)
        S_low = shape_function(a_low, self.B_low)

        y_CST_up = CST_calculate(self.C_up, S_up, self.psi_up)
        y_CST_low = CST_calculate(self.C_low, S_low, self.psi_low)

        return y_CST_up, y_CST_low


# #############################################################
# ################### airfoil_Generate  #######################
# #############################################################
# 传入上下翼面系数用于生成翼型.dat数据
# up_cf上表面CST系数 low_cf下表面CST系数 order多项式阶数 number离散点个数
def airfoil_Generate(up_cf, low_cf, order, number):
    # 前缘加密方法
    x_sin = np.linspace(0, np.pi / 2, number + 1)
    x_up = 0.5 * np.sin(2 * x_sin - np.pi / 2) + 0.5
    x_low = x_up

    # 预先生成初始系数
    cst_generator = CST_generator(order, x_up, x_low)
    temp_up, temp_low = cst_generator.generate_airfoil(up_cf, low_cf)
    Y_CST_up = temp_up.reshape(-1)
    Y_CST_low = temp_low.reshape(-1)
    # print(x_up)
    # print(Y_CST_up)
    # 检验是否满足曲率条件
    flag, kappa = Stress_Check(x_up, Y_CST_up)
    # print(flag)

    # 数据规整便于输入处理
    x_up_reversed = x_up[::-1]
    y_up_reversed = Y_CST_up[::-1]
    x_combined = np.concatenate((x_up_reversed, x_low))
    y_combined = np.concatenate((y_up_reversed, Y_CST_low))

    # # 创建包含两个子图的图形
    # fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))
    # # 绘制上下表面曲线图
    # ax1.plot(x_up, Y_CST_up, label='Upper Surface')
    # ax1.plot(x_low, Y_CST_low, label='Lower Surface')
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
    return x_combined, y_combined, flag


# #############################################################
# ##################### Geom_Generate #########################
# #############################################################
# 用于输入一组数据进行几何生成
# 生成文件后写入翼型文件共与vsp交互的vsp_4 gemo读取
# 返回生成条件是否满足要求
def Geom_Generate(pop, tip_airfoil_path, root_airfoil_path):
    # 内部参数设置
    bernstein_order = 7
    num = 100
    inner_max_range = 0.1  # 如果不能成功配平则重置个体的比率范围
    flag1 = 1  # 处理是否符合几何条件 根部变量
    flag2 = 1  # 处理是否符合几何条件 稍部变量
    iterations = -1  # 重置次数
    max_iteration = 5  # 最大重置次数
    while (flag1 or flag2) and iterations <= max_iteration:  # flag1 flag2 其中至少有一个是1 并且iterations小于最大步数
        # 添加扰动 完成扰动到翼型的转化
        if flag1:
            root_up, root_low, _, _ = algorithm_1.disturb_modify(pop)
            # 根部写入
            a, b, flag = airfoil_Generate(root_up, root_low, bernstein_order, num)
            # 不满足条件重新生成
            if flag != 0:
                print("root airfoil reset")
                pop[:8] = np.random.uniform(-inner_max_range / 2, inner_max_range / 2, 8)
            # 满足条件写入根部翼型文件
            else:
                writefile(a, b, root_airfoil_path)
                flag1 = 0
        else:
            pass

        if flag2:
            _, _, tip_up, tip_low = algorithm_1.disturb_modify(pop)
            a, b, flag = airfoil_Generate(tip_up, tip_low, bernstein_order, num)
            if flag != 0:
                print("tip airfoil reset")
                pop[15:23] = np.random.uniform(-inner_max_range / 2, inner_max_range / 2, 8)
            else:
                # 稍部写入
                writefile(a, b, tip_airfoil_path)
                flag2 = 0
        # 步数增加1
        iterations += 1
    # 如果检测到超出了最大的重置数目还没有满足条件直接置为默认参数 （经过检验默认参数满足几何约束 且在气动计算中能正常输出结果）
    if iterations > max_iteration:
        pop = np.zeros(30)
    else:
        pass
    # 需要考虑没法处理重新进行生成的问题
    # 如果iterations大于0则表明发生了重置 需要在外部更新种群
    return pop, iterations


# #############################################################
# ##################### Geom_Generate #########################
# #############################################################
def Stress_Check(x, y):
    # 输入标准弦长 前缘和后缘不贴片的距离
    chord = 0.3
    le_ignore_length = 2e-2
    te_ignore_length = 6e-3
    flag = 0
    threshold = 2.85  # deg20 曲率上限
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
    n_points = len(x)

    for i in range(n_points):
        if x[i] > le_ignore_length and flag1 == 0:
            if i <= 1:
                start_id = 0
                flag1 = 1
            else:
                start_id = i - 2
                flag1 = 1
        elif x[i] > te_ignore_length and flag2 == 0:
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
        kappa[i] = curvature([x[i], x[i + 1], x[i + 2]], [y[i], y[i + 1], y[i + 2]])
    # 遍历在索引范围内的kappa 检测其曲率是否超标
    for i in range(start_id, end_id, 1):
        if kappa[i] > threshold:
            flag = 1
        else:
            pass
    return flag, kappa


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

# filename = "D:\\aircraft design competition\\24solar\\models\\v0\\v0.0\\NACA3412.dat"
# a, b = readFile(inputPath=filename)
# filename = ("D:\\aircraft design competition\\24solar\\design_model\\whole_wing_optimization"
#             "\\python\\airfoil_file\\output_test.dat")
# writefile(a, b, filename)

# root_up = [0.20027, 0.06942, 0.26365, 0.01476, 0.27234, 0.14104, 0.17256, 0.20330]
# root_low = [-0.20027, -0.08095, -0.21768, -0.12756, -0.14138, -0.24818, 0.04641, 0.16772]
# tip_up = [0.15727, 0.04782, 0.19466, -0.10213, 0.22039, -0.03441, 0.06494, 0.06763]
# tip_low = [-0.15727, -0.33244, -0.11350, -0.26489, -0.29933, -0.14698, -0.27400, -0.25044]

# 创建CST_generator实例
# 成功创建VSP3参数化几何文件
# x_sin = linspace(0,pi/2,num+1);
# x = 0.5*sin(2*x_sin-pi/2)+0.5;

# 验证 vsp_4.create_Geom_2 Geom_Generate 工作正常

# tip = ("D:\\aircraft design competition\\24solar\\design_model\\whole_wing_optimization\\python"
#        "\\airfoil_file\\root.dat")
# root = ("D:\\aircraft design competition\\24solar\\design_model\\whole_wing_optimization\\python"
#         "\\airfoil_file\\tip.dat")
# test_pop = np.zeros(30)
# test_pop,test_iteration = Geom_Generate(test_pop, tip, root)
# print(test_pop,test_iteration)
# vsp_4.create_Geom_2(tip, root)

# # 曲率函数的使用与验证
# # 存在除0错误
# # 改进一下
