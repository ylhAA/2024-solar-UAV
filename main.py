# import Optimization_2
# # 这个是主文件，主要用来进行控制
# Optimization_2.PSO_PCA_Optimization(30, 30)
# 改进的配平组件 共求解900个点 每个点调用openvsp 分别求解sweep 五个点 aero 单个点共6个状态 求解5400次 预计时间小于10小时
# 可能存在提前退出，那是因为收敛到方差小于一定值 或者模型不合理，求解出来全是0 我用小样本目前还没出这个问题
# 此外，设置优化目标为诱导阻力系数最小，同时如果计算所得速度大于风洞实验值或者配平工况下升力系数小于估计值同时满足，则舍弃这个点的数据
# 基础是NASA-SC-0412 NACA-3412 upsidedown 已经扩大了搜索范围
import Optimization_3
import file_paths_generate
file_paths_generate.check_and_generate_path()
Optimization_3.PSO_PCA_Optimization(2, 4)
