## main
主文件 用来对计算方法和搜索规模进行指定

例子：
Optimization_1.PSO_PCA_Optimization(5, 8)
调用PSO_PCA 方法进行优化 迭代步数5 个体数目8

## Optimization
**用于优化计算的主体**

### iteration_ctrl(evaluate, variance)
方均根收敛准则 判定适应度值是否趋同

### PSO_PCA_Optimization(iteration, population)
降维粒子群算法 输入两个参数 总迭代步数 和 个体数目
有几个内部参数

**rate**  

用来控制初始搜索范围的系数 表示迭代一定比例后开始进行收缩
例如rate = 0.5 当迭代步数超过了rate*iteration后更换搜索模式

**max_range**  

扰动范围 生成一个30维的向量是整个机体的参数化表达，其种群是在基本机翼上的叠加，例如 max_range = 0.08 表示叠加在基本翼型上的扰动范围是（-0.04, 0.04）

**solve_mode**

求解的方式 用来控制求解函数具体内容如下：
1. 0 全流程求解，控制焦点求解，力矩配平（攻角计算） 气动参数计算
2. 1 力矩配平（攻角计算），焦点代理估计，气动参数计算 （由于没算焦点 很可能返回错误 没啥用）
3. 2 攻角和焦点代理估计 气动参数计算
4. 3 零攻角 气动参数计算
5. 4 代数式代理模型（验证程序能否跑通，读写是否正常，算法是否收敛 结果秒出）

**population_file**
用于存储种群数据

**supervise_file**
用于监控
返回代数 时间 适应度

### GWO_PCA_Optimization(iteration, population)
降维灰狼算法

还没写（doge）

### GA_Optimization(iteration, population)
遗传算法

也还没写


## algorithm
优化算法的更新方式

### generate(population, max_range)
生成种群的初始值


### disturb_modify(person)
输入一个个体 返回叠加扰动后的翼型数据

### data_package(population, pop)
输入种群（扰动种群）
返回字典组（翼型CST数据）用于气动计算


### PSO_PCA_0(pop, pre_pop, p_best, evaluate, population, rate)
不区分搜索模式的PSO方法

**pop** 种群的扰动完整数据 

**p_best** 个体最佳记录 

**pre_pop** 上一次迭代的种群数据 

**population** 种群个体数目

**evaluate** 评估值数组

**rate** 更新速率（直接乘在搜索参数上乘上的值）

**n_component**  
降阶次数 例如n_component = 4 就是把30维度的数据降维成4维 降阶后次数高则保留信息多，由于数据压缩造成的损失小，可能导致收敛慢。
此外如果阶数过低可能压缩掉了太多信息，也可能导致不收敛，也无法返回可靠的结果。
如果将n_component置为30 PSO-PCA算法则退化为PSO方法

**inertia_eff**  惯性参数

按照自身速度更新的比率 设置高则收敛慢 但是搜索范围大 不容易过早收敛

**guide_eff**   全局寻优参数

按照全局最优值进行更新的比率 如果设置过高则收敛过快 反之不收敛

**self_eff**    自身寻优参数

向自身的历史最优值的更新比率，跟惯性参数有相似的地方，但是加入此项可能提升优化效果


### PSO_PCA_1(pop, pre_pop, p_best, evaluate, population, rate, iterations)

和上一个函数的区别是采用了不同阶段不同更新模式的方式

**rate** 
这里的rate和上面的不同 代表了更新模式的控制 在迭代步数到达总迭代步数*rate后更换搜索模式

**iterations**
输入的当前迭代步数 用来判定采用什么更新模式

### ratio_determine(population, iterations, ratio)

返回更新参数的函数


## solver
求解模块

### solve(data, flag)
求解的整合
此处的flag就是前面的**solve_mode**位于
**PSO_PCA_Optimization(iteration, population)**

### evaluate(cl, cdi, flag)
评估函数模块

**flag**
1. flag = 0 简单函数代理
2. flag = 1 升力系数/诱导阻力系数
3. flag = 2 接入稳态爬升模型 还没有写（主要要解决cd的计算不准确的问题）

## balancing 
配平模块
### cal_xcg(up_root, low_root, up_tip, low_tip, stag)
**用于气动焦点计算**

up_root, low_root, up_tip, low_tip 这些都是CST参数数组

stag 是CST的阶数 但是由于基础翼型的设置不能调整 已经置为7
这个参数的选择理由见下图（stag = 7实际上是8参数）：
![alt text](image.png)
**max_iterations** 最大迭代步数 
**var_value_ctrl** 数值变化控制
**epsilon**  更新步长控制

**焦点计算中正常判断停止有三个条件 任意满足其一就终止**

1. value <= var_value_ctrl cmy方均值变化小于控制值
2. iteration_gap <= epsilon 更新步长小于控制值
3. iterations >= max_iterations 超过迭代步数 

如果重心位置更新发散超过合理范围 终止配平返回报错给

**balance(flag, up_root, low_root, up_tip, low_tip, stag)**

**var_ctrl** 数值调整参数 由于均方根或者均方值都大于0 用双初值牛顿迭代可能发散 将函数下降得到和0的交点
![alt text](cd1f496613f97cc31c4926b703f979c.jpg)

**Xcg_0 = 0**  双初值法初始迭代位置 选择原因见上图
**Xcg_1 = 0.1**  

### var_cal(cmy_array)
用来计算均方值
如果要修改方均根或者其他方式需要修改
此外如果模型的精度不好 也需要修改

**var_value_ctrl** 数值变化控制 
**epsilon**  更新步长控制 
**var_ctrl**  在较小的值内 越大越稳定
特别是有翼梢小翼的模型 容易出现数值不稳定 这些限制条件需要放宽


### alpha_cal(up_root, low_root, up_tip, low_tip, stag, xcg)
攻角配平函数返回攻角

**value_ctrl** 是数值变化的控制 这个应该是单调函数 没有减去一个值来提高精度的问题
其他逻辑和**cal_xcg** 相同

### balance(flag, up_root, low_root, up_tip, low_tip, stag)
这是计算配平工况的整合 稳定裕度是8%

计算后返回攻角 重心 和执行情况（会写入文件balancing.txt）这个文件不会清零 多次运行会一直追加数据

错误类型有4种
1. 超过焦点计算最大步数 （可能没有发散，但是要求精度太苛刻 模型精度低）
2. 焦点计算发散 （跳出了合理值 模型有问题或者初始条件不适用）
3. 超过配平计算最大步数
4. 配平计算发散
**一旦出现错误工况将被设置为默认值并返回错误信息**
**在solver中评估值会置0** 
**同时跳过气动计算**


## vsp
**与Openvsp的交互部分用于进行几何生成和气动计算**

### create_Geom_0(root_up, root_low, tip_up, tip_low, stage)
几何生成 有翼梢小翼的模型 计算残差大 不可靠 需要调整
翼梢小翼的参数是确定的不作为优化变量

### create_Geom_1(root_up, root_low, tip_up, tip_low, stage)
几何生成 没有翼梢小翼的模型  计算残差相对可靠 

### vsp_aero_sweep(x_cg) 

读取已经生成的模型进行计算 选取1-5度攻角的CMy作为气动焦点计算检验标准（可以考虑设置的少一点 但是攻角不能大 不然好像不稳定）

**对于计算的高级设置**

**N_cpu** 调用cpu核数 检查所用电脑的cpu物理核数 可以全部占满

**Iter** 单个工况迭代步数 可以通过GUI计算后设置 应该只要算到残差不掉了就行了  
 
### vsp_aero(x_cg, aoa)

气动参数的计算 内部计算设置相同







