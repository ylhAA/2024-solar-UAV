import openvsp as vsp


# #############################################################
# ##################### 初始化OpenVSP模型########################
# #############################################################
def create_Geom():  # input A1[],A2[]
    vsp.VSPCheckSetup()
    vsp.VSPRenew()

    # 添加一个主翼
    main_wing = vsp.AddGeom("WING")
    vsp.InsertXSec(main_wing, 1, vsp.XS_GENERAL_FUSE)

    # 更改主翼参数
    vsp.SetParmVal(main_wing, "SectTess_U", "XSec_1", 4)  # 内段弦长方向划分
    vsp.SetParmVal(main_wing, "Tess_W", "Shape", 30)  # 用来调整展长方向划分
    vsp.SetParmVal(main_wing, "Root_Chord", "XSec_1", 0.31800)
    vsp.SetParmVal(main_wing, "Tip_Chord", "XSec_1", 0.30000)
    vsp.SetParmVal(main_wing, "Span", "XSec_1", 0.05000)
    vsp.SetParmVal(main_wing, "Area", "XSec_1", 0.01545)
    vsp.SetParmVal(main_wing, "Sweep", "XSec_1", 20)

    vsp.Update()

    vsp.SetParmVal(main_wing, "SectTess_U", "XSec_2", 25)  # 外段弦长方向划分 注意一定要跟 xsec_2自己放在一起不然小心报错
    vsp.SetParmVal(main_wing, "Root_Chord", "XSec_2", 0.30000)
    vsp.SetParmVal(main_wing, "Tip_Chord", "XSec_2", 0.30000)
    vsp.SetParmVal(main_wing, "Span", "XSec_2", 1.22000)
    vsp.SetParmVal(main_wing, "Area", "XSec_2", 0.36600)
    vsp.SetParmVal(main_wing, "Sweep", "XSec_2", 20)

    vsp.Update()

    surf0 = vsp.GetXSecSurf(main_wing, 0)
    vsp.ChangeXSecShape(surf0, 0, vsp.XS_CST_AIRFOIL)
    xsec_0 = vsp.GetXSec(surf0, 0)
    vsp.Update()
    vsp.SetUpperCST(xsec_0, 3, [0.11, 0.22, 0.33, 0.21])  # A1[1:]
    vsp.SetLowerCST(xsec_0, 3, [0, -0.112, -0.131, -0.113])  # A1[2:]

    vsp.ChangeXSecShape(surf0, 1, vsp.XS_CST_AIRFOIL)
    xsec_1 = vsp.GetXSec(surf0, 1)
    vsp.Update()
    vsp.SetUpperCST(xsec_1, 3, [0.11, 0.22, 0.33, 0.21])  # A1[1:]
    vsp.SetLowerCST(xsec_1, 3, [0, -0.112, -0.131, -0.113])  # A1[2:]

    vsp.ChangeXSecShape(surf0, 2, vsp.XS_CST_AIRFOIL)
    xsec_2 = vsp.GetXSec(surf0, 2)
    vsp.Update()
    vsp.SetUpperCST(xsec_2, 3, [0.11, 0.22, 0.33, 0.21])  # A2[1:]
    vsp.SetLowerCST(xsec_2, 3, [0, -0.112, -0.131, -0.113])  # A2[2:]
    vsp.Update()

    # 添加翼梢小翼
    wing_let_1 = vsp.AddGeom("WING")
    # 更改翼梢小翼参数
    vsp.SetParmVal(wing_let_1, "Root_Chord", "XSec_1", 0.36000)
    vsp.SetParmVal(wing_let_1, "Tip_Chord", "XSec_1", 0.24500)
    vsp.SetParmVal(wing_let_1, "Span", "XSec_1", 0.11500)
    vsp.SetParmVal(wing_let_1, "Sweep", "XSec_1", 45)
    vsp.SetParmVal(wing_let_1, "X_Rel_Location", "XForm", 0.562)
    vsp.SetParmVal(wing_let_1, "Y_Rel_Location", "XForm", 1.275)
    vsp.SetParmVal(wing_let_1, "X_Rel_Rotation", "XForm", 90)
    vsp.Update()

    # 设置截面为椭圆形 如果缺省就是 NACA0010
    surf1 = vsp.GetXSecSurf(wing_let_1, 0)
    vsp.ChangeXSecShape(surf1, 0, vsp.XS_ELLIPSE)
    vsp.ChangeXSecShape(surf1, 1, vsp.XS_ELLIPSE)
    vsp.Update()

    wing_let_2 = vsp.AddGeom("WING")
    # 更改翼梢小翼参数
    vsp.SetParmVal(wing_let_2, "Root_Chord", "XSec_1", 0.36000)
    vsp.SetParmVal(wing_let_2, "Tip_Chord", "XSec_1", 0.24500)
    vsp.SetParmVal(wing_let_2, "Span", "XSec_1", 0.11500)
    vsp.SetParmVal(wing_let_2, "Sweep", "XSec_1", 45)
    vsp.SetParmVal(wing_let_2, "X_Rel_Location", "XForm", 0.562)
    vsp.SetParmVal(wing_let_2, "Y_Rel_Location", "XForm", 1.275)
    vsp.SetParmVal(wing_let_2, "X_Rel_Rotation", "XForm", -90)
    vsp.Update()

    # 设置截面为椭圆形 如果缺省就是 NACA0010
    surf2 = vsp.GetXSecSurf(wing_let_2, 0)
    vsp.ChangeXSecShape(surf2, 0, vsp.XS_ELLIPSE)
    vsp.ChangeXSecShape(surf2, 1, vsp.XS_ELLIPSE)
    vsp.Update()
    # 保存文件
    file_name = "D:\\aircraft design competition\\24solar\\design_model\\whole_wing_optimization\\vsp\\test2.vsp3"
    vsp.WriteVSPFile(file_name, vsp.SET_ALL)

    print("Geom Generate COMPLETE")


# #############################################################
# ####################vsp-aero调用计算##########################
# #############################################################

def vsp_aero():
    # 检查和清除否则模型会叠加干扰后续模型的建立
    vsp.VSPCheckSetup()
    vsp.VSPRenew()

    # 几何文件写入
    filename_vsp_aero_ana = ("D:\\aircraft design competition\\24solar\\design_model"
                             "\\whole_wing_optimization\\vsp\\test2.vsp3")
    # 这里是为了调用几何
    vsp.ReadVSPFile(filename_vsp_aero_ana)

    # 分析文件命名
    comp_geom = "VSPAEROComputeGeometry"
    print(comp_geom)

    # 设置defaults
    vsp.SetAnalysisInputDefaults(comp_geom)
    analysis_name_results = vsp.ExecAnalysis(comp_geom)  # 返回的是一个ID
    print("The Geom Result:\n", analysis_name_results)
    # 把所有的分析模式全部输出来
    for analysis in vsp.ListAnalysis():
        print(analysis)

    # 分析方法
    analysis_name = "VSPAEROSweep"  # 找到一个适合的求解器 不要用single point
    vsp.SetIntAnalysisInput(comp_geom, "AnalysisMethod", (1, vsp.VORTEX_LATTICE))
    # 打印该求解器所有可以更改的参数
    print("VSPAEROSweep 所有可选参数\n")
    vsp.PrintAnalysisInputs(analysis_name)

    # 设置参考翼面

    # 手动设置 来自GUI总体参数确定后的 From Model
    S_ref = [0.763]
    vsp.SetDoubleAnalysisInput(analysis_name, "Sref", S_ref, 0)
    b_ref = [2.540]
    vsp.SetDoubleAnalysisInput(analysis_name, "bref", b_ref, 0)
    c_ref = [0.304]
    vsp.SetDoubleAnalysisInput(analysis_name, "cref", c_ref, 0)

    # 自动计算设置 (暂时不成功)
    # ref_flag = [0]
    # vsp.SetIntAnalysisInput(analysis_name, "RefFlag", ref_flag, 0)
    vsp.Update()

    # 设置来流参数
    mach_speed = [0.02053]  # 7.00m/s altitude 0m
    vsp.SetDoubleAnalysisInput(analysis_name, "MachStart", mach_speed, 0)
    machNpts = [1]
    vsp.SetIntAnalysisInput(analysis_name, "MachNpts", machNpts, 0)
    alpha = [0.0]
    vsp.SetDoubleAnalysisInput(analysis_name, "AlphaStart", alpha, 0)
    alphaNpts = [1]
    vsp.SetIntAnalysisInput(analysis_name, "AlphaNpts", alphaNpts, 0)
    vsp.Update()
    rho = [1.225]  # 近地面大气密度参数
    vsp.SetDoubleAnalysisInput(analysis_name, "Rho", rho, 0)
    Re = [143651]  # 近地面 7m/s 参考长度为弦长 0.30m
    vsp.SetDoubleAnalysisInput(analysis_name, "ReCref", Re, 0)

    # 设置重心
    xcg = [0.2]
    vsp.SetDoubleAnalysisInput(analysis_name, "Xcg", xcg, 0)

    # 高级设置
    N_cpu = [6]
    vsp.SetIntAnalysisInput(analysis_name, "NCPU", N_cpu, 0)
    Iter = [10]
    vsp.SetIntAnalysisInput(analysis_name, "WakeNumIter", Iter, 0)
    vsp.Update()
    print("analysis parameter modified\n COMPLETED\n")
    print("VSPAEROSweep 参数输入结果\n")
    vsp.PrintAnalysisInputs(analysis_name)

    # #############################################################
    # 进行中间的保存尝试 观察vsp3是否正常
    file_name = "D:\\aircraft design competition\\24solar\\design_model\\whole_wing_optimization\\vsp\\test2.vsp3"
    vsp.WriteVSPFile(file_name, vsp.SET_ALL)
    # #############################################################

    # 开始计算
    # #############################################################
    print("\tExecution...")
    test = vsp.ExecAnalysis(analysis_name)
    print("COMPLETE")
    # #############################################################

    # 后处理
    # #############################################################
    # 返回一个列表里面是所有可用结果的名称
    results = vsp.GetAllResultsNames()
    # 查找point的ID并赋值给point_id
    point_id = vsp.FindResultsID("point")
    # 一样的也是一个查找
    polar_id = vsp.FindResultsID("VSPAERO_Polar")

    # 用来了解可用的结果集合和数据的名称（全部打印出来）
    for result in results:
        data_names = vsp.GetAllDataNames(vsp.FindResultsID(result))
        for data_name in data_names:
            print(f"{result} > {data_name}")

    # 用来统计CL的数据量并打印第一个
    print("num of data in 'VSPAERO_Polar > CL': ", vsp.GetNumData(polar_id, "CL"))
    cl = vsp.GetDoubleResults(polar_id, "CL", 0)
    print(f"CL = {cl}\n")
    # 这里的polar_id 赋值的是"VSPAERO_Polar" 也就是结果组的一个值 引用了结果组的L_D的部分
    print("num of data in 'VSPAERO_Polar > L_D': ", vsp.GetNumData(polar_id, "L_D"))
    l_d = vsp.GetDoubleResults(polar_id, "L_D", 0)
    print(f"L/D = {l_d}\n")
    print("num of data in 'area': ", vsp.GetNumData(point_id, "area"))
    area = vsp.GetDoubleResults(point_id, "area", 0)
    print(f"area = {area}\n")
    vsp.WriteResultsCSVFile(test,
                            "D:\\aircraft design competition\\24solar\\design_model\\whole_wing_optimization\\vsp"
                            "\\result.csv")


create_Geom()
vsp_aero()
