import openvsp as vsp


# 最初的一种生成方法 外形总体参数满足 翼型不满足
def create_0():
    # 设置机翼的展长和弦长数据
    span = 1.27  # 展长（单位：米）
    root_chord = 0.318  # 根部弦长（单位：米）
    tip_chord = 0.3  # 稍部弦长（单位：米）

    # 机翼几何参数
    wing = vsp.AddGeom("WING")
    vsp.SetParmVal(wing, "Span", "XSec_1", span)
    vsp.SetParmVal(wing, "Root_Chord", "XSec_1", root_chord)
    vsp.SetParmVal(wing, "Tip_Chord", "XSec_1", tip_chord)
    vsp.SetParmVal(wing, "Sweep", "XSec_1", 20.0)  # 后掠角（单位：度）
    vsp.Update()
    # 把翼型给赋值上去
    vsp.SetParmVal(wing, "Camber", "XSecCurve_0", 0.02)
    vsp.Update()

    # vsp3文件保存
    file_name = "D:\\aircraft design competition\\24solar\\design_model\\whole_wing_optimization\\vsp\\test0.vsp3"
    vsp.WriteVSPFile(file_name, vsp.SET_ALL)


# #######################################################
# 满足CST翼型生成
def create_1():
    # 设置机翼的展长和弦长数据
    span = 1.27  # 展长（单位：米）
    root_chord = 0.318  # 根部弦长（单位：米）
    tip_chord = 0.3  # 稍部弦长（单位：米）

    # 机翼几何参数
    wing = vsp.AddGeom("WING")
    vsp.SetParmVal(wing, "Span", "XSec_1", span)
    vsp.SetParmVal(wing, "Root_Chord", "XSec_1", root_chord)
    vsp.SetParmVal(wing, "Tip_Chord", "XSec_1", tip_chord)
    vsp.SetParmVal(wing, "Sweep", "XSec_1", 20.0)  # 后掠角（单位：度）
    vsp.SetParmVal(wing, "Tess_W", "Shape", 20)  # 用来调整展长方向划分
    vsp.SetParmVal(wing, "Tess_U", "Shape", 20)  # 弦长方向划分但是没用
    vsp.SetParmVal(wing, "SectTess_U", "XSec_1", 20)  # 弦长方向划分
    vsp.Update()
    # 把翼型给赋值上去
    # 特别注意surf是横截曲面，是整个面包含了众多横截面 而xsec才是横截面
    surf0 = vsp.GetXSecSurf(wing, 0)
    vsp.ChangeXSecShape(surf0, 0, vsp.XS_CST_AIRFOIL)
    vsp.ChangeXSecShape(surf0, 1, vsp.XS_CST_AIRFOIL)
    xsec_0 = vsp.GetXSec(surf0, 0)
    xsec_1 = vsp.GetXSec(surf0, 1)
    vsp.SetUpperCST(xsec_0, 3, [0.11, 0.22, 0.33, 0.21])
    vsp.SetLowerCST(xsec_0, 3, [0, -0.112, -0.131, -0.113])
    vsp.SetUpperCST(xsec_1, 3, [0.11, 0.22, 0.33, 0.21])
    vsp.SetLowerCST(xsec_1, 3, [0, -0.112, -0.131, -0.113])

    vsp.Update()

    # 调整参数 这个部分暂时搞不定
    # vsp3文件保存
    file_name = "D:\\aircraft design competition\\24solar\\design_model\\whole_wing_optimization\\vsp\\test1.vsp3"
    vsp.WriteVSPFile(file_name, vsp.SET_ALL)  # 完成文件的保存


# ##########################################################################3
# 几何生成函数2
def create_2():  # input A1[],A2[]
    # 初始化OpenVSP模型
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
    vsp.SetParmVal(wing_let_1, "Y_Rel_Location", "XForm", 1.271)
    vsp.SetParmVal(wing_let_1, "X_Rel_Rotation", "XForm", 90)

    vsp.Update()

    wing_let_2 = vsp.AddGeom("WING")
    # 更改翼梢小翼参数
    vsp.SetParmVal(wing_let_2, "Root_Chord", "XSec_1", 0.36000)
    vsp.SetParmVal(wing_let_2, "Tip_Chord", "XSec_1", 0.24500)
    vsp.SetParmVal(wing_let_2, "Span", "XSec_1", 0.11500)
    vsp.SetParmVal(wing_let_2, "Sweep", "XSec_1", 45)
    vsp.SetParmVal(wing_let_2, "X_Rel_Location", "XForm", 0.562)
    vsp.SetParmVal(wing_let_2, "Y_Rel_Location", "XForm", 1.271)
    vsp.SetParmVal(wing_let_2, "X_Rel_Rotation", "XForm", -90)

    # 保存文件
    file_name = "D:\\aircraft design competition\\24solar\\design_model\\whole_wing_optimization\\vsp\\test2.vsp3"
    vsp.WriteVSPFile(file_name, vsp.SET_ALL)

    print("Geom Generate COMPLETE")


# ####################################################################
# vsp-aero调用计算
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
    # 一共是这些东西然后我们选择一个
    # BladeElement
    # CfdMeshAnalysis
    # CompGeom
    # CpSlicer
    # DegenGeom
    # EmintonLord
    # FeaMeshAnalysis
    # MassProp
    # ParasiteDrag
    # PlanarSlice
    # Projection
    # SurfaceIntersection
    # SurfacePatches
    # VSPAEROComputeGeometry
    # VSPAERODegenGeom
    # VSPAEROReadPreviousAnalysis
    # VSPAEROSinglePoint
    # VSPAEROSweep
    # WaveDrag

    # 分析方法
    analysis_name = "VSPAEROSweep"  # 找到一个适合的求解器 不要用single point
    vsp.SetIntAnalysisInput(comp_geom, "AnalysisMethod", (1, vsp.VORTEX_LATTICE))
    # 打印该求解器所有可以更改的参数
    print("VSPAEROSweep 所有可选参数\n")
    vsp.PrintAnalysisInputs(analysis_name)

    # 所有VSPAEROSweep 求解器可以使用的参数
    # #######################################################
    # [input_name][type]       [  # ]	[current values-->]
    #     2DFEMFlag                     integer      	1	0
    # ActuatorDiskFlag              integer      	1	0
    # AlphaEnd                      double       	1	10.000000
    # AlphaNpts                     integer      	1	3
    # AlphaStart                    double       	1	1.000000
    # AlternateInputFor
    # g      integer      	1	0
    # AnalysisMethod                integer      	1	0
    # AutoTimeNumRevs               integer      	1	5
    # AutoTimeStepFlag              integer      	1	1
    # BetaEnd                       double       	1	0.000000
    # BetaNpts                      integer      	1	1
    # BetaStart                     double       	1	0.000000
    # CGGeomSet                     integer      	1	0
    # Clmax                         double       	1	-1.000000
    # ClmaxToggle                   integer      	1	0
    # FarDist                       double       	1	-1.000000
    # FarDistToggle                 integer      	1	0
    # FixedWakeFlag                 integer      	1	0
    # FromSteadyState               integer      	1	0
    # GeomSet                       integer      	1	0
    # GroundEffect                  double       	1	-1.000000
    # GroundEffectToggle            integer      	1	0
    # HoverRamp                     double       	1	0.000000
    # HoverRampFlag                 integer      	1	0
    # KTCorrection                  integer      	1	0
    # MachEnd                       double       	1	0.000000
    # MachNpts                      integer      	1	1
    # MachStart                     double       	1	0.000000
    # Machref                       double       	1	0.300000
    # ManualVrefFlag                integer      	1	0
    # MassSliceDir                  integer      	1	0
    # MaxTurnAngle                  double       	1	-1.000000
    # MaxTurnToggle                 integer      	1	0
    # NCPU                          integer      	1	4
    # NoiseCalcFlag                 integer      	1	0
    # NoiseCalcType                 integer      	1	0
    # NoiseUnits                    integer      	1	0
    # NumMassSlice                  integer      	1	10
    # NumTimeSteps                  integer      	1	25
    # NumWakeNodes                  integer      	1	64
    # Precondition                  integer      	1	0
    # ReCref                        double       	1	10000000.000000
    # ReCrefEnd                     double       	1	20000000.000000
    # ReCrefNpts                    integer      	1	1
    # RedirectFile                  string       	1	stdout
    # RefFlag                       integer      	1	0
    # Rho                           double       	1	0.002377
    # RotateBladesFlag              integer      	1	0
    # Sref                          double       	1	100.000000
    # Symmetry                      integer      	1	0
    # TimeStepSize                  double       	1	0.001000
    # UnsteadyType                  integer      	1	0
    # Vinf                          double       	1	100.000000
    # Vref                          double       	1	100.000000
    # WakeNumIter                   integer      	1	5
    # WingID                        string       	1
    # Xcg                           double       	1	0.000000
    # Ycg                           double       	1	0.000000
    # Zcg                           double       	1	0.000000
    # bref                          double       	1	1.000000
    # cref                          double       	1	1.000000
    # #############################################################

    # 设置参考翼面
    # 设置参考模型
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

    # analysis_method = vsp.GetIntAnalysisInput(analysis_name, "AnalysisMethod")  # 返回分析方法的索引
    # analysis_method = list(analysis_method)  # 类型转化一下
    # analysis_method[0] = vsp.VORTEX_LATTICE  # 这里的涡格法直接就在vsp下
    # vsp.SetIntAnalysisInput(analysis_name, "AnalysisMethod", analysis_method)
    # # 本来最后还有一个index（analysis_method是一个方法数组）但是现在缺省默认为0
    #
    # # 获取当前设置的分析方法
    # analysis_method = vsp.GetIntAnalysisInput(analysis_name, "AnalysisMethod")
    #
    # # 检查分析方法是否为涡格法
    # if analysis_method[0] == vsp.VORTEX_LATTICE:
    #     print("分析方法已正确设置为涡格法")
    # else:
    #     print("分析方法未正确设置为涡格法")
    #
    # # 列出输入里类型和当前的值
    # print("\n分析中输入类型\n")
    # vsp.PrintAnalysisInputs(analysis_name)

    # # ####################################################
    # 进行中间的保存尝试 观察vsp3是否正常
    file_name = "D:\\aircraft design competition\\24solar\\design_model\\whole_wing_optimization\\vsp\\test2.vsp3"
    vsp.WriteVSPFile(file_name, vsp.SET_ALL)
    # # ###############################################
    #
    # #############################################################
    # 开始计算
    print("\tExecution...")
    test = vsp.ExecAnalysis(analysis_name)
    print("COMPLETE")
    # 返回一个列表里面是所有可用结果的名称
    results = vsp.GetAllResultsNames()
    # 查找point的ID并赋值给point_id
    point_id = vsp.FindResultsID("point")
    # 一样的也是一个查找
    polar_id = vsp.FindResultsID("VSPAERO_Polar")

    # 用来了解可用的结果和数据的名称（全部打印出来）
    for result in results:
        data_names = vsp.GetAllDataNames(vsp.FindResultsID(result))
        for data_name in data_names:
            print(f"{result} > {data_name}")
    # 用来统计CL的数据量并打印第一个
    print("num of data in 'VSPAERO_Polar > CL': ", vsp.GetNumData(polar_id, "CL"))
    cl = vsp.GetDoubleResults(polar_id, "CL", 0)
    print(f"CL = {cl}\n")

    print("num of data in 'VSPAERO_Polar > L_D': ", vsp.GetNumData(polar_id, "L_D"))
    l_d = vsp.GetDoubleResults(polar_id, "CL", 0)
    print(f"L/D = {l_d}\n")
    print("num of data in 'area': ", vsp.GetNumData(point_id, "area"))
    area = vsp.GetDoubleResults(point_id, "area", 0)
    print(f"area = {area}\n")
    vsp.WriteResultsCSVFile(test,
                            "D:\\aircraft design competition\\24solar\\design_model\\whole_wing_optimization\\vsp\\result.csv")

    # # 开始进行单步骤的计算
    #
    # # 参考截面设置
    # vsp.SetIntAnalysisInput(analysis_name, "GeomSet", [], 0)
    #
    # # 获取当前所选几何集合的范围(检查部分)
    # geom_set = vsp.GetIntAnalysisInput(analysis_name, "GeomSet")
    # # 检查几何集合是否为空列表 特别注意 我们需要所有的几何体参与运算 将GeomSet设置为空列表就是All
    # if len(geom_set) == 0:
    #     print("几何集合已正确设置为使用所有几何集合")
    # else:
    #     print("几何集合未正确设置为使用所有几何集合")
    #
    # # 设置参考面积 :( 这个API文档谁写的也太偷懒了，坑死人了
    #
    # # vsp.SetDoubleAnalysisInput(analysis_name, "Sref_", [0.0], 0)
    # # vsp.SetDoubleAnalysisInput(analysis_name, "Bref_", [0.0], 0)
    # # vsp.SetDoubleAnalysisInput(analysis_name, "Cref_", [0.0], 0)
    # # wid = vsp.FindGeomsWithName("WingGeom")
    # # vsp.SetIntAnalysisInput(analysis_name, "WingID", wid, 0)

    # vsp.Update()
    #
    # # 打印输入类型和值
    # vsp.PrintAnalysisInputs(analysis_name)
    # print("")

    # # 继续处理
    # print("\tExecution...")
    # rid = vsp.ExecAnalysis(analysis_name)
    # print("COMPLETE")
    #
    # vsp.PrintResults(rid)

    # history_res = vsp.FindLatestResultsID("VSPAERO_History")
    # load_res = vsp.FindLatestResultsID("VSPAERO_Load")
    # CL = vsp.GetDoubleResults(history_res, "CL", 0)
    # cl = vsp.GetDoubleResults(load_res, "cl", 0)
    #
    # print("CL at 0 angle of attack:")
    # for i in range(len(CL)):
    #     print(CL[i])
    # print("")
    #
    # print("cl at 0 angle of attack: ")
    # for i in range(len(cl)):
    #     print(cl[i])
    # print("")
    # print("")


create_2()
vsp_aero()
