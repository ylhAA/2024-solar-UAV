import numpy as np

import Gemo_Generate
import vsp_4
import paths
import CST_Generate_1

arr = np.array([0.23491444598398273, 0.33358884593665983, 0.2410681245948411, 0.2, 0.4, -0.4, -0.4, 0.2,
                0.2626700635899731, 0.2, 0.2, -0.35398911012660284, 0.2302333423466283, -0.16407931006779278, 1.0])
Gemo_Generate.CST_airfoil_file_generate(arr, paths.tip_file,
                                        paths.mid_file, paths.root_file)
vsp_4.create_Geom_3(paths.tip_file, paths.mid_file, paths.root_file)
