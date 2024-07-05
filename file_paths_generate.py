import os
import paths


# 这是一个路径检查和生成的代码
# 如果对应路径没有则创建
def create_directories_if_not_exist(path):
    directory = os.path.dirname(path)
    if not os.path.exists(directory):
        os.makedirs(directory)


def check_and_generate_path():
    create_directories_if_not_exist(paths.fig_path)
    create_directories_if_not_exist(paths.population_file)
    create_directories_if_not_exist(paths.supervise_file)
    create_directories_if_not_exist(paths.root_file)
    create_directories_if_not_exist(paths.mid_file)
    create_directories_if_not_exist(paths.tip_file)
    create_directories_if_not_exist(paths.vsp_file)
    create_directories_if_not_exist(paths.dictionary_file)
    create_directories_if_not_exist(paths.input_weight_file)
    create_directories_if_not_exist(paths.output_weight_file)
    create_directories_if_not_exist(paths.random_file)
    create_directories_if_not_exist(paths.cmy_NN)
    create_directories_if_not_exist(paths.cl_NN)
    create_directories_if_not_exist(paths.cdi_NN)
    print("Paths Checked")
    return 0
