import os

def get_project_root() -> str:
    """
    获取工程根目录（无论脚本在哪个目录运行，都能返回正确的根目录）
    原理：基于当前文件的绝对路径，向上推导到工程根目录
    """
    # 当前文件（path_utils.py）的绝对路径
    current_file = os.path.abspath(__file__)
    # __file__ = 当前这个.py文件自己的完整路径（在哪里）它是Python内置的特殊变量，不用定义，直接用。
    # 当前文件所在目录（utils/）
    current_dir = os.path.dirname(current_file)
    # 工程根目录（utils/ 的上一级）
    project_root = os.path.dirname(current_dir)
    return project_root

def get_abs_path(relative_path: str) -> str:
    """
    将工程内的相对路径转为绝对路径（统一路径基准）
    :param relative_path: 相对于工程根目录的路径，如 "config/rag.yml"
    yml是一种专门用来写配置文件的格式，在你的 RAG 项目里，它就是用来存各种参数的 “配置表”。
    :return: 绝对路径
    """
    project_root = get_project_root()
    return os.path.join(project_root, relative_path)
    # os.path.join 是 Python 里专门用来拼接文件路径的函数。
