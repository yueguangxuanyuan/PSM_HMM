import os

PROJECT_ROOT = os.path.dirname(os.path.realpath(__file__))  # 获取项目根目录

DATA_ROOT_PATH = os.path.join(PROJECT_ROOT, "data/")  # 文件路径

PIC_ROOT_PATH = os.path.join(PROJECT_ROOT, "pic/")  # 图片路径

OUT_ROOT_PATH = os.path.join(PROJECT_ROOT, "out/")  # 默认输出路径

VALIDATE_OUT_ROOT_PATH = os.path.join(PROJECT_ROOT, "validate_out/")  # 默认验证输出路径

FEATURE_DELIMITER = ',';

STATE_DELIMITER = '-';

NULL_OCCUPY = 'UNKNOWN';

DATA_FILE_NAME='psmSequence-e1-t1';#数据文件名称

OBSERVE_STATE_LIST = ["UU","UN","UY","NU","NN","NY","YU","YN","YY"]; #隐式状态定义