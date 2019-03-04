from config import *;
import copy
from common.Typehelper import strToType

def loadData():
    _datafile = open(DATA_ROOT_PATH+SCORE_FILE_NAME  , 'r');

    # 读取头部信息
    _headerLine = _datafile.readline().rstrip('\n');
    headerArray = _headerLine.split(FEATURE_DELIMITER);
    _headerTypeLine = _datafile.readline().rstrip('\n');
    headerTypeArray = _headerTypeLine.split(FEATURE_DELIMITER);

    # 读取数据
    _data = [];
    for _line_record in _datafile:
        _line_record = _line_record.rstrip('\n');
        record_array = _line_record.split(FEATURE_DELIMITER);
        _data.append(record_array);

    return _data;

def loadFeatureData():
    """
    加载学生特征数据
    :return:
    """
    _datafile = open(DATA_ROOT_PATH+FEATURE_DATA_FILE_NAME  , 'r');

    # 读取头部信息
    _headerLine = _datafile.readline().rstrip('\n');
    headerArray = _headerLine.split(FEATURE_DELIMITER);

    _headerTypeLine = _datafile.readline().rstrip('\n');
    headerTypeArray = _headerTypeLine.split(FEATURE_DELIMITER);

    # 读取数据
    _student_data = [];
    for _line_record in _datafile:
        _line_record = _line_record.rstrip('\n');
        record_array = _line_record.split(FEATURE_DELIMITER);
        for _index, _value in enumerate(record_array):
            if (_value != NULL_OCCUPY):
                try:
                    record_array[_index] = strToType(headerTypeArray[_index], record_array[_index]);
                except:
                    print(record_array[0]);
                    exit(-1);

        _student_data.append(record_array);

    # 移除空值行
    _index_of_line_contain_null = [];
    for _index, _line in enumerate(_student_data):
        for item in _line:
            if item == NULL_OCCUPY:
                _index_of_line_contain_null.append(_index);
                break;
    # print(index_of_line_contain_null);
    _index_of_line_contain_null.reverse();
    for _index in _index_of_line_contain_null:
        del _student_data[_index]

    return _student_data,headerArray;

def getScoreData():
    _origin_data = loadData();
    return _origin_data;