from config import *
import copy

def loadData():
    _datafile = open(DATA_ROOT_PATH + DATA_FILE_NAME , 'r');

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
        record_array[1] = record_array[1].split(STATE_DELIMITER);
        _data.append(record_array);

    return _data;

def getHMMData():
    _origin_data = loadData();

    observeStateMap = {};
    for index,state in enumerate(OBSERVE_STATE_LIST):
        observeStateMap[state] = index;

    _data = copy.deepcopy(_origin_data);
    for index in range(_origin_data.__len__()):
        psmStateSequence = _origin_data[index][1];

        for t in range(psmStateSequence.__len__()):
            _data[index][1][t] = observeStateMap[psmStateSequence[t]];

    return _data;

if __name__ == "__main__":
    _data = loadData();
    print(_data)
    print(getHMMData())