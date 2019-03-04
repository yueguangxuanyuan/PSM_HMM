from dao import getHMMData
from Validate_Dao import getScoreData
from Validate_Dao import loadFeatureData
from  Cluster_Result import *;
import copy;
import matplotlib.pyplot as plt

def convert_index_to_Uid():
    _data = getHMMData();

    index_to_uid_dic = {};
    for index in range(_data.__len__()):
        index_to_uid_dic[index] = _data[index][0];

    _out_data = copy.deepcopy(cluster_result);
    for cluster_index in range(cluster_result.__len__()):
        for item_index,value in enumerate(cluster_result[cluster_index]):
            _out_data[cluster_index][item_index] = index_to_uid_dic[value];

    print(_out_data);

def get_score_data_map(align = 10):
    _score_data = getScoreData();
    _score_data_map = {};
    for index in range(_score_data.__len__()):
        _score_data_map[_score_data[index][0]] = float(_score_data[index][1]) // align;
    return _score_data_map;

def get_uid_to_score(align = 10):
    _score_data_map = get_score_data_map(align);
    _uid_to_score = [];
    for cluster_data in cluster_result_transferred:
        the_map = {};
        for uid in cluster_data:
            the_map[uid] = _score_data_map[uid];
        _uid_to_score.append(the_map);
    return _uid_to_score;

def check_score_distribution():
    _uid_to_score = get_uid_to_score();

    plt.figure();
    cluster_num = cluster_result_transferred.__len__();

    for cluster_index in range(cluster_num):
        x_list = []
        y_list = []
        for score in range(11):
            x_list.append(score);
            y_list.append(0);

        for uid in _uid_to_score[cluster_index]:
            y_list[int(_uid_to_score[cluster_index][uid])] += 1;

        plt.subplot(cluster_num,1,cluster_index+1);
        plt.bar(x_list,y_list);
        plt.xlabel("score");
        plt.ylabel("count");

    plt.show();

from scipy import stats

def check_score_with_anova():
    _uid_to_score = get_uid_to_score(align=10);
    _data = [];
    for cluster_data in _uid_to_score:
        _cluster_score = [];
        for uid in cluster_data :
            _cluster_score.append(cluster_data[uid]);
        if _cluster_score.__len__() > 10:
            _data.append(_cluster_score);

    w,p = stats.levene(*_data);
    print(w,p) #p 小于0.05表示方差不齐，不能继续进行anova检验
    f,p_f = stats.f_oneway(*_data)
    print(f,p_f)


def check_different_feature_with_anova():
    _student_data, headerArray = loadFeatureData();

    print( "feature,w,p,f,p_f" )
    for feature_index in range(1,headerArray.__len__()):
        if headerArray[feature_index] == "unknownCount":
            continue;
        _uid_to_feature_map = {};
        for item in _student_data :
            _uid_to_feature_map[item[0]] = item[feature_index];

        cluster_with_feature = [];
        for cluster_index in range(cluster_result_transferred.__len__()):
            feature_array = [];
            for uid in cluster_result_transferred[cluster_index]:
                if uid in _uid_to_feature_map:
                    feature_array.append(_uid_to_feature_map[uid]);
            cluster_with_feature.append(feature_array);

        w,p = stats.levene(*cluster_with_feature);
        f,p_f = stats.f_oneway(*cluster_with_feature);

        print(headerArray[feature_index],",",w,",",p,",",f,",",p_f);


def check_different_feature_with_anova_limit2():
    _student_data, headerArray = loadFeatureData();

    for i in range(cluster_result_transferred.__len__()-1):
        for j in range(i+1,cluster_result_transferred.__len__()):
            print(i,j);
            compare_array = [i,j];
            print( "feature,w,p,f,p_f" )
            for feature_index in range(1,headerArray.__len__()):
                if headerArray[feature_index] == "unknownCount":
                    continue;
                _uid_to_feature_map = {};
                for item in _student_data :
                    _uid_to_feature_map[item[0]] = item[feature_index];

                cluster_with_feature = [];
                for cluster_index in compare_array:
                    feature_array = [];
                    for uid in cluster_result_transferred[cluster_index]:
                        if uid in _uid_to_feature_map:
                            feature_array.append(_uid_to_feature_map[uid]);
                    cluster_with_feature.append(feature_array);

                w,p = stats.levene(*cluster_with_feature);
                f,p_f = stats.f_oneway(*cluster_with_feature);

                print(headerArray[feature_index],",",w,",",p,",",f,",",p_f);

if __name__ == "__main__":
    #convert_index_to_Uid();
    #check_score_distribution();
    # check_score_with_anova();
    check_different_feature_with_anova_limit2()