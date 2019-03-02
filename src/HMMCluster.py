import numpy as np
from HMM import *
from config import *
from dao import getHMMData
import copy

def do_HMM_Cluster_On_data(observe_sequence_list,num_of_cluster ,num_of_hidden,num_of_observe,MAX_ITERATION = 100):
    #随机生成n个HMM
    a_matrix_list = []
    for index in range(num_of_cluster):
        a_matrix_list.append(init_A(num_of_hidden));
    a_matrix_list = np.array(a_matrix_list);

    b_matrix_list = [];
    for index in range(num_of_cluster):
        b_matrix_list.append(init_B(num_of_hidden,num_of_observe));
    b_matrix_list = np.array(b_matrix_list);

    pi_list = [];
    for index in range(num_of_cluster):
        pi_list.append(init_PI(num_of_hidden));
    pi_list = np.array(pi_list)

    #将数据切分成n分
    K = observe_sequence_list.__len__();#总的数据数
    data_num_in_each_cluster = K//num_of_cluster #每个簇 初始的数量，如果不能整除，那么多的对象分到最后一个簇中

    data_of_clusters = [];
    for index in range(num_of_cluster):
        data_of_clusters.append({});

    for index,sequence in enumerate(observe_sequence_list):
        index_of_cluster = index//data_num_in_each_cluster;
        if index_of_cluster >= num_of_cluster :
            index_of_cluster = num_of_cluster -1;
        data_of_clusters[index_of_cluster][index] = sequence;


    #针对
    misConvergence =True;

    iterationCount = 0;
    while misConvergence:
        if iterationCount >= MAX_ITERATION :
            print("reach max_iteration");
            break;

        iterationCount +=1;
        if iterationCount % 10 == 0 :
            print("enter iteration : ",iterationCount);

        #重新训练num_of_cluster 个聚类
        for index in range(num_of_cluster):
            _sub_dataset = [];
            for _data_item in data_of_clusters[index].values():
                _sub_dataset.append(_data_item);
            a_matrix_list[index],b_matrix_list[index],pi_list[index]\
                = baum_welch_multipleObservation(a_matrix_list[index],b_matrix_list[index],pi_list[index],_sub_dataset,iteration=1);

        #获取新的划分结果
        new_cluster_result = cluster_with_N_HMM(a_matrix_list,b_matrix_list,pi_list,observe_sequence_list);

        #判断是否收敛，这里用结果是否产生不变用的划分表示
        misConvergence = judgePatitionIsSame(data_of_clusters,new_cluster_result,K);
        print("is misconvergence : ",misConvergence);

        data_of_clusters = copy.deepcopy(new_cluster_result);

    print("end cluster");

def cluster_with_N_HMM(a_matrix_list ,b_matrix_list ,pi_list , o_sequence_list):
    num_of_cluster = a_matrix_list.__len__();
    data_of_inner_clusters = [];
    for index in range(num_of_cluster):
        data_of_inner_clusters.append({});

    for index,o_sequence in enumerate(o_sequence_list):
        p_of_sequence = -1;
        cluster_index_of_sequence = -1;

        for cluster_index in range(num_of_cluster):
            a_matrix,p_forward = forword(a_matrix_list[cluster_index],b_matrix_list[cluster_index],pi_list[cluster_index],o_sequence);

            if p_forward > p_of_sequence :
                p_of_sequence = p_forward;
                cluster_index_of_sequence = cluster_index;

        data_of_inner_clusters[cluster_index_of_sequence][index] = o_sequence;

    return data_of_inner_clusters;

def make_dic_of_cluster_result(cluster_result):
    """
    根据分类结果，返回item_index到对应聚类的字典索引
    :param cluster_result:
    :return:
    """
    cluster_result_dic = {};

    for cluster_index in range(cluster_result.__len__()):
        for index in cluster_result[cluster_index] :
            cluster_result_dic[index] = cluster_index;

    return cluster_result_dic;

def judgePatitionIsSame(pre_cluster_result,new_cluster_result,K):
    pre_cluster_result_dic  = make_dic_of_cluster_result(pre_cluster_result);
    new_cluster_result_dic = make_dic_of_cluster_result(new_cluster_result);

    disMatchCount = 0;
    for i in range(K-1):
        for j in range(1,K):
            is_pre_same_cluster = pre_cluster_result_dic[i] == pre_cluster_result_dic[j];
            is_new_same_cluster = new_cluster_result_dic[i] == new_cluster_result_dic[j];

            disMatchCount += is_new_same_cluster^is_pre_same_cluster;

    return disMatchCount > 0 ;

if __name__ == "__main__":
    # 数据准备
    M = OBSERVE_STATE_LIST.__len__();
    _data = getHMMData();
    o_sequence_List = [];
    for index in range(_data.__len__()):
        o_sequence_List.append(_data[index][1]);

    do_HMM_Cluster_On_data(o_sequence_List,3,3,M)