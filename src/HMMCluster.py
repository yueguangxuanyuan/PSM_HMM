import numpy as np
from HMM import *
from config import *
from dao import getHMMData
import copy
from contraint import compute_PMI_of_HMM_Clusters
from Common import make_dic_of_cluster_result
from HiddenStatesDetector import detect_best_hidden_state_num

def do_HMM_Cluster_On_data(observe_sequence_list,num_of_cluster ,num_of_hidden,num_of_observe,MAX_ITERATION = 100,show_progress = False):
    """
    假设所有聚类的HMM都同质，在已知隐藏状态个数已知的情况下训练HMM聚类
    :param observe_sequence_list:
    :param num_of_cluster:
    :param num_of_hidden:
    :param num_of_observe:
    :param MAX_ITERATION:
    :param show_progress:
    :return:
    """
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


    #判断是否收敛
    misConvergence =True;

    iterationCount = 0;
    while misConvergence:
        if iterationCount >= MAX_ITERATION :
            if show_progress :
                print("reach max_iteration");
            break;

        iterationCount +=1;
        if show_progress and iterationCount % 10 == 0 :
            print("enter iteration : ",iterationCount);

        #重新训练num_of_cluster 个聚类
        for index in range(num_of_cluster):
            _sub_dataset = [];
            for _data_item in data_of_clusters[index].values():
                _sub_dataset.append(_data_item);
            a_matrix_list[index],b_matrix_list[index],pi_list[index]\
                = baum_welch_multipleObservation(a_matrix_list[index],b_matrix_list[index],pi_list[index],_sub_dataset,showProgress=show_progress);

        #获取新的划分结果
        new_cluster_result = cluster_with_N_HMM(a_matrix_list,b_matrix_list,pi_list,observe_sequence_list);

        #判断是否收敛，这里用结果是否产生不变用的划分表示
        misConvergence = judgePatitionIsSame(data_of_clusters,new_cluster_result,K);
        if(show_progress):
            print("is misconvergence : ",misConvergence);

        data_of_clusters = copy.deepcopy(new_cluster_result);
    if(show_progress):
        print("end cluster");

    return a_matrix_list,b_matrix_list,pi_list,data_of_clusters;


def do_HMM_Cluster_On_data_with_dN(observe_sequence_list,num_of_cluster,num_of_observe,MAX_ITERATION = 100,show_progress = False):
    #随机生成n个HMM
    a_matrix_list = []
    b_matrix_list = [];
    pi_list = [];

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

    #判断是否收敛
    misConvergence =True;

    iterationCount = 0;
    while misConvergence:
        if iterationCount >= MAX_ITERATION :
            if show_progress :
                print("HMMC reach max_iteration");
            break;

        iterationCount +=1;
        if show_progress and iterationCount % 10 == 0 :
            print("HMMC enter iteration : ",iterationCount);

        #重新训练num_of_cluster 个聚类
        a_matrix_list.clear();
        b_matrix_list.clear();
        pi_list.clear();
        for index in range(num_of_cluster):
            _sub_dataset = [];
            for _data_item in data_of_clusters[index].values():
                _sub_dataset.append(_data_item);

            #在每个数据集上找到最好的隐变量个数
            best_N = detect_best_hidden_state_num(_sub_dataset,num_of_observe);

            a_matrix = init_A(best_N);
            b_matrix = init_B(best_N,num_of_observe);
            pi = init_PI(best_N);

            a_matrix, b_matrix, pi = baum_welch_multipleObservation(a_matrix, b_matrix, pi,_sub_dataset,showProgress=False);

            a_matrix_list.append(a_matrix);
            b_matrix_list.append(b_matrix);
            pi_list.append(pi);

        #获取新的划分结果
        new_cluster_result = cluster_with_N_HMM(a_matrix_list,b_matrix_list,pi_list,observe_sequence_list);

        #判断是否收敛，这里用结果是否产生不变用的划分表示
        misConvergence = judgePatitionIsSame(data_of_clusters,new_cluster_result,K);
        if(show_progress):
            print("HMMC is misconvergence : ",misConvergence);

        if misConvergence :
            new_cluster_result = recheck_cluster_result(new_cluster_result,a_matrix_list,b_matrix_list,pi_list);

        data_of_clusters = copy.deepcopy(new_cluster_result);

    if(show_progress):
        print("HMMC end cluster");

    return a_matrix_list,b_matrix_list,pi_list,data_of_clusters;


def recheck_cluster_result(cluster_result , a_matrix_list,b_matrix_list,pi_list):
    #处理空簇，保证簇的数量不下降
    empty_cluster_index_array = [];
    for cluster_index in range(cluster_result.__len__()):
        if cluster_result[cluster_index].__len__() == 0:
            empty_cluster_index_array.append(cluster_index);

    while empty_cluster_index_array.__len__() > 0:
        max_cluster_count = 0;
        max_cluster_index = -1;
        #找到最大的簇
        for cluster_index in range(cluster_result.__len__()):
            if cluster_result[cluster_index].__len__() > max_cluster_count:
                max_cluster_count = cluster_result[cluster_index].__len__();
                max_cluster_index = cluster_index;

        #找到距离最远的元素
        min_p = 1;
        min_p_index = -1;
        for item_index in cluster_result[max_cluster_index]:
            alphaMatrix,pforward = forword(a_matrix_list[max_cluster_index],b_matrix_list[max_cluster_index],pi_list[max_cluster_index], \
                                           cluster_result[max_cluster_index][item_index]);
            if pforward < min_p :
                min_p = pforward;
                min_p_index = item_index;

        #重新分配元素
        target_index = empty_cluster_index_array.pop()
        cluster_result[target_index][min_p_index] = cluster_result[max_cluster_index][min_p_index];
        cluster_result[max_cluster_index].pop(min_p_index);

    return cluster_result;

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

    a_matrix_list, b_matrix_list, pi_list, data_of_clusters = do_HMM_Cluster_On_data_with_dN(o_sequence_List,4,M,show_progress=True)
    #a_matrix_list, b_matrix_list, pi_list, data_of_clusters = do_HMM_Cluster_On_data(o_sequence_List,3,4,M,show_progress=True)

    PMI = compute_PMI_of_HMM_Clusters(a_matrix_list,b_matrix_list,pi_list,data_of_clusters,o_sequence_List);
    print(PMI);