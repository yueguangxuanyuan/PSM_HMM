import numpy as np;
from HMM import forword;
from HMM import backword;
import math
from Common import make_dic_of_cluster_result


def compute_bic_of_HMM(a_matrix,b_matrix,pi,observeSequenceList,W_k=None):

    N = a_matrix.shape[0] ; #隐状态的数量
    M = b_matrix.shape[1] ; #显示状态的数量

    K = observeSequenceList.__len__();

    if W_k is None:
        W_k = np.ones(K)/K;#默认权重相同

    BIC = 0 ;

    for k,sequence in enumerate(observeSequenceList):
        alphaMatrix,p_forword = forword(a_matrix,b_matrix,pi,sequence);
        betaMatrix, p_backword = backword(a_matrix, b_matrix, pi, sequence);

        if(p_forword > p_backword):
            p_forword = p_backword;

        if p_forword == 0 :
            # print(k)
            pass
        else :
            BIC += math.log(p_forword,math.e);

    BIC *=2;

    if K > 0 :
        BIC -= (N*N + N*M + N)*math.log(K,math.e);

    return BIC;


def compute_PMI_of_HMM_Clusters(a_matrix_list,b_matrix_list,pi_list,cluster_result,sequence_list):
    num_of_clusters = a_matrix_list.__len__();#聚类数
    K = sequence_list.__len__();

    #计算lamda
    p_lamda_list = [];
    for cluster_index in range(num_of_clusters):
        p_lamda_list.append( float(len(cluster_result[cluster_index])) / K);

    #计算 计算元
    itemMatrix = np.zeros((K,num_of_clusters));
    for sequence_index,sequence in enumerate(sequence_list):
        for cluster_index in range(num_of_clusters) :
            a_matrix,p_forward = forword(a_matrix_list[cluster_index],b_matrix_list[cluster_index],pi_list[cluster_index],sequence);
            itemMatrix[sequence_index][cluster_index] = p_forward*p_lamda_list[cluster_index];
    itemMatrixLinSum = itemMatrix.sum(axis=1);

    #计算互信息
    cluster_result_dic = make_dic_of_cluster_result(cluster_result);
    MI_list = [];
    for sequence_index in range(K):
        MI_I = 0;
        if itemMatrix[sequence_index][cluster_result_dic[sequence_index]] !=  0 :
            MI_I =math.log(itemMatrix[sequence_index][cluster_result_dic[sequence_index]],math.e)\
                       - math.log(itemMatrixLinSum[sequence_index],math.e)

        MI_list.append(MI_I);

    #计算互信息
    PMI = 0;
    for value in MI_list :
        PMI += value;

    J = num_of_clusters;#万一出现空聚类的情况，确保PMI的正确计算
    for cluster_index in range(num_of_clusters):
        if cluster_result[cluster_index].__len__() == 0:
            J -= 1;

    PMI /= J;

    return PMI;