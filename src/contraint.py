import numpy as np;
from HMM import forword;
from HMM import backword;
import math


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

    BIC -= (N*N + N*M + N)*math.log(K,math.e);

    return BIC;


def compute_PMI_of_HMM_Clusters(a_matrix_list,b_matrix_list,pi_list,observe_sequence_list):
    pass;