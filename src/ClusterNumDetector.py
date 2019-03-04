from config import *;
from HMMCluster import *;
import time;
import matplotlib.pyplot as plt;
from contraint import compute_PMI_of_HMM_Clusters

def try_different_hmm_cluster_nums(o_sequence_List,M):
    # 实验配置值
    N_start = 2;
    N_end = 7;

    N_list = [];
    pmi_list = [];
    for N in range(N_start, N_end + 1):
        print("CND now try N = ", N);
        N_list.append(N);

        PMI = 0 ;#尝试规避掉PMI值为0的影响
        iteration = 0;
        while PMI == 0 and iteration < 2:
            iteration += 1;
            a_matrix_list, b_matrix_list, pi_list, data_of_clusters = do_HMM_Cluster_On_data_with_dN(o_sequence_List, N, M,show_progress=False);
            PMI = compute_PMI_of_HMM_Clusters(a_matrix_list, b_matrix_list, pi_list, data_of_clusters, o_sequence_List);

        pmi_list.append(PMI);

    return N_list,pmi_list;

def detect_best_hmm_cluster_num(o_sequence_List,M):
    N_list, pmi_list = try_different_hmm_cluster_nums(o_sequence_List,M);
    pass

def draw_pmi_with_different_cluster_num(o_sequence_List,M):
    N_list, pmi_list = try_different_hmm_cluster_nums(o_sequence_List,M);
    pic_path = OUT_ROOT_PATH + "CND-" + time.strftime('%Y-%m-%d_%H-%M-%S',time.localtime(time.time()));
    plt.figure();
    plt.plot(N_list,pmi_list);

    for a,b in zip(N_list,pmi_list):
        plt.text(a,b,b,ha="center",va="bottom");

    plt.xlabel("N");
    plt.ylabel("PMI");
    plt.title("PMI of N");
    plt.savefig(pic_path);
    plt.show();


if __name__ == "__main__":
    # 数据准备
    M = OBSERVE_STATE_LIST.__len__();
    _data = getHMMData();
    o_sequence_List = [];
    for index in range(_data.__len__()):
        o_sequence_List.append(_data[index][1]);
    draw_pmi_with_different_cluster_num(o_sequence_List,M);