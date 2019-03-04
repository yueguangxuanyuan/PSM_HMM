from  singleTrail import try_train_HMM_on_Data
from HMMCluster import do_HMM_Cluster_On_data_with_dN;
from config import *
from dao import getHMMData
from contraint import compute_PMI_of_HMM_Clusters


if __name__ == "__main__":
    # 数据准备
    M = OBSERVE_STATE_LIST.__len__();
    _data = getHMMData();
    o_sequence_List = [];
    for index in range(_data.__len__()):
        o_sequence_List.append(_data[index][1]);

    best_N = 3;
    a_matrix_list, b_matrix_list, pi_list, data_of_clusters = do_HMM_Cluster_On_data_with_dN(o_sequence_List,best_N,M,show_progress=True);

    for cluster_index in range(best_N):
        print("==== cluster ",cluster_index , " ====");
        print(a_matrix_list[cluster_index])
        print(b_matrix_list[cluster_index])
        print(pi_list[cluster_index])

        item_index_array = []
        for item_index in data_of_clusters[cluster_index]:
            item_index_array.append(item_index);

        print(item_index_array)
        print("\n")

    print(compute_PMI_of_HMM_Clusters(a_matrix_list,b_matrix_list,pi_list,data_of_clusters,o_sequence_List))