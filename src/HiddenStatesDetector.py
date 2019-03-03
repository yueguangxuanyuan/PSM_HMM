from config import *
from HMM import *
from contraint import compute_bic_of_HMM
from dao import getHMMData
import matplotlib.pyplot as plt
import time

def try_differen_hidden_state_nums(observe_sequence_list,M):
    # 实验配置值
    repeat_times = 10;
    N_start = 1;
    N_end = 7;

    N_list = [];
    average_bic_list = [];
    for N in range(N_start, N_end + 1):
        #print("HSD :: now try N = ", N);
        N_list.append(N);
        bic_list = [];
        for index in range(repeat_times):
            a_matrix = init_A(N);
            b_matrix = init_B(N, M);
            pi = init_PI(N);
            a_matrix, b_matrix, pi = baum_welch_multipleObservation(a_matrix, b_matrix, pi, observe_sequence_list, None, 1);

            bic = compute_bic_of_HMM(a_matrix, b_matrix, pi, observe_sequence_list)
            bic_list.append(bic);

        sum_of_bic = 0;
        for bic in bic_list:
            sum_of_bic += bic;
        average_bic_list.append(float(sum_of_bic) / repeat_times);

    return N_list,average_bic_list;

def detect_best_hidden_state_num(observe_sequence_list,M):
    N_list, average_bic_list = try_differen_hidden_state_nums(observe_sequence_list, M);

    max_bic_index = -1;
    max_bic = - float('inf');

    for index,bic in enumerate(average_bic_list):
        if bic > max_bic :
            max_bic = bic;
            max_bic_index = index;

    #draw_bic_with_different_hidden_state_num(None,0 , N_list,average_bic_list);

    return N_list[max_bic_index];

def draw_bic_with_different_hidden_state_num(observe_sequence_list,M ,N_list = None, average_bic_list=None):
    if N_list == None:
        N_list, average_bic_list = try_differen_hidden_state_nums (observe_sequence_list,M);

    pic_path = OUT_ROOT_PATH + "HSD-"+time.strftime('%Y-%m-%d_%H-%M-%S',time.localtime(time.time()));

    plt.figure();
    plt.plot(N_list,average_bic_list);
    plt.xlabel("N");
    plt.ylabel("BIC");
    plt.title("average BIC of N");
    plt.savefig(pic_path);
    plt.show();


if __name__ == "__main__":
    # 数据准备
    M = OBSERVE_STATE_LIST.__len__();
    _data = getHMMData();
    o_sequence_List = [];
    for index in range(_data.__len__()):
        o_sequence_List.append(_data[index][1]);
    print(detect_best_hidden_state_num(o_sequence_List,M));
    #print(OUT_ROOT_PATH + time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())));