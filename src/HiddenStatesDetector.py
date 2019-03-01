from config import *
from HMM import *
from contraint import compute_bic_of_HMM
from dao import getHMMData
import matplotlib.pyplot as plt
import time

def detect_best_hidden_state_num():
    #实验配置值
    repeat_times = 10;
    N_start = 1;
    N_end = 10;

    #数据准备
    M = OBSERVE_STATE_LIST.__len__();
    _data = getHMMData();
    o_sequence_List = [];
    for index in range(_data.__len__()):
        o_sequence_List.append(_data[index][1]);

    N_list = [];
    average_bic_list = [];
    for N in range(N_start,N_end+1):
        print("now try N = ",N);
        N_list.append(N);
        bic_list = [];
        for index in range(repeat_times):
            a_matrix = init_A(N);
            b_matrix = init_B(N,M);
            pi = init_PI(N);
            a_matrix,b_matrix,pi = baum_welch_multipleObservation(a_matrix,b_matrix,pi,o_sequence_List,None,10);

            bic = compute_bic_of_HMM(a_matrix,b_matrix,pi,o_sequence_List)
            bic_list.append(bic);

        sum_of_bic = 0;
        for bic in bic_list :
            sum_of_bic += bic;
        average_bic_list.append(float(sum_of_bic)/repeat_times);

    pic_path = OUT_ROOT_PATH + time.strftime('%Y-%m-%d_%H-%M-%S',time.localtime(time.time()));

    plt.figure();
    plt.plot(N_list,average_bic_list);
    plt.xlabel("N");
    plt.ylabel("BIC");
    plt.title("average BIC of N");
    plt.savefig(pic_path);
    plt.show();


if __name__ == "__main__":
    detect_best_hidden_state_num();
    #print(OUT_ROOT_PATH + time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())));