from HMM import *
from config import *
from dao import getHMMData
from Util import normalize_matrix_line
from Util import shelterSmallValue
from contraint import compute_bic_of_HMM

def try_train_HMM_on_Data():
    N = 8;
    M = OBSERVE_STATE_LIST.__len__();

    a_matrix = init_A(N);
    b_matrix = init_B(N,M);
    pi = init_PI(N);

    # print(a_matrix);
    # print(b_matrix);
    # print(pi);

    _data = getHMMData();

    o_sequence_List = [];

    for index in range(_data.__len__()):
        o_sequence_List.append(_data[index][1]);


    a_matrix,b_matrix,pi = baum_welch_multipleObservation(a_matrix,b_matrix,pi,o_sequence_List,None,10);

    # print(normalize_matrix_line(shelterSmallValue(a_matrix,1e-5,0)));
    # print(normalize_matrix_line(shelterSmallValue(b_matrix,1e-5,0)));
    # print(normalize_matrix_line(shelterSmallValue(np.array([pi]),1e-5,0))[0]);

    print(compute_bic_of_HMM(a_matrix,b_matrix,pi,o_sequence_List))


if __name__ == "__main__":
    try_train_HMM_on_Data();