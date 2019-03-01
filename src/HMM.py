import numpy as np
from Util import normalize_matrix_line

def init_A(N):
    """
    init  matrix of transition probability
    :return:
    """
    A = np.random.rand(N,N);
    A = normalize_matrix_line(A);
    return A;


def init_B(N,M):
    """
    init matrix of emission probability
    :return:
    """
    B = np.random.rand(N,M);
    B = normalize_matrix_line(B);
    return B;

def init_PI(N):
    """
    init matrix of original state probability
    :return:
    """
    PI = np.random.rand(1,N);
    PI = normalize_matrix_line(PI);
    return PI[0];

def forword(a_matrix,b_matrix,pi, observe_seq):
    """
    前向传播
    :return:
    """
    N = a_matrix.shape[0]; #隐式状态的数量
    T = observe_seq.__len__(); #观察序列的长度

    alphaMatrix = np.zeros((T,N));

    #初始化
    for i in range(N):
        alphaMatrix[0][i] = pi[i]*b_matrix[i][observe_seq[0]];

    #递推
    for t in range(1,T):
        for current_state in range(N):
            alphaMatrix[t][current_state] = 0;
            for pre_state in range(N):
                alphaMatrix[t][current_state] += alphaMatrix[t-1][pre_state]*a_matrix[pre_state][current_state];
            alphaMatrix[t][current_state] *= b_matrix[current_state][observe_seq[t]];

    #计算该模型下获得该观测序列的概率
    P_O_of_lamda = 0;
    for state in range(N):
        P_O_of_lamda += alphaMatrix[T-1][state];

    return alphaMatrix,P_O_of_lamda;


def backword(a_matrix,b_matrix,pi, observe_seq):
    """
    后向传播
    :return:
    """
    N = a_matrix.shape[0];  # 隐式状态的数量
    T = observe_seq.__len__();  # 观察序列的长度

    betaMatrix = np.zeros((T,N));

    #初始化
    for state in range(N):
        betaMatrix[T-1][state] = 1;

    #递推
    for t in range(T-2,-1,-1):
        for state in range(N):
            betaMatrix[t][state] = 0 ;
            for next_state in range(N):
                betaMatrix[t][state] += a_matrix[state][next_state]*b_matrix[next_state][observe_seq[t+1]]*betaMatrix[t+1][next_state];

    #计算模型下观察序列的概率
    P_O_of_lamda = 0;
    for state in range(N):
        P_O_of_lamda += pi[state]*b_matrix[state][observe_seq[0]]*betaMatrix[0][state];

    return betaMatrix, P_O_of_lamda;

def compute_gama(alphaMatrix,betaMatrix):
    """
    计算 时间为t时 处于 对象处于状态i的概率
    :return:
    """
    N = alphaMatrix.shape[1];  # 隐式状态的数量
    T = alphaMatrix.shape[0];  # 观察序列的长度

    P_I_O_of_lamda = np.zeros((T,N))#在t时刻下，在模型基础上获得如此观察序列，并且当前状态为i的概率

    for t in range(T):
        for state in range(N):
            P_I_O_of_lamda[t][state] = alphaMatrix[t][state]*betaMatrix[t][state];

    P_I_O_of_lamda_sum_by_t = P_I_O_of_lamda.sum(axis=1);

    gama = np.zeros((T,N));

    for t in range(T):
        for state in range(N):
            if(P_I_O_of_lamda_sum_by_t[t] == 0) :
                gama[t][state] = P_I_O_of_lamda[t][state];
            else:
                gama[t][state] = P_I_O_of_lamda[t][state]/P_I_O_of_lamda_sum_by_t[t];

    return gama;

def compute_xi(a_matrix,b_matrix,alphaMatrix,betaMatrix,observe_seq):
    """
    计算t时刻为状态i，t+1时刻为j的概率
    :return:
    """
    N = alphaMatrix.shape[1];  # 隐式状态的数量
    T = alphaMatrix.shape[0];  # 观察序列的长度
    P_I_O_of_lamda = np.zeros((T-1,N, N))  # 在t时刻下，在模型基础上获得如此观察序列，并且当前状态为i,下一个状态为j的概率

    for t in range(T-1):
        for state in range(N):
            for next_state in range(N):
                P_I_O_of_lamda[t][state][next_state] = alphaMatrix[t][state]*a_matrix[state][next_state]*b_matrix[next_state][observe_seq[t+1]]*betaMatrix[t+1][next_state];

    P_I_O_of_lamda_sum_by_t = P_I_O_of_lamda.sum(axis=2).sum(axis=1);

    xi = np.zeros((T - 1, N, N))
    for t in range(T-1):
        for state in range(N):
            for next_state in range(N):
                if( P_I_O_of_lamda_sum_by_t[t] == 0) :
                    xi[t][state][next_state] = P_I_O_of_lamda[t][state][next_state]
                else :
                    xi[t][state][next_state] = P_I_O_of_lamda[t][state][next_state]/P_I_O_of_lamda_sum_by_t[t];

    return xi;

def baum_welch(a_matrix, b_matrix, pi, sequence , iteration = 20):
    """
    采用
    baum_welch算法训练数据
    这里只能针对单条数据进行训练
    :param a_matrix:
    :param b_matrix:
    :param pi:
    :return:
    """
    N = a_matrix.shape[0];  # 隐式状态的数量
    M = b_matrix.shape[1];  # 显示状态的数量

    T = sequence.__len__();  # 观察序列的长度

    iter = 0 ;
    while iter < iteration :
        iter += 1;
        alphaMatrix,p = forword(a_matrix,b_matrix,pi,sequence);
        if p == 0 :
            #print("fronterror data ", k, " iter ", iter)
            continue;
        betaMatrix,p1 = backword(a_matrix,b_matrix,pi,sequence);
        if p1 == 0:
            #print("backerror data " , k , " iter ",iter )
            continue;

        gama = compute_gama(alphaMatrix,betaMatrix);
        xi = compute_xi(a_matrix,b_matrix,alphaMatrix,betaMatrix,sequence);

        #更新A
        for state in range(N):
            for next_state in range(N):
                numerator = 0;
                for t in range(T-1):
                    numerator += xi[t][state][next_state];
                denominator = 0;
                for t in range(T-1):
                    denominator += gama[t][state];

                if np.isnan(numerator) or np.isnan(denominator):
                    a_matrix[state][next_state] = 0;
                elif denominator == 0 :
                    a_matrix[state][next_state] = numerator
                else:
                    a_matrix[state][next_state] = numerator/denominator;

        #更新B
        for state in range(N):
            for o_state in range(M):
                numerator = 0;
                for t in range(T):
                    if sequence[t] == o_state :
                        numerator += gama[t][state];
                denominator = 0;
                for t in range(T):
                    denominator += gama[t][state];

                if np.isnan(numerator) or np.isnan(denominator):
                    b_matrix[state][o_state] = 0;
                elif denominator == 0 :
                    b_matrix[state][o_state] = numerator;
                else:
                    b_matrix[state][o_state] = numerator/denominator;

        #更新PI
        for state in range(N):
            pi[state] = gama[0][state];

    return a_matrix,b_matrix,pi;

def baum_welch_multipleObservation(a_matrix, b_matrix, pi, sequenceList ,W_k=None, iteration = 1 , showProgress = False):
    """
    采用
    baum_welch算法训练数据
    :param a_matrix:
    :param b_matrix:
    :param pi:
    :return:
    """
    N = a_matrix.shape[0];  # 隐式状态的数量
    M = b_matrix.shape[1];  # 显示状态的数量

    K = sequenceList.__len__(); # 观察序列的数量

    if W_k is None :
        W_k = np.ones(K)/K;

    iter = 0;
    while iter < iteration:
        iter += 1;

        if showProgress and iter%10 == 0 :
            print("enter iter ",iter);

        a_numerator = np.zeros(a_matrix.shape) ;
        b_numerator = np.zeros(b_matrix.shape) ;
        pi_numerator = np.zeros(pi.__len__()) ;

        a_denominator = np.zeros(a_matrix.shape);
        b_denominator = np.zeros(b_matrix.shape);
        pi_denominator = np.zeros(pi.__len__());

        for k, sequence in enumerate(sequenceList):
            T = sequence.__len__();  # 观察序列的长度
            alphaMatrix,p_forward = forword(a_matrix,b_matrix,pi,sequence);

            betaMatrix,p_backward = backword(a_matrix,b_matrix,pi,sequence);

            gama = compute_gama(alphaMatrix,betaMatrix);
            xi = compute_xi(a_matrix,b_matrix,alphaMatrix,betaMatrix,sequence);

            #计算A的影响变量
            for state in range(N):
                for next_state in range(N):
                    numerator = 0;
                    for t in range(T-1):
                        numerator += xi[t][state][next_state];
                    denominator = 0;
                    for t in range(T-1):
                        denominator += gama[t][state];

                    a_numerator[state][next_state] += W_k[k]*p_forward*numerator;
                    a_denominator[state][next_state] += W_k[k]*p_forward*denominator;

            #计算B的影响变量
            for state in range(N):
                for o_state in range(M):
                    numerator = 0;
                    for t in range(T):
                        if sequence[t] == o_state :
                            numerator += gama[t][state];
                    denominator = 0;
                    for t in range(T):
                        denominator += gama[t][state];

                    b_numerator[state][o_state] += W_k[k] * p_forward*numerator;
                    b_denominator[state][o_state] += W_k[k]*p_forward*denominator;

            #计算PI的影响变量
            for state in range(N):
                pi_numerator[state] = W_k[k]*p_forward*gama[0][state];
                pi_denominator[state] = W_k[k]*p_forward;

        #更新A
        for state in range(N):
            for next_state in range(N):
                a_matrix[state][next_state] = a_numerator[state][next_state]/a_denominator[state][next_state];

        #更新B
        for state in range(N):
            for o_state in range(M):
                b_matrix[state][o_state] = b_numerator[state][o_state] / b_denominator[state][o_state];

        #更新PI
        for state in range(N):
            pi[state] = pi_numerator[state]/pi_denominator[state];

    return a_matrix,b_matrix,pi;


if __name__ == "__main__":
    N = 3;
    M = 2;
    # a_matrix = init_A(N);
    # b_matrix = init_B(N,M);
    # pi = init_PI(N);
    a_matrix = np.array([[0.5,0.2,0.3],[0.3,0.5,0.2],[0.2,0.3,0.5]]);
    b_matrix = np.array([[0.5,0.5],[0.4,0.6],[0.7,0.3]]);
    pi = [0.2,0.4,0.4];

    o_sequence = [0,1,0];

    alphaMatrix,p = forword(a_matrix,b_matrix,pi,o_sequence);
    betaMatrix,p1 = backword(a_matrix,b_matrix,pi,o_sequence);

    gama = compute_gama(alphaMatrix,betaMatrix)
    xi = compute_xi(a_matrix,b_matrix,alphaMatrix,betaMatrix,o_sequence)

    print("a_matrix")
    print(a_matrix);
    print("b_matrix")
    print(b_matrix);
    print("pi")
    print(pi);
    print("o_sequence")
    print(o_sequence)
    print(alphaMatrix)
    print(p)
    print(betaMatrix);
    print(p1)
    print(gama)
    print(xi);

    a_matrix, b_matrix,pi= baum_welch(a_matrix, b_matrix, pi, [o_sequence], 20)

    print("a_matrix")
    print(a_matrix);
    print("b_matrix")
    print(b_matrix);
    print("pi")
    print(pi);