import copy
import numpy as np

def normalize_matrix_line(matrixM):
    mShape = matrixM.shape;
    lineSum = [];
    resultMatrix = copy.deepcopy(matrixM);

    for i in range(mShape[0]):
        sum = 0;
        for j in range(mShape[1]):
            sum += matrixM[i][j];
        lineSum.append(sum);

    for i in range(mShape[0]):
        for j in range(mShape[1]):
            resultMatrix[i][j] = float(matrixM[i][j])/lineSum[i];

    return resultMatrix;

def shelterSmallValue(matrixM,threshold  , replace = 0):
    resultMatrix = copy.deepcopy(matrixM);

    location = np.where(matrixM < threshold);
    count = location[0].__len__();

    for index in range(count):
        resultMatrix[location[0][index]][location[1][index]] = replace;

    return resultMatrix;


if __name__ == "__main__":
    matrixM = np.random.randn(3,4);

    print(matrixM);

    print(shelterSmallValue(matrixM,1e-10,0));