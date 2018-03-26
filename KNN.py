import numpy as np
import pandas as pd
import operator

def readData(file_route):
    data = pd.read_table(file_route, header=None, delim_whitespace=True)
    labels = data.loc[:,data.shape[1]-1]
    data = data.loc[:, data.shape[1]-2]
    return data, labels

def classify(inX, data, labels, k):
    # tile(): Construct an array by repeating A the number of times given by reps.
    # >> > a = np.array([0, 1, 2])
    # >> > np.tile(a, 2)
    # array([0, 1, 2, 0, 1, 2])
    # >> > np.tile(a, (2, 2))
    # array([[0, 1, 2, 0, 1, 2],
    #        [0, 1, 2, 0, 1, 2]])
    # >> > np.tile(a, (2, 1, 2))
    # array([[[0, 1, 2, 0, 1, 2]],
    #        [[0, 1, 2, 0, 1, 2]]])
    # argsort(): 将array排序 ，返回排序后的index
    # >>> x = np.array([3, 1, 2])
    # >>> np.argsort(x)
    # array([1, 2, 0])
    # >> > itemgetter(1)('ABCDEFG')
    # 'B'
    # >> > itemgetter(1, 3, 5)('ABCDEFG')
    # ('B', 'D', 'F')
    # >> > itemgetter(slice(2, None))('ABCDEFG')
    # 'CDEFG'
    dataSize = data.shape[0]
    diffMat = np.tile(inX, (dataSize, 1)) - data
    sqDiffMat = diffMat ** 2
    sqDistance = sqDiffMat.sum(axis=1)
    distance = sqDistance ** 0.5
    sortedDistIndices = distance.argsort()
    classCount = {}
    for i in range(k):
        voteLabel = labels[sortedDistIndices[i]]
        classCount[voteLabel] = classCount.get(voteLabel, 0) + 1
    sortedClassCount = sorted(classCount.iteritems(), #迭代器
                                 key=operator.itemgetter(1),
                                 reverse=True)
    return sortedClassCount[0][0]

