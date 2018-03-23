import numpy as np
import pandas as pd

def read_txt(file_path):
    data = pd.read_table(file_path, header=None, delim_whitespace=True)
    return data

train = read_txt("C:/Song-Code/MLiAData/Ch05/HorseColicTraining.txt")
test = read_txt("C:/Song-Code/MLiAData/Ch05/HorseColicTest.txt")

# def colic_test():
#     train_data_set = open('D:\DataSet\HorseColicTraining.txt')
#     training_set = []
#     training_labels = []
#     # 按行读取文件，形成一个list，其中每个元素为文件中的一行
#     for line in train_data_set.readlines():
#         curr_line = line.strip().split('\n')
#         # 将文件每行的不同特征数据分开
#         for s in curr_line:
#             curr_line_processing = s.split()
#             curr_line_processing = curr_line_processing[:2] + curr_line_processing[3:24]
#             # 处理缺失数据,将?替换成0,马的年龄有1,2,9，不知9是否为缺失数据
#             for i in range(len(curr_line_processing)):
#                 if curr_line_processing[i] == '?':
#                     curr_line_processing[i] = 0.0
#                 curr_line_processing[i] = float(curr_line_processing[i])
#                 # 数据类别标签，书中将死亡和安乐死合并，从而使类别标签变为了两个0和1，
#                 # 这里也这样处理，以后可考虑用softmax模型
#             print(curr_line_processing[21])
#             if curr_line_processing[21] == 3.0000 or curr_line_processing[21] == 2.0000:
#                 curr_line_processing[21] = 1.0000
#             else:
#                 curr_line_processing[21] = 0.0000
#             training_set.append(curr_line_processing[:21])
#             training_labels.append(curr_line_processing[21])
#     train_paras = stochastic_gradient_descent(np.array(training_set), training_labels)
#     train_data_set.close()
#     # 读取测试数据
#     test_data_set = open('D:\DataSet\HorseColicTest.txt')
#     test_data = []
#     test_labels = []
#     num = 0
#     err_count = 0
#     for line in test_data_set.readlines():
#         num += 1
#         curr_line = line.strip().split('\n')
#         curr_line_processing = curr_line[0].split()
#         for i in range(len(curr_line_processing)):
#             if curr_line_processing[i] == '?':
#                 curr_line_processing[i] = 0.0
#             curr_line_processing[i] = float(curr_line_processing[i])
#         curr_line_processing = curr_line_processing[:2] + curr_line_processing[3:24]
#         if curr_line_processing[21] == 3.0 or curr_line_processing[21] == 2.0:
#             curr_line_processing[21] = 1.0
#         else:
#             curr_line_processing[21] = 0.0
#         test_data.append(curr_line_processing[:21])
#         test_labels.append(curr_line_processing[21])
#     # 将test_data数据带入，并与test_labels的数据比较
#     for i in range(len(test_data)):
#         if int(classify(np.array(test_data[i]), train_paras)) != int(test_labels[i]):
#             err_count += 1
#     err_rate = float(err_count/num)
#     print("这次测试的错误率为%f" % err_rate)
#     # 错误率可以明显看出是几个固定值的随机选一个，在同一次测试中也会出现几个相同的数，Why？？？？？
#     return err_rate
#
#
# def sigmoid(x):
#     try:
#         n = 1.0/(1 + np.exp(-x))
#     except OverflowError:
#         n = 1.0
#     return n
#
#
# def classify(x, paras):
#     prob = sigmoid(sum(x * paras))
#     if prob > 0.5:
#         return 1.0
#     else:
#         return 0.0
#
#
# def stochastic_gradient_descent(data_matrix, labels, num_iteration=500):
#     m, n = np.shape(data_matrix)
#     paras = np.ones(n)
#     for j in range(num_iteration):
#         data_index = list(range(m))
#         for i in range(m):
#             pace_length = 4 / (1.0 + i + j) + 0.0001
#             rand_index = int(np.random.uniform(0, len(data_index)))
#             h = sigmoid(sum(data_matrix[rand_index] * paras))
#             error = labels[rand_index] - h
#             paras = paras + pace_length * error * data_matrix[rand_index]
#             del(data_index[rand_index])
#     return paras


# def multi_test(test_num=10):
#     err_rate_sum = 0.0
#     for i in range(test_num):
#         err_rate_sum += colic_test()
#     print('完成%d次迭代之后，平均错误率为%f' % (test_num, float(err_rate_sum)/test_num))



