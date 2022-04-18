import numpy as np
from numpy.linalg import *
import csv
import math

# 格式化数据
file_reader = csv.reader(open("ML/ex1/watermelon3.csv", "r", encoding='utf-8'))
file_writer = csv.writer(open("ML/ex1/data_exe.csv", "w", encoding='utf-8', newline=''))
data_pre = []
for i in file_reader:
    data_pre.append(i)
name = data_pre[0]
# 用一个变量记录表头后，去除data_pre中的表头
del (data_pre[0])
data_exe = data_pre
# print(data_exe)
# 字符串转数字
seze = {"青绿": 1, "乌黑": 0.5, "浅白": 0}
gendi = {"蜷缩": 1, "稍蜷": 0.5, "硬挺": 0}
qiaosheng = {"浊响": 1, "沉闷": 0.5, "清脆": 0}
wenli = {"清晰": 1, "稍糊": 0.5, "模糊": 0}
qibu = {"凹陷": 1, "稍凹": 0.5, "平坦": 0}
chugan = {"硬滑": 1, "软粘": 0}
haogua = {"是": 1, "否": 0}
# 0编号	1色泽	2根蒂	3敲声	4纹理	5脐部	6触感	7密度	8含糖率	 9好瓜
for i in data_exe:
    i[1] = seze[i[1]]
    i[2] = gendi[i[2]]
    i[3] = qiaosheng[i[3]]
    i[4] = wenli[i[4]]
    i[5] = qibu[i[5]]
    i[6] = chugan[i[6]]
    i[9] = haogua[i[9]]

# 将处理完的列表数据写入文件
file_writer.writerow(name)  # 写入表头
file_writer.writerows(data_exe)
data = np.asarray(data_exe)
# print(data)

length = len(data)
train_cnt = 1000  # 训练次数
# 7:3划分数据集
train_len = length * 7 // 10
# 将字符串数组转换为float类型  np.astype(np.float) 转换数据类型
train_data = np.concatenate(
    (data[0:train_len // 2].astype(np.float), data[8:8 + train_len - train_len // 2].astype(np.float)))
test_data = np.concatenate(
    (data[train_len // 2:8].astype(np.float), data[8 + train_len - train_len // 2:].astype(np.float)))

train_x = np.concatenate((train_data[:, 1:7], np.zeros((train_len, 1), dtype=int) + 1),
                         axis=1)  # 沿轴1连接array数组, 形成hat(x) 矩阵
train_y = train_data[:, 9]  # 标签 1*17
# print(train_x); print(train_y)
# 初始化omega，b，beta，x
omega = np.zeros((train_len, 6)) + 0.5
b = np.zeros((train_len, 1)) + 0.3
beta = np.concatenate((omega, b), axis=1)


# print(beta); print(beta[:,:6]) #omega
# 开始计算，牛顿迭代法
def train(beta, train_x, train_y, train_cnt):
    newbeta = beta
    for j in range(train_cnt):
        f1 = 0  # 一阶导数
        f2 = 0  # 二阶导数
        oldbeta = newbeta
        for i in range(train_len):
            # 计算 wx+b
            z = np.dot(oldbeta[i, :].T, train_x[i, :])
            # print(z)
            # 计算p1和p0
            p1 = math.exp(z) / (1 + math.exp(z))
            p0 = 1 / (1 + math.exp(z))
            f1 = f1 - train_x[i, :] * (train_y[i] - p1)
            f2 = f2 + np.dot(train_x[i, :], train_x[i, :].T) * p1 * (1 - p1)
            # print(train_y[i],p1,p0)
        # print(f1,f2)
        # print(pow(f2,-1)*f1)
        newbeta = oldbeta - pow(f2, -1) * f1
    return newbeta


# 得到结果
newbeta = train(beta, train_x, train_y, train_cnt)
omega = newbeta[0, :6]
b = newbeta[0, 6]

# 计算正确率
pre = 0  # 记录预测值
cnt = 0  # 对测试数量计数
sum = 0  # 对正确值计数
test_x = np.concatenate((test_data[:, 1:7], np.zeros((length - train_len, 1), dtype=int) + 1), axis=1)
test_y = test_data[:, 9]
# 判断预测正确性
for i in test_x:
    i = i[:6]
    if np.dot(omega, i) + b < 0:
        pre = 0
        if (pre == test_y[cnt]):
            sum = sum + 1
            print("数据", test_data[cnt, :-1], "是反例")
        else:
            print("数据", test_data[cnt, :-1], "判断错误")
    else:
        pre = 1
        if (pre == test_y[cnt]):
            sum = sum + 1
            print("数据", test_data[cnt, :-1], "是正例")
        else:
            print("数据", test_data[cnt, :-1], "判断错误")
    cnt = cnt + 1
print("正确率", sum / (length - train_len))
