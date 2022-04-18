# %%
import numpy as np
import random
import math
import csv
import sys
import matplotlib.pyplot as plt  # 约定俗成的写法plt
from sklearn.model_selection import train_test_split  # 只是用来划分数据集和测试集

# %%
def sigmoid(x):  # 激活函数
    return 1 / (1 + math.pow(math.e, -x))


def MSL(y1, y2):  # 均方误差，对应平方和
    return sum(np.power(y1 - y2, 2)) * 0.5


# %%
# 读入数据
def read_deal(path):
    file_reader = csv.reader(open(path, "r"))
    data = []
    for i in file_reader:
        data.append(i)
    name = data[0]
    del (data[0])
    data = np.asarray(data).astype(np.float)

    length = len(data)          # 记录样本数量
    x_len = len(data[0]) - 1    # 每个样本中属性值数量
    x = data[:, :x_len]
    y_true = data[:, x_len]     # 每个样本对应的种类标签
    s=set(y_true)               
    y_len=len(s)                # 所有样本的种类数量
    y = np.zeros((length, y_len), dtype=float)      # 扩展成独热码，这样可以用输出层表示
    for i in range(length):
        y[i, int(y_true[i])] = 1
    return name, length, x, y , x_len, y_len


# %%
# 开始训练
def train(length, lay_in, lay_hide, lay_out, x, y, train_cnt,n):
    loss = []  # MSL
    v=np.random.random((lay_in,lay_hide))   # 输入到隐层的激活系数
    u=np.random.random(lay_hide)         # 隐层阈值
    w=np.random.random((lay_hide,lay_out)) # 隐层到输出的激活系数
    o=np.random.random(lay_out)         # 输出层阈值
    for c in range(train_cnt):
        sys.stderr.write("\r迭代次数:" + str(c+1) + "\\" + str(train_cnt))
        sys.stderr.flush()
        L = 0   # 总均方误差
        for i in range(length):
            a = np.dot(x[i], v)  # 和u组成隐藏层输入
            b = np.zeros(lay_hide)  # 得到隐藏层输出
            for j in range(lay_hide):
                b[j] = sigmoid(a[j] - u[j])
            beta = np.dot(b, w)  # 和o组成输出层输入
            out = np.zeros(lay_out)  # 输出层输出
            for j in range(lay_out):
                out[j] = sigmoid(beta[j] - o[j])
            g = out * (1 - out) * (y[i] - out)
            # 计算Δw
            w_del = np.zeros((lay_hide, lay_out))
            for j in range(lay_hide):
                w_del[j] = n * b[j] * g
            # 计算Δθ
            o_del = -n * g
            # 计算Δv
            v_del = np.zeros((lay_in, lay_hide), dtype=float)
            # 计算Δγ
            u_del = np.zeros(lay_hide)
            e = np.zeros(lay_hide)
            for h in range(lay_hide):
                e[h] = b[h] * (1 - b[h]) * (sum(w[h] * g))

            for j in range(lay_in):
                v_del[j] = n * x[i, j] * e
            u_del = -n * e

            w = w + w_del  # 隐藏层到输出层的系数更新
            o = o + o_del  # 输出层输入阈值的更新
            v = v + v_del  # 输入层到隐藏层的系数更新
            u = u + u_del  # 隐藏层输入阈值的更新
            L += MSL(y[i], out)
        loss.append(L)  # 记录这次迭代的均方误差
    return w, o, v, u, loss


# %%
# 对数据处理得到对应输入x和输出y矩阵
# name, length, x, y, x_len, y_len = read_deal("../ex1/iris_training.csv")
name, length, x, y, x_len, y_len = read_deal("car.csv")
# 网络初始化参数
lay_in = x_len
lay_hide = x_len+2
lay_out = y_len
epoch = 100     # 迭代次数
n = 0.1         # 学习率
# 划分数据集
train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.3)
train_x = np.array(train_x)
train_y = np.array(train_y)
test_x = np.array(test_x)
test_y = np.array(test_y)
test_len=len(test_x)        # 测试次数
train_len=length-test_len   # 训练次数
# 训练网络得到系数矩阵和偏移量
w, o, v, u, loss = train(train_len, lay_in, lay_hide, lay_out, train_x, train_y, epoch,n)
print()
print("v=",v)
print("u=",u)
print("w=",w)
print("o=",o)
cnt = 0
for i in range(test_len):
    a = np.dot(test_x[i], v)  # 和u组成隐藏层输入
    b = np.zeros(lay_hide)  # 得到隐藏层输出
    for j in range(lay_hide):
        b[j] = sigmoid(a[j] - u[j])
    beta = np.dot(b, w)  # 和o组成输出层输入
    out = np.zeros(lay_out)  # 输出层输出
    for j in range(lay_out):
        out[j] = sigmoid(beta[j] - o[j])

    # print(np.argmax(out))
    if (np.argmax(out) == np.argmax(test_y[i])):
        cnt += 1

# print(loss)
plt.plot(loss)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()
print("正确率:%f" % float(cnt / test_len))
