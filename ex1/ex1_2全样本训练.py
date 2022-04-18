# %%
import numpy as np
from numpy.linalg import *
import csv
import math
import random

# %%
# 读入数据
def read_deal(path):
    file_reader = csv.reader(open(path,"r",encoding='utf-8'))
    data=[]
    for i in file_reader:
        data.append(i)
    name=data[0]
    del(data[0])
    data=np.asarray(data).astype(np.float)
    # print(name);print(data)
 
    # 初始化omega，b，beta，x
    # np.astype(np.float) 转换数据类型
    length=len(data)
    x_hat=np.concatenate((data[:,:4].astype(np.float),np.zeros((length,1),dtype=int)+1),axis=1)         # 沿轴1连接array数组, 形成hat(x) 矩阵
    y=data[:,4].astype(np.float)                                    # 标签 1*17
    # print(x_hat); print(y)
    omega = np.zeros((length,4))+0
    b = np.zeros((length,1))+0
    beta = np.concatenate((omega,b),axis=1)
    # print(beta) 
    return x_hat,y,beta,length




# %%
# 开始计算
def train(beta,x_hat,y,length,train_cnt):
    oldbeta=beta
    newbeta=beta
    for j in range(train_cnt):
        f1=0       # 一阶导数
        f2=0       # 二阶导数
        oldbeta=newbeta
        for i in range(length):
            # 计算 wx+b
            z=np.dot(oldbeta[i,:].T,x_hat[i,:])
            # 计算p1和p0
            p1=math.exp(z)/(1+math.exp(z))
            p0=1/(1+math.exp(z))
            f1=f1-x_hat[i,:]*(y[i]-p1)
            f2=f2+np.dot(x_hat[i,:],x_hat[i,:].T)*p1*(1-p1)
        
        # print(f1,f2)
        # print(pow(f2,-1)*f1)
        newbeta=oldbeta-pow(f2,-1)*f1
        # print(oldbeta[0,:].T,x_hat[0,:],z,y[0],p1,f1,f2,1/f2*f1,newbeta[0,:])
    return newbeta

# %%
def model_part(beta, x_hat, y, length, train_cnt):
    newbeta = train(beta, x_hat, y, length, train_cnt)
    omega = newbeta[0, :4]
    b = newbeta[0, 4]
    sum = 0
    cnt = 0
    for i in x_hat:
        i = i[:4]
        if np.dot(omega, i) + b < 0:
            flag = 0
            if (flag == y[cnt]):
                sum = sum + 1
        else:
            flag = 1
            if (flag == y[cnt]):
                sum = sum + 1
        cnt = cnt + 1
    rate = sum / length
    return omega, b, rate

# %%
path="iris_training.csv"
x_hat,y,beta,length=read_deal(path)
# 将0作为正例
y0=y.copy()
for i in range(length):
    if(y0[i]==0):
        y0[i]=1
    else:
        y0[i]=0
omega0,b0,rate0=model_part(beta,x_hat,y0,length,100)
print("模型一准确率",rate0)

# 将1作为正例
y1=y.copy()
for i in range(length):
    if(y1[i]==1):
        y1[i]=1
    else:
        y1[i]=0
omega1,b1,rate1=model_part(beta,x_hat,y1,length,100)
print("模型二准确率",rate1)

# 将2作为正例
y2=y.copy()
for i in range(length):
    if(y2[i]==2):
        y2[i]=1
    else:
        y2[i]=0
omega2,b2,rate2=model_part(beta,x_hat,y2,length,100)
print("模型三准确率",rate2)

# %%
pre0=[]
pre1=[]
pre2=[]
pre=[0 for i in range(length)]

# 计算正确率
flag=0
cnt=0
sum=0

# 0为正例
for i in x_hat:
    i=i[:4]
    if np.dot(omega0,i)+b0 < 0:
        pre0=[1,2]
    else:
        pre0=[0]
    pre[cnt]=pre0
    cnt+=1


# %%
# 1为正例
flag=0
cnt=0
sum=0
for i in x_hat:
    i=i[:4]
    if np.dot(omega1,i)+b1 < 0:
        pre1=[0,2]
    else:
        pre1=[1]
    pre[cnt].extend(pre1)
    cnt+=1

# %%
# 2为正例
cnt=0
for i in x_hat:
    i=i[:4]
    if np.dot(omega2,i)+b2 < 0:
        pre2=[0,1]
    else:
        pre2=[2]
    pre[cnt].extend(pre2)
    cnt+=1

# %%
# print(pre)
ans=[0 for i in range(length)]
cnt=0
# 比较3个模型预测值数量，确定最终预测
for i in pre:
    r1=random.randint(0,1)
    r2=random.randint(0,2)
    num0=i.count(0)
    num1=i.count(1)
    num2=i.count(2)
    # print(num0,num1,num2)
    # 一家独大
    if(num0>num1 and num0>num2):
        ans[cnt]=0
    elif(num1>num0 and num1>num2):
        ans[cnt]=1
    elif(num2>num0 and num2>num1):
        ans[cnt]=2
    # 两家一样大
    elif(num0>num2 and num1>num2 and num0==num1):
        if r1==0:
            ans[cnt]=0
        else:
            ans[cnt]=1
    elif(num0>num1 and num2>num1 and num0==num2):
        if r1==0:
            ans[cnt]=0
        else:
            ans[cnt]=2
    elif(num1>num0 and num2>num0 and num1==num2):
        if r1==0:
            ans[cnt]=1
        else:
            ans[cnt]=2
    else:   #  三家一样大
        r=random.randint(0,2)
        if r2==0:
            ans[cnt]=0
        elif r2==1:
            ans[cnt]=1
        else:
            ans[cnt]=2
    cnt+=1
# print(ans)

# %%
# 准确率
sum=0
for i in range(length):
    if(ans[i]==y[i]):
        sum+=1
print("整体准确率",sum/length)




