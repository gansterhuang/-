import numpy as np
import matplotlib.pyplot as plt
import copy
import math


#定义数据点
x_=np.linspace(1,3,100)
print(np.shape(x_))

y_=np.zeros(100,)
for i in range(100):
    y_[i]=(x_[i]-2)**2+1

x_measure=x_+np.random.normal(0,0.01,100)
y_measure=y_+np.random.normal(0,0.01,100)

data=np.array([x_measure,y_measure])
print(np.shape(data))

#plt.scatter(x_measure,y_measure)
#plt.show()



'''
PKPCA代码部分
'''
#核函数参数
thegama=3
#均方根参数
rho=0.0001
#定义潜隐变量维度q
q=10  #q《 time_step


# 定义标准化矩阵的函数
def standardization(data):
    # 读取data的数据格式
    n_sensor = np.shape(data)[0]
    time_step = np.shape(data)[1]

    # 计算每个sensor的均值和方差
    mu = np.mean(data, axis=1)
    sigma = np.std(data, axis=1)

    # 数据处理
    con = np.zeros((n_sensor, time_step))
    for i in range(n_sensor):
        con[i] = (data[i] - mu[i]) / sigma[i]

    return con, mu, sigma


def kernal_fun(x1,x2):
    #采用高斯核函数
    len=np.shape(x1)[0]
    temp=0.0
    for i in range(int(len)):
        temp=temp+(x1[i]-x2[i])**2

    kernal=-np.e**(temp/(2*thegama**2))
    return kernal



#input 为数据
'''
定义PKPCA函数，由于该方法的大部分过程无法显示表达，
故可以做计算的部分只有重构误差
'''
#def PKPCA(input):


input,_,_=standardization(data)

#获取输入数据维度
n_senor=np.shape(input)[0]
N=time_step = np.shape(input)[1]



'''
计算核函数矩阵
'''
input_tran=np.transpose(input)#将数据矩阵转置，方便获取每个数据点的信息

#核函数矩阵初始化
K0=np.zeros((N,N))
#为K0赋值
for i in range(N):
    for j in range(N):
        K0[i][j]=kernal_fun(input_tran[i],input_tran[j])


# 对K0求特征值及特征向量
evals, evecs = np.linalg.eig(K0)

topk_evecs = np.real(evecs[:, range(q)] ) # 提取出前q大特征值对应的特征向量

topk_evals =  np.real(evals[0:q] ) # 提取出前q大特征值对应的特征值


#定义式中重要计算参数
Vq_matrix=topk_evecs
print('Vq矩阵的维度',np.shape(Vq_matrix))

Numda_q_matrix=np.identity(q)
for i in range(q):
    Numda_q_matrix[i][i]=topk_evals[i]
print('Numda_q_matrix矩阵的维度',np.shape(Numda_q_matrix))

S_matrix=np.reshape(np.array([(1/np.float(N)) for i in range(N)]),(N,1)) # S_matrix 格式为（N,1）
print('S_matrix矩阵的维度',np.shape(S_matrix))
J_matrix=np.float(N)**(-0.5)*(np.identity(N)-np.matmul(S_matrix,np.reshape(np.array([1.0 for _ in range(N)]),(1,N))))
print('J_matrix矩阵的维度',np.shape(J_matrix))


#Q矩阵中间的矩阵
Q_middle=np.identity(q)-rho*np.linalg.inv(Numda_q_matrix)#计算出初始矩阵
for i in range(q):
    Q_middle[i][i]=(Q_middle[i][i])**0.5
print('Q_middle矩阵的维度',np.shape(Q_middle))

#R矩阵
R_matrix=np.identity(q)
print('R_matrix',np.shape(R_matrix))

#计算得到R矩阵
Q_matrix=np.matmul(np.matmul(Vq_matrix,Q_middle),R_matrix)
print('Q_matrix',np.shape(Q_matrix))


'''


到这里从训练集定义的参数已经被定义完全，
下面相关表达式皆和新数据点相关


'''

def reconstruction_error(x_new):
    #kx_new参数
    kx_new=np.array([1 for _ in range(N)])
    for i in range(N):
        kx_new[i]=kernal_fun(x_new,input_tran[i])

    kx_new=np.reshape((kx_new),(N,1))
    #print('kx_new',np.shape(kx_new))


    #hx_new参数
    hx_new=kx_new-np.matmul(K0,S_matrix)
    #print('hx_new',np.shape(hx_new))

    #gx_new参数
    gx_new=kernal_fun(x_new,x_new)-2*np.matmul(np.transpose(kx_new),S_matrix)+np.matmul(np.matmul(np.transpose(S_matrix),K0),S_matrix)
    #print('gx_new',np.shape(gx_new))

    '''
    计算重构误差
    '''
    hyTJQ_matrix=np.matmul(np.matmul(np.transpose(hx_new),J_matrix),Q_matrix)
    #print('hyTJQ_matrix',np.shape(hyTJQ_matrix))

    K_matrix=np.matmul(np.matmul(np.transpose(J_matrix),K0),J_matrix)
    #print('K_matrix',np.shape(K_matrix))

    Res_middle=np.linalg.inv(np.matmul(np.matmul(np.transpose(Q_matrix),K_matrix),Q_matrix))
    #print('Res_middle',np.shape(Res_middle))

    Reconstruction=gx_new-np.matmul(np.matmul(hyTJQ_matrix,Res_middle),np.transpose(hyTJQ_matrix))

    return Reconstruction
print(reconstruction_error([0.1,0.1]))

#计算原数据集的重构误差
res_list=np.zeros((N,))
for i in range(N):
    res_list[i]=reconstruction_error(input_tran[i])


plt.scatter(input[0],input[1])
plt.plot(input[0],res_list)
plt.show()