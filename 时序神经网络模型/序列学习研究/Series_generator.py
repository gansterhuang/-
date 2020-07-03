'''
本文件在于生成一些具有较强序列特征的data，供深度学习使用

'''

import numpy as np
import matplotlib.pyplot as plt
import copy
import math
from mpl_toolkits.mplot3d import Axes3D


def series_1():
    x=np.zeros((1000,))
    x[0] = 1
    x[1]=1
    x[2]=2
    x[3]=3
    for i in range(4,1000,1):
        x[i]=(np.sin(x[i-1])+np.cos(x[i-3])+2.0)/(np.sin(x[i-3])+np.cos(x[i-2])+1.0)

    return x

def series_2():
    x = np.zeros((1000,))
    x[0]=1
    x[1] = 1
    x[2] = 2
    x[3] = 3
    for i in range(4, 1000, 1):
        x[i] = ((np.sin(x[i - 1]) + 5*np.log(x[i - 3]) + 2.0) / (np.sin(x[i - 3]) + 0.1*np.log(x[i - 2]) + 1.0))*np.sin(x[i-3])

    return x

def series_3():
    x = np.zeros((1000,))
    x[0] = 1
    x[1] = 1
    x[2] = 2
    x[3] = 3
    for i in range(4, 1000, 1):
        x[i] = (0.01*(np.sin(x[i - 1]) + 5*np.log(x[i - 3]) + 2.0) / ((np.sin(x[i - 3])+np.log(x[i - 2]))/(np.sin(x[i-2]+2)) + 1.0)+40)*(np.sin(x[i-2])+2)*(np.e**(np.sin(x[i-1]+2)))

    return x



def series_4(b,c):
    x = np.zeros((1000,))
    x[0] = 1
    x[1] = 1
    x[2] = 2
    x[3] = 4
    for i in range(4, 1000, 1):
        x[i] = (c[i-1]/b[i-1]**3)-0.5*c[i-2]-b[i-3]*0.1*(x[i-2]/b[i-2]**2)+x[i-1]/c[i-1]+c[i-3]*(0.5)

    return x

def series_5(a,b,c):
    x = np.zeros((1000,))
    x[0] = 1
    x[1] = 1
    x[2] = 2
    x[3] = 4
    for i in range(4, 1000, 1):
        x[i] = a[i-1]-(c[i-1] / b[i - 1] ** 3) - 0.5 * c[i - 2] - b[i - 3] * 0.1 * (x[i - 2] / c[i - 2] ** 2) + x[i - 1] / c[
            i - 1] +a[i-2]* c[i - 3] * (0.05)+x[i-1]*0.1/x[i-2]

    return x
















'''
分割线，上面为原生数组
'''
def series_4_noise():
    d_sensor = d# + np.random.normal(0, 3, 1000)
    return d_sensor

def series_3_noise():
    c_sensor = c #+ np.random.normal(0, 3, 1000)
    return c_sensor

def series_2_noise():
    b_sensor = b# + np.random.normal(0, 3, 1000)
    return b_sensor


a=series_1()
b=series_2()
c=series_3()
d=series_4(b,c)
e=series_5(a,b,c)



b_sensor=b+np.random.normal(0,0.3,1000)
c_sensor=c+np.random.normal(0,3,1000)
d_sensor=d+np.random.normal(0,3,1000)


plt.subplot(511)
plt.plot(range(np.shape(a)[0]),a)
plt.subplot(512)
plt.plot(range(np.shape(b)[0]),b)
plt.subplot(513)
plt.plot(range(1000),c)
plt.subplot(514)
plt.plot(range(np.shape(d)[0]),d)
plt.subplot(515)
plt.plot(range(np.shape(e)[0]),e)


plt.show()
