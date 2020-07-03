import numpy  as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

'''
系统有质量分别为m1,m2,m3的小车构成，m1通过并联的k1弹簧、c1阻尼器与墙壁相连
m2通过并联的k2弹簧、c2阻尼器与m1相连，m3通过并联的k3弹簧、c3阻尼器与m2相连，为方便起见m1=k1=c1=1,m2=k2=c2=2,m3=k3=c3=3,
'''



#计算100s内的位移情况，计算步长为0.01s
dt=0.1
step_number=1000

t=np.array(np.linspace(0.1,100,1000))#此处从0.01开始，到100，正好10000步

#F1与F3输入
# F3=5*np.sin(t)-t
# for i in range(0,100,4):
#     F3[i*10:(i+1)*10]=-40

F3=(t/5.0)*np.sin(t)

F1=np.array(np.linspace(0,10,1000))
F1[400:600]=0


def data(input_1,input_2):   #input格式为(10000,)
    #状态空间有6个变量，定义占位数组
    x1=np.array(np.linspace(0.1,100,1000))
    x2=np.array(np.linspace(0.1,100,1000))
    x3=np.array(np.linspace(0.1,100,1000))
    x4=np.array(np.linspace(0.1,100,1000))
    x5=np.array(np.linspace(0.1,100,1000))
    x6=np.array(np.linspace(0.1,100,1000))


    #状态空间离散化，进行求解
    for i in range(step_number-1):#此处减去1的目的是考虑到最后一步的xi【】+1，不减1会造成数组多一个数
        x1[i+1]=x4[i]*dt+x1[i]
        x2[i + 1] = x5[i] * dt+x2[i]
        x3[i + 1] = x6[i] * dt+x3[i]
        x4[i+1] = (-3*x1[i]+2*x2[i]-3*x4[i]+2*x5[i]+input_1[i]) * dt+x4[i]
        x5[i+1]=(x1[i]-2.5*x2[i]+1.5*x3[3]+x4[4]-2.5*x5[i]+1.5*x6[i])*dt+x5[i]
        x6[i+1]=(x2[i]-x3[i]+x5[i]-x6[i]+(1/3)*input_2[i])*dt+x6[i]

    return x1,x2,x3

data_1,data_2,data_3=data(F1,F3)

# #绘制x1 x2 x3 三段位移曲线
# plt.subplot(511)
# plt.plot(t,data_1)
# plt.title('x1_curve')
# plt.subplot(512)
# plt.plot(t,data_2)
# plt.title('x2_curve')
# plt.subplot(513)
# plt.plot(t,data_3)
# plt.title('x3_curve')
#
# plt.subplot(514)
# plt.plot(t,F1)
# plt.title('F1_curve')
# plt.subplot(515)
# plt.plot(t,F3)
# plt.title('F3_curve')
#
# plt.show()