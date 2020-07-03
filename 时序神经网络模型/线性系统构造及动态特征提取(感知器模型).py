import numpy  as np
import matplotlib.pyplot as plt
import linear_system_function as lsf

import tensorflow as tf
import copy
#import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"#指定在第0块GPU上跑


"过程说明及样本构造" \
"" \
"" \
""
#系统有质量分别为m1,m2,m3的小车构成，m1通过并联的k1弹簧、c1阻尼器与墙壁相连
#m2通过并联的k2弹簧、c2阻尼器与m1相连，m3通过并联的k3弹簧、c3阻尼器与m2相连，为方便起见m1=k1=c1=1,m2=k2=c2=2,m3=k3=c3=3,


#计算100s内的位移情况，计算步长为0.01s
dt=0.01
t=np.array(np.linspace(0.01,100,10000))#此处定义步长，总步数要与lsf funtion里对应上

#F1与F3输入
# F1=np.array(np.zeros(10000,))
# for i in range(0,100,2):
#     F1[i*100:(i+1)*100]=-50

F1=2*t

F3=5*np.sin(t)-t


#调用函数计算三段曲线，即调用函数求解常系数线性微分方程
x1,x2,x3=lsf.data(F1,F3)

#绘制x1 x2 x3 三段位移曲线
plt.subplot(511)
plt.plot(t,x1,label='x1')
plt.subplot(512)
plt.plot(t,x2,label='x2')
plt.subplot(513)
plt.plot(t,x3,label='x3')
plt.subplot(514)
plt.plot(t,F1,label='F1')
plt.subplot(515)
plt.plot(t,F3,label='F2')

plt.show()

#构造数据样本，每组输入应为2个数据集F1,F3，输出为三个X1,X2,X3
data=np.array([F1,F3,x1,x2,x3])

draw_data=copy.copy(data)#深浅层copy的问题，留draw_data用以以后的绘图

data=np.transpose(data) #这里data输出为（10000，5）
np.random.shuffle(data)#shuffle默认对第一个维度进行打乱
#这里不再特意构造训练集，tensorflow里每次从data里取batch个作为训练集即可

print(np.shape(data))

""
"tensorflow部分"
''
''

#训练参数设定
number=5000#训练集样本总数
batch_size=25#每个批次里的样本集数量
n_batch=200




#定义placeholder
x=tf.placeholder(tf.float32,[None,2])#每次训练集输入格式为（batch_size,2）
y=tf.placeholder(tf.float32,[None,3])#每次训练集结果式为（batch_size,3）

#创建一个神经网络
W_fc1=tf.Variable(tf.truncated_normal([2,60]))
b_fc1=tf.Variable(tf.truncated_normal([60]))
h_fc1=tf.nn.tanh(tf.matmul(x,W_fc1)+b_fc1)

W_fc2=tf.Variable(tf.truncated_normal([60,60]))
b_fc2=tf.Variable(tf.truncated_normal([60]))
h_fc2=tf.nn.tanh(tf.matmul(h_fc1,W_fc2)+b_fc2)

W_fc3=tf.Variable(tf.truncated_normal([60,20]))
b_fc3=tf.Variable(tf.truncated_normal([20]))
h_fc3=tf.nn.tanh(tf.matmul(h_fc2,W_fc3)+b_fc3)

W_fc4=tf.Variable(tf.truncated_normal([20,3]))
b_fc4=tf.Variable(tf.truncated_normal([3]))
prediction=tf.matmul(h_fc3,W_fc4)+b_fc4


#二次代价函数
loss=tf.reduce_mean(tf.square(y-prediction))
#使用梯度下降法
train_step=tf.train.AdamOptimizer(0.00005).minimize(loss)

#初始化变量
init=tf.global_variables_initializer()



sess=tf.Session(config=tf.ConfigProto(log_device_placement=True)) #指定GPU计算
# sess=tf.Session()
sess.run(init)
for epoch in range(1000):

    train_x=np.reshape(np.array(data[:,(0,1)]),(10000,2))#取data前两列
    train_y = np.reshape(np.array(data[:,(2,3,4)]),(10000,3))#取data后三列

    for i in range(n_batch):

        sess.run(train_step,feed_dict={x:train_x[batch_size*i:batch_size*(i+1)],
                                       y:train_y[batch_size*i:batch_size*(i+1)]})


        if epoch % 10 == 0:
            print('loss='+str(sess.run(loss, feed_dict={x:train_x[batch_size*i:batch_size*(i+1)],
                                                        y:train_y[batch_size*i:batch_size*(i+1)]}))+'    times= '+str(epoch))


    np.random.shuffle(data)#每次epcoh结束，再次重新打乱data顺序


"" \
"" \
"" \
"测试部分"

#用训练好的模型从最初始的状态开始计算该数列
y_plot=np.array([])
for i in range(10000):
    y_plot = sess.run(prediction, feed_dict={x: np.reshape(np.array([draw_data[0],draw_data[1]]),(10000,2))})


print(np.shape(y_plot))
x1_plot=np.reshape(y_plot[:,[0]],(10000,))
x2_plot=np.reshape(y_plot[:,[1]],(10000,))
x3_plot=np.reshape(y_plot[:,[2]],(10000,))



fig = plt.figure()

#ax1部分,绘制x1输出
ax1=fig.add_subplot(511) # subplot（总共几张图，这张图画在第几列，这张图画在第几行）
ax1.plot(t,x1_plot,label='x1预测值')
ax1.plot(t,x1,label='x1实际值')

#ax2部分,绘制x2输出
ax2=fig.add_subplot(512)
ax2.plot(t,x2_plot,label='x2预测值')
ax2.plot(t,x2,label='x2实际值')

ax3=fig.add_subplot(513)
ax3.plot(t,x3_plot,label='x3预测值')
ax3.plot(t,x3,label='x3实际值')

aF1=fig.add_subplot(514)
aF1.plot(t,F1,label='F1输入')


aF3=fig.add_subplot(515)
aF3.plot(t,F3,label='F3输入')

plt.show()