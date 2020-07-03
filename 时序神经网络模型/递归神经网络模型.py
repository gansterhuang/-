import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import linear_system_function as lsf
import matplotlib.pyplot as plt
import copy


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
F1=np.array(np.zeros(10000,))
for i in range(0,100,2):
    F1[i*100:(i+1)*100]=-50

# F1=2*t

F3=5*np.sin(t)-t


#调用函数计算三段曲线，即调用函数求解常系数线性微分方程
x1,x2,x3=lsf.data(F1,F3)

#定义标准化矩阵的函数
def standardization(data):
    mu = np.mean(data, axis=0)
    sigma = np.std(data, axis=0)
    return (data - mu) / sigma

# #X和F由于量纲原因，数值大小差距比较大，故标准化数据
# x1=standardization(x1)
# x2=standardization(x2)
# x3=standardization(x3)
# F1=standardization(F1)
# F3=standardization(F3)



#绘制x1 x2 x3 三段位移曲线，绘制F1,F3两段载荷曲线
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




"这里构造数据集，训练集每次输入的格式为10个数据，即前5个时刻的F1,F3输入"
"训练集的结果输入格式为第5个时刻的x1,x2,x3"

data=np.zeros((8000,30),dtype=float)#定义具有（8000，30）格式的初始数组
for i in range(8000):
    data[i]=np.array([x1[i+2],x2[i+2],x3[i+2],
                   x1[i+3], x2[i+3], x3[i+3],
                   x1[i+4], x2[i+4], x3[i+4],
                   x1[i+5], x2[i+5], x3[i+5],
                   x1[i+6], x2[i+6], x3[i+6],
                   F1[i],  0.,F3[i],
                   F1[i + 1], 0.,F3[i + 1],
                   F1[i + 2], 0.,F3[i + 2],
                   F1[i + 3], 0.,F3[i + 3],
                   F1[i + 4], 0.,F3[i + 4]])
    #这里每个temp为一个样本，想法是先让网络看i+2--i+6时刻的x1,x2,x3初值，
    # 然后再依次看i---i+4时刻的F1 和 F3的值，观察是否能学习会微分方程

data_random=copy.copy(data)
np.random.shuffle(data_random)#打乱data顺序,这里data维度（8000,30）
print(np.shape(data_random))

#构造训练集

#训练集输入
data_x=np.zeros((8000,5,3),dtype=float)#定义如（8000,5,3）的格式
for i in range(8000):
    data_x[i]=[data_random[i][15:18],
        data_random[i][18:21],
        data_random[i][21:24],
        data_random[i][24:27],
        data_random[i][27:30],]

#训练集输出
data_y=np.zeros((8000,5,3),dtype=float)#定义如（8000,5,3）的格式
for i in range(8000):
    data_y[i]=[data_random[i][0:3],
        data_random[i][3:6],
        data_random[i][6:9],
        data_random[i][9:12],
        data_random[i][12:15]]


print(np.shape(data_x))
print(data_x[1])


"tensorflow 组建rnn网络部分"


#输入数组是7*3
# 序列段长度，即是几步
time_step = 5
# 隐藏层节点数目，每个LSTM内部神经元数量
rnn_unit = 30
# cell层数
lstm_layers = 4
# 序列段批处理数目
batch_size = 50
# batch数目
n_batch=8000//batch_size
# 输入维度
input_size = 3
# 输出维度
output_size = 3
# 学习率
lr = 0.00006

#这里的none表示第一个维度可以是任意值
x=tf.placeholder(tf.float32,[None,time_step, input_size])#
y=tf.placeholder(tf.float32,[None, time_step, output_size])



#定义输入输出权值及偏置值
'同时注明一点，这里的bias及weights的写法必须要是这样，后面的saver函数才能正常调用'
weights = {
    'in': tf.Variable(tf.random_normal([input_size, rnn_unit])),
    'out': tf.Variable(tf.constant(0.1, shape=[rnn_unit, output_size]))
}
biases = {
    'in': tf.Variable(tf.random_normal([rnn_unit])),
    'out': tf.Variable(tf.constant(0.1, shape=[output_size]))
}




def lstm(batch):

    # 定义输入的权值及偏置值
    w_in = weights['in']
    b_in = biases['in']

    # 对输入rnn网络的数据做前置处理
    input = tf.reshape(x, [-1, input_size])  #x被置位（bacth_size*time_step,input_size） 的格式（250,3）
    input_rnn = tf.matmul(input, w_in) + b_in
    input_rnn = tf.reshape(input_rnn, [-1, time_step, rnn_unit])#这里是真实输入rnn网络格式的数据

    cell = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.BasicLSTMCell(rnn_unit) for _ in range(lstm_layers)])
    init_state = cell.zero_state(batch, dtype=tf.float32)

    output_rnn, final_states = tf.nn.dynamic_rnn(cell, input_rnn, initial_state=init_state, dtype=tf.float32)

    output = tf.reshape(output_rnn, [-1, rnn_unit])

    print(tf.shape(output))

    # 定义输出权值及偏置值，并对LSTM的输出值做处理
    w_out = weights['out']
    b_out = biases['out']
    pred = tf.matmul(output, w_out) + b_out

    return pred, final_states



def train_lstm():
    global batch_size
    with tf.variable_scope("sec_lstm"):
        pred, _ = lstm(batch_size)

    loss = tf.reduce_mean(tf.square(tf.reshape(pred, [-1]) - tf.reshape(y, [-1])))
    train_op = tf.train.AdamOptimizer(lr).minimize(loss)

    #定义变量存储参数
    saver = tf.train.Saver(tf.global_variables())
    loss_list = []


    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(500):  # We can increase the number of iterations to gain better result.

            for i in range(n_batch):

                _, loss_ = sess.run([train_op, loss], feed_dict={x: data_x[batch_size * i:batch_size * (i + 1)],
                                                                 y: data_y[batch_size * i:batch_size * (i + 1)]})

                loss_list.append(loss_)



                if epoch % 10 == 0:
                    print("Number of epoch:", epoch, " loss:", loss_list[-1])

                    if epoch > 0 and loss_list[-2] > loss_list[-1]:
                        saver.save(sess, 'model_save1\\modle.ckpt')


train_lstm()#运行LSTM网络




"测试部分"

#测试集输入，
test_number=8000

test_x=np.zeros((test_number,5,3),dtype=float)#定义如（8000,5,3）的格式
for i in range(test_number):
    test_x[i]=[data[i][15:18],
        data[i][18:21],
        data[i][21:24],
        data[i][24:27],
        data[i][27:30]]  #这里的test_x实则为F1 F3不同时刻输入的组合

print(np.shape(test_x))


#用模型计算得出的输出
with tf.variable_scope("sec_lstm", reuse=tf.AUTO_REUSE):
    pred, _ = lstm(1)  #这里预测，所以输入的batch_size为1就可以
saver = tf.train.Saver(tf.global_variables())
with tf.Session() as sess:
    saver.restore(sess, 'model_save1\\modle.ckpt')


    predict_x1 = np.array([])
    predict_x2 = np.array([])
    predict_x3 = np.array([])
    for i in range(test_number):
        next_seq = sess.run(pred, feed_dict={x: [test_x[i]]})  #next_seq输出格式为（time_step,output_size）,这里为（5,3,）
        np.array(next_seq)
        predict_x1=np.append(predict_x1,next_seq[0][0])
        predict_x2 = np.append(predict_x2, next_seq[0][1])
        predict_x3 = np.append(predict_x3, next_seq[0][2])


    print(np.shape(predict_x1))


    #绘制图像
    plt.subplot(511)
    plt.plot(t, x1, label='x1')
    plt.plot(t[2:2+test_number], predict_x1, color='r')

    plt.subplot(512)
    plt.plot(t, x2, label='x2')
    plt.plot(t[2:2+test_number], predict_x2, color='r')

    plt.subplot(513)
    plt.plot(t, x3, label='x3')
    plt.plot(t[2:2+test_number], predict_x3, color='r')

    plt.subplot(514)
    plt.plot(t, F1, label='F1')
    plt.subplot(515)
    plt.plot(t, F3, label='F2')

    plt.show()
