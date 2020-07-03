import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import linear_system_function as lsf
import Series_generator as sg
import matplotlib.pyplot as plt
import copy


"过程说明及样本构造" \
'''
#系统有质量分别为m1,m2,m3的小车构成，m1通过并联的k1弹簧、c1阻尼器与墙壁相连
#m2通过并联的k2弹簧、c2阻尼器与m1相连，m3通过并联的k3弹簧、c3阻尼器与m2相连，为方便起见m1=k1=c1=1,m2=k2=c2=2,m3=k3=c3=3,

'''



#计算100s内的位移情况，计算步长为0.01s
dt=0.1
t=np.array(np.linspace(0.1,100,1000))#此处定义步长，总步数要与lsf funtion里对应上

#定义训练集样本数量
data_number=500


#F1与F3输入
# F1=np.array(np.zeros(1000,))
# for i in range(0,100,2):
#     F1[i*10:(i+1)*10]=-50
#
# # F1=t*0
#
F3=5*np.sin(t)-t

# F3=(t/5.0)*np.sin(t)

F1=np.array(np.linspace(0,10,1000))
F1[400:600]=0


#调用函数计算三段曲线，即调用函数求解常系数线性微分方程
x1,x2,x3=lsf.data(F1,F3)

# x1=x1+np.random.normal(0,1,1000)

#定义标准化矩阵的函数
def standardization(data):
    mu = np.mean(data, axis=0)
    sigma = np.std(data, axis=0)
    return (data - mu) / sigma

#X和F由于量纲原因，数值大小差距比较大，故标准化数据
x1=standardization(x1)
x2=standardization(x2)
x3=standardization(x3)
F1=standardization(F1)
F3=standardization(F3)



#绘制x1 x2 x3 三段位移曲线，绘制F1,F3两段载荷曲线
plt.subplot(511)
plt.plot(t,x1,label='x1')
plt.title('x1_curve')
plt.subplot(512)
plt.plot(t,x2,label='x2')
plt.title('x2_curve')
plt.subplot(513)
plt.plot(t,x3,label='x3')
plt.title('x3_curve')
plt.subplot(514)
plt.plot(t,F1,label='F1')
plt.title('F1_curve')
plt.subplot(515)
plt.plot(t,F3,label='F3')
plt.title('F3_curve')

plt.show()




"这里构造数据集，训练集每次输入的格式为10个数据，即前5个时刻的F1,F3输入"
"训练集的结果输入格式为第5个时刻的x1,x2,x3"

data=np.zeros((data_number,5),dtype=float)#定义具有（500，5）格式的初始数组
for i in range(data_number):
    data[i]=np.array([x2[i],x2[i+1],x2[i+2],x2[i+3],x1[i+4]])
            #这里数组里前4个数为输入，后一个为输出

data_random=copy.copy(data)
np.random.shuffle(data_random)#打乱data顺序,这里data维度（500,5）
print(np.shape(data_random))

#构造训练集

#训练集输入
data_x=np.zeros((data_number,4,1),dtype=float)#定义如（500,4）的格式
for i in range(data_number):
    data_x[i]=[[data_random[i][0]],
               [data_random[i][1]],
               [data_random[i][2]],
               [data_random[i][3]]
               ]


#训练集输出
data_y=np.zeros((data_number,1),dtype=float)#定义如（500,）的格式
for i in range(data_number):
    data_y[i]=[data_random[i][4]]



print(np.shape(data_x))
print(data_x[1])


"tensorflow 组建rnn网络部分"



# 序列段长度，即是几步
time_step = 4
# 隐藏层节点数目，每个LSTM内部神经元数量
rnn_unit = 20
# cell层数
gru_layers = 2
# 序列段批处理数目
batch_size = 20
# batch数目
n_batch=data_number//batch_size
# 输入维度
input_size = 1
# 输出维度
output_size = 1
# 学习率
lr = 0.0003
#前置网络的隐藏层神经元数量
hidden_size=5

#输出层网络的隐藏层神经元数量
out_hidden_size=5

#这里的none表示第一个维度可以是任意值
x=tf.placeholder(tf.float32,[None,time_step, input_size])#
y=tf.placeholder(tf.float32,[None, output_size])



#定义输入输出权值及偏置值
'同时注明一点，这里的bias及weights的写法必须要是这样，后面的saver函数才能正常调用'
weights = {
    'in': tf.Variable(tf.random_normal([input_size, hidden_size])),
    'in_hidden': tf.Variable(tf.random_normal([hidden_size, rnn_unit])),
    'out_hidden': tf.Variable(tf.random_normal([rnn_unit, out_hidden_size])),
    'out': tf.Variable(tf.constant(0.1, shape=[out_hidden_size, output_size]))
}
biases = {
    'in': tf.Variable(tf.random_normal([hidden_size])),
    'in_hidden': tf.Variable(tf.random_normal([rnn_unit])),
    'out_hidden': tf.Variable(tf.random_normal([out_hidden_size])),
    'out': tf.Variable(tf.constant(0.1, shape=[output_size]))
}


'''
结论上来说，如果cell为LSTM，那 state是个tuple，分别代表Ct 和 ht，其中 ht与outputs中的对应的最后一个时刻的输出相等，
假设state形状为[ 2，batch_size, cell.output_size ]，outputs形状为 [ batch_size, max_time, cell.output_size ]，
那么state[ 1, batch_size, : ] == outputs[ batch_size, -1, : ]；如果cell为GRU，那么同理，state其实就是 ht，state ==outputs[ -1 ]
'''
'''
LSTM的输入（batch_size,time_step,input_size）
LSTM的output(batch_size,time_step,hidden_units)
LSTM的state输出（2，batch_size,hidden_units）其中这里的2是由于Ct和ht拼接而成的

GRU的输入（batch_size,time_step,input_size）
GRU的output(batch_size,time_step,hidden_units)
GRU的state输出（batch_size,hidden_units）

'''


def lstm(batch):

    # 定义输入的权值及偏置值
    w_in = weights['in']
    b_in = biases['in']

    w_hidden = weights['in_hidden']
    b_hidden = biases['in_hidden']


    # 对输入rnn网络的数据做前置处理
    input = tf.reshape(x, [-1, input_size])  #x被置位（bacth_size*time_step,input_size） 的格式（250,3），由于是三维数组无法直接乘，需要进行处理


    #前置网络的隐藏层处理
    input_hidden=tf.nn.relu(tf.matmul(input, w_in) + b_in)

    input_rnn = tf.nn.relu(tf.matmul(input_hidden, w_hidden) + b_hidden)

    input_rnn = tf.reshape(input_rnn, [-1, time_step, rnn_unit])#这里是真实输入rnn网络格式的数据

    #定义GRU网络的参数
    GRU_cells = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.GRUCell(rnn_unit) for _ in range(gru_layers)])

    init_state = GRU_cells.zero_state(batch, dtype=tf.float32)

    output_rnn, final_states = tf.nn.dynamic_rnn(GRU_cells, input_rnn, initial_state=init_state, dtype=tf.float32)

    '''
    由于本网络是由两个GRU单元堆叠而成，所以state的输出是2*hidden_unit的，state要取[-1]
    '''
    output = tf.reshape(final_states[-1], [-1, rnn_unit])


    # 定义输出权值及偏置值，并对LSTM的输出值做处理
    w_out_hidden=weights['out_hidden']
    b_out_hidden = biases['out_hidden']

    w_out = weights['out']
    b_out = biases['out']

    out_hidden=tf.nn.relu(tf.matmul(output,w_out_hidden)+b_out_hidden)
    pred = tf.matmul(out_hidden, w_out) + b_out

    print(pred.shape.as_list())#查看pred维数

    return pred, final_states



def train_lstm():
    global batch_size
    with tf.variable_scope("sec_lstm"):
        pred, _ = lstm(batch_size)

    loss = tf.reduce_mean(tf.square(pred - y))
    train_op = tf.train.AdamOptimizer(lr).minimize(loss)

    #定义变量存储参数
    saver = tf.train.Saver(tf.global_variables())
    loss_list = []


    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(1000):  # We can increase the number of iterations to gain better result.

            for i in range(n_batch):

                _, loss_ = sess.run([train_op, loss], feed_dict={x: data_x[batch_size * i:batch_size * (i + 1)],
                                                                 y: data_y[batch_size * i:batch_size * (i + 1)]})

                loss_list.append(loss_)



                if epoch % 10 == 0:
                    print("Number of epoch:", epoch, " loss:", loss_list[-1])

                    if epoch > 0 and loss_list[-2] > loss_list[-1]:
                        saver.save(sess, 'model_save1\\GRU_curve_fit_modle_1.ckpt')


train_lstm()#运行LSTM网络




"测试部分"

#测试集输入，
test_number=950
test_data=np.zeros((test_number,4,1),dtype=float)#定义具有（500，4,1）格式的初始数组
for i in range(test_number):
    test_data[i]=np.array([[x2[i]],[x2[i+1]],[x2[i+2]],[x2[i+3]]])
            #这里数组里前4个数为输入，后一个为输出


#用模型计算得出的输出
with tf.variable_scope("sec_lstm", reuse=tf.AUTO_REUSE):
    prey, _ = lstm(1)  #这里预测，所以输入的batch_size为1就可以
saver = tf.train.Saver(tf.global_variables())

with tf.Session() as sess:
    saver.restore(sess, 'model_save1\\GRU_curve_fit_modle_1.ckpt')


    pre_list=[]
    for i in range(test_number):
        next_seq = sess.run(prey, feed_dict={x: [test_data[i]]})

        pre_list.append(next_seq)

    pre_list=np.reshape(pre_list,(950,))

    #绘制图像
    plt.plot(t, x1)
    plt.plot(t[5:5+test_number], pre_list, color='r')


    plt.show()
