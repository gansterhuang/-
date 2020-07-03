import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
import copy
#调用另外一个文件夹里的TE过程处理文件
import sys
sys.path.append('D:\我的文件夹\python_code\脱烃烷塔')
import debutainer as DBT





#定义训练集样本数量
data_number=1000

data=DBT.data_generator()#(2394,8)



#定义标准化矩阵的函数
def standardization(data):
    mu = np.mean(data, axis=0)
    sigma = np.std(data, axis=0)
    return (data - mu) / sigma


data=standardization(data)


plt.subplot(811)
plt.plot(range(np.shape(data[:,0])[0]),data[:,0])
plt.title('u1_curve')
plt.subplot(812)
plt.plot(range(np.shape(data[:,1])[0]),data[:,1])
plt.title('u2_curve')
plt.subplot(813)
plt.plot(range(np.shape(data[:,2])[0]),data[:,2])
plt.title('u3_curve')
plt.subplot(814)
plt.plot(range(np.shape(data[:,3])[0]),data[:,3])
plt.title('u4_curve')
plt.subplot(815)
plt.plot(range(np.shape(data[:,4])[0]),data[:,4])
plt.title('u5_curve')
plt.subplot(816)
plt.plot(range(np.shape(data[:,5])[0]),data[:,5])
plt.title('u6_curve')
plt.subplot(817)
plt.plot(range(np.shape(data[:,6])[0]),data[:,6])
plt.title('u7_curve')
plt.subplot(818)
plt.plot(range(np.shape(data[:,7])[0]),data[:,7])
plt.title('y_curve')
plt.show()


'''
脱烃烷塔前7列为输入数据，后一列为输出数据

'''

#对数据做一份copy
data_random=copy.copy(data)
np.random.shuffle(data_random)#打乱data顺序,这里data维度（1000,8）
print(np.shape(data_random))

#构造训练集

#训练集输入
data_x=np.zeros((data_number,4,7),dtype=float)#定义如（data_number,time_step,input_size)的格式
for i in range(data_number):
    data_x[i]=[data_random[i][0:7],
               data_random[i+1][0:7],
               data_random[i+2][0:7],
               data_random[i+3][0:7],
               ]



#训练集输出
data_y=np.zeros((data_number,1),dtype=float)#定义如（500,）的格式
for i in range(data_number):
    data_y[i]=[data_random[i+3][7]]

#将格式整理一下
data_x=np.array(data_x)
data_y=np.array(data_y)


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
input_size = 7
# 输出维度
output_size = 1
# 学习率
lr = 0.00016
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

    out_hidden=tf.nn.tanh(tf.matmul(output,w_out_hidden)+b_out_hidden)
    pred = 5*(tf.matmul(out_hidden, w_out) + b_out)

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
                        saver.save(sess, 'model_save1\\GRU_debutanizer.ckpt')


train_lstm()#运行LSTM网络




"测试部分"

#测试集输入，
test_number=2000
test_data=np.zeros((test_number,4,7),dtype=float)#定义具有（500，4,1）格式的初始数组
for i in range(test_number):
    test_data[i]=[data[i][0:7],
               data[i+1][0:7],
               data[i+2][0:7],
               data[i+3][0:7],
               ]

test_data=np.array(test_data)
            #这里数组里前4个数为输入，后一个为输出


#用模型计算得出的输出
with tf.variable_scope("sec_lstm", reuse=tf.AUTO_REUSE):
    prey, _ = lstm(1)  #这里预测，所以输入的batch_size为1就可以
saver = tf.train.Saver(tf.global_variables())


pre_list=[]
with tf.Session() as sess:
    saver.restore(sess, 'model_save1\\GRU_debutanizer.ckpt')




    for i in range(test_number):
        next_seq = sess.run(prey, feed_dict={x: [test_data[i]]})


        pre_list.append(next_seq)

# 计算均方根误差
pre_list_np=np.array(pre_list)
pre_list_np=np.reshape(pre_list_np,(test_number,))


rmse = 0.0
for i in range(test_number):
    rmse=rmse+(pre_list_np[i]-data[i+time_step][7])**2

rmse=np.sqrt((rmse/test_number))
rmse=np.around(rmse, decimals=6)




#绘制图像
plt.plot(range(np.shape(data)[0]), data[:,7])
plt.plot(range(np.shape(data)[0])[time_step:time_step+test_number], pre_list_np, color='r')
plt.text(1500, 8, 'rmse='+str(rmse), fontsize=20)

plt.show()