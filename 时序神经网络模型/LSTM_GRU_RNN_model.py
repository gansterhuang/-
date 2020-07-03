import tensorflow as tf
import numpy as np
import time
from datetime import timedelta


# 记录训练花费的时间
def get_time_dif(start_time):
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

# 这里和卷积神经网络那不同，RNN中的输入维度是（batch-size，28，28），而不是（batch-size，784）
X_train = X_train.astype(np.float32) / 255.0
X_test = X_test.astype(np.float32) / 255.0
y_train = y_train.astype(np.int32)
y_test = y_test.astype(np.int32)
X_valid, X_train = X_train[:5000], X_train[5000:]
y_valid, y_train = y_train[:5000], y_train[5000:]


def shuffle_batch(X, y, batch_size):
    rnd_idx = np.random.permutation(len(X))
    n_batches = len(X) // batch_size
    for batch_idx in np.array_split(rnd_idx, n_batches):
        X_batch, y_batch = X[batch_idx], y[batch_idx]
        yield X_batch, y_batch


n_steps = 28
n_inputs = 28
n_neurons = 100
n_outputs = 10
n_layers = 3

learning_rate = 0.001

X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.int32, [None])


# 选择记忆细胞
def cell_selected(cell):
    if cell == "RNN":

        # 指定激活函数为ReLU函数，然后构造三个RNN细胞状态
        # 构建堆叠的RNN模型
        # 每个时刻都有一个输出和一个隐状态（或多个隐状态），我们只取最后一个输出和隐状态
        # 但是TensofFlow中不知道为啥取了最后时刻的三个隐状态，用于计算最终输出。
        rnn_cells = [tf.nn.rnn_cell.BasicRNNCell(num_units=n_neurons, activation=tf.nn.relu)
                     for layer in range(n_layers)]

        multi_layer_cell = tf.nn.rnn_cell.MultiRNNCell(rnn_cells)

        outputs, states = tf.nn.dynamic_rnn(multi_layer_cell, X, dtype=tf.float32)
        return tf.concat(axis=1, values=states)

    elif cell == "LSTM":

        # 构造三个LSTM记忆细胞,不用管激活函数
        # states[-1]中包含了长期状态和短期状态，这里取最后一个循环层的短期状态
        gru_cells = [tf.nn.rnn_cell.LSTMCell(num_units=n_neurons)
                     for layer in range(n_layers)]
        multi_cell = tf.nn.rnn_cell.MultiRNNCell(gru_cells)
        outputs, states = tf.nn.dynamic_rnn(multi_cell, X, dtype=tf.float32)
        return states[-1][1]

    elif cell == "GRU":
        # GRU和LSTM大致相同，但是states[-1]中只包含了短期状态。
        gru_cells = [tf.nn.rnn_cell.GRUCell(num_units=n_neurons)
                     for layer in range(n_layers)]
        multi_cell = tf.nn.rnn_cell.MultiRNNCell(gru_cells)
        outputs, states = tf.nn.dynamic_rnn(multi_cell, X, dtype=tf.float32)
        return states[-1]


def build_and_train():
    # 调用上面定义的选择记忆细胞的函数，定义损失函数
    logits = tf.layers.dense(cell_selected(cell), n_outputs, name="softmax")
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
    loss = tf.reduce_mean(xentropy, name="loss")

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    training_op = optimizer.minimize(loss)

    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    n_epochs = 50
    batch_size = 100

    with tf.Session() as sess:
        init.run()
        start_time = time.time()

        # 记录总迭代步数，一个batch算一步
        # 记录最好的验证精度
        # 记录上一次验证结果提升时是第几步。
        # 如果迭代2000步后结果还没有提升就中止训练。
        total_batch = 0
        best_acc_val = 0.0
        last_improved = 0
        require_improvement = 2000

        flag = False
        for epoch in range(n_epochs):
            for X_batch, y_batch in shuffle_batch(X_train, y_train, batch_size):
                sess.run(training_op, feed_dict={X: X_batch, y: y_batch})

                # 每次迭代10步就验证一次
                # # 如果验证精度提升了，就替换为最好的结果，并保存模型
                if total_batch % 10 == 0:
                    acc_batch = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
                    acc_val = accuracy.eval(feed_dict={X: X_valid, y: y_valid})
                    if acc_val > best_acc_val:
                        best_acc_val = acc_val
                        last_improved = total_batch
                        save_path = saver.save(sess, "./my_model_Cell_Selected.ckpt")
                        improved_str = 'improved!'
                    else:
                        improved_str = ''
                    time_dif = get_time_dif(start_time)
                    msg = 'Epoch:{0:>4}, Iter: {1:>6}, Acc_Train: {2:>7.2%}, Acc_Val: {3:>7.2%}, Time: {4} {5}'
                    print(msg.format(epoch, total_batch, acc_batch, acc_val, time_dif, improved_str))
                # 记录总迭代步数
                total_batch += 1

                # 如果2000步以后还没提升，就中止训练。
                if total_batch - last_improved > require_improvement:
                    print("Early stopping in  ", total_batch, " step! And the best validation accuracy is ",
                          best_acc_val, '.')
                    flag = True
                    break
            if flag:
                break

    with tf.Session() as sess:
        saver.restore(sess, "./my_model_Cell_Selected.ckpt")
        acc_test = accuracy.eval(feed_dict={X: X_test, y: y_test})
        print("\nTest_accuracy:{0:>7.2%}".format(acc_test))


if __name__ == "__main__":
    cell = "LSTM"  # RNN/LSTM/GRU,在这里选择记忆细胞
    build_and_train()