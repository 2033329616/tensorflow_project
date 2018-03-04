#encoding:utf-8

"""
功能:多层感知机(multi-layer perceptron)识别mnist数据集
版本:v1.0
作者:<<Tensorflow实战>>第四章多层感知机示例
日期:02/27/2018
结果:
Epoch:0001 cost= 0.271617672
Epoch:0002 cost= 0.104115218
Epoch:0003 cost= 0.069747007
Epoch:0004 cost= 0.051422647
Epoch:0005 cost= 0.038202624
accuracy: 0.9798
经验:在loss为nan时,可以降低学习率或使用tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y), reduction_indices=[1]))来求损失函数
"""
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

# 导入数据并创建一个session
mnist = input_data.read_data_sets('./MNIST_data', one_hot=True)
sess = tf.InteractiveSession()

# 定义节点,权重和偏置
in_uints = 784        #输入节点数
h1_uints = 300        #隐含层节点数
W1 = tf.Variable(tf.truncated_normal([in_uints, h1_uints], stddev= 0.1))  # 初始化截断正态分布,标准差为0.1
b1 = tf.Variable(tf.zeros([h1_uints]))
W2 = tf.Variable(tf.zeros([h1_uints, 10]))   #mnist数据集共10类,所以W2的形状为(h1_uints,10)
b2 = tf.Variable(tf.zeros([10]))

# 输入数据和dropout的比率
x = tf.placeholder(tf.float32, [None, in_uints])
keep_prob = tf.placeholder(tf.float32)

# 定义模型结构, 输入层-隐层-输出层
hidden1 = tf.nn.relu(tf.matmul(x, W1) + b1)               # relu激活函数
hidden1_drop = tf.nn.dropout(hidden1, keep_prob)          # dropout随机使部分节点置零,克服过拟合
y  = tf.nn.softmax(tf.matmul(hidden1_drop, W2) + b2)      # softmax多分类
y_ = tf.placeholder(tf.float32, [None, 10])

# 定义损失函数和优化器
# cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y), reduction_indices=[1])) # 在学习率为0.3时,加reduce_mean后cost从nan变为正常!!!!!!!!!!!!1
cross_entropy = -tf.reduce_sum(y_*tf.log(y))                                          # 学习率为0.1,cost变为正常
optimizer = tf.train.AdagradOptimizer(0.01).minimize(cross_entropy)

# 初始化所有的参数
init = tf.initialize_all_variables()
sess.run(init)

# 通过循环训练数据
n_samples = int(mnist.train.num_examples)    #总样本数
batch_size =  128                           #batch_size数目
epochs = 5                                  #epochs数目
display_step = 1                            #迭代多少次显示loss

for epoch in range(epochs):
	avg_cost = 0
	total_batch = int(n_samples / batch_size)
	for i in range(total_batch):
		batch_xs, batch_ys = mnist.train.next_batch(batch_size)
		# optimizer.run({x:batch_xs, y_:batch_ys, keep_prob:0.75})
		# loss = sess.run(cross_entropy, feed_dict={x:batch_xs, y_:batch_ys, keep_prob:0.75})
		loss, opt = sess.run((cross_entropy, optimizer),feed_dict={x:batch_xs, y_:batch_ys, keep_prob:1})
		avg_cost += loss / n_samples * batch_size
		# print(loss)
	if epoch % display_step ==0:
		print('Epoch:%04d'% (epoch + 1), 'cost=', '{:.9f}'.format(avg_cost))

# 模型的评估
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print('accuracy:', accuracy.eval({x:mnist.test.images, y_:mnist.test.labels, keep_prob:1.0}))