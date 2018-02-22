# encoding:utf-8

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

# 读入mnist数据集
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# 启动交互式对话,不用每次都sess.run()
sess = tf.InteractiveSession()

# 输入图像和目标输出类别来创建节点
x  = tf.placeholder('float', shape=[None, 784])
y_ = tf.placeholder('float', shape=[None, 10]) 

# 定义权重和偏置  x*W + b
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# 在session中初始化所有变量
sess.run(tf.initialize_all_variables())

# 模型定义和损失函数
y = tf.nn.softmax(tf.matmul(x, W) + b)
# minibatch里的每张图片的交叉熵值都加起来
cross_entropy = -tf.reduce_sum(y_*tf.log(y)) 

# 用数据来训练模型
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
for i in range(1000):
	batch = mnist.train.next_batch(50)
	train_step.run(feed_dict={x:batch[0], y_:batch[1]})

# 评估模型
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
print(accuracy.eval(feed_dict={x:mnist.test.images, y_:mnist.test.labels}))