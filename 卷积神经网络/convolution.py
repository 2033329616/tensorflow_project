# encoding:utf-8

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

mnist = input_data.read_data_sets('./MNIST_data', one_hot=True)

# 初始化权重
def weight_variable(shape):
	"""截断正态分布,标准差为0.1"""
	initial = tf.truncated_normal(shape, stddev=0.1)
	print(initial)
	return tf.Variable(initial)

# 偏置初始化
def bias_variable(shape):
	"""初始化一个0.1常量组成的向量"""
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial)

# 卷积???
def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
# 池化???
def max_pool_2x2(x):
	return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

x = tf.placeholder('float')
y_ = tf.placeholder('float')

"""第一层卷积,x是4维的:第2、第3维对应图片的宽、高,最后一维代表图片的颜色通道数,灰度图的通道数为1,rgb彩色图则为3"""
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
x_image = tf.reshape(x, [-1, 28, 28, 1])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# 第二层卷积???
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# 全连接层
W_fc1 = weight_variable([7*7*64, 1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# Dropout
keep_prob = tf.placeholder('float')
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)


# 输出层
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)


# 训练和评估
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y_conv),reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()


for i in range(2000):
	batch = mnist.train.next_batch(50)
	if i%100 ==0:
		train_accuracy = accuracy.eval(feed_dict={x:batch[0],y_:batch[1], keep_prob:1.0})
		print('step%i, training accuracy %g' %(i, train_accuracy))                  #%g根据实际情况输出
	train_step.run(feed_dict={x:batch[0], y_:batch[1], keep_prob:0.5})
print('test accuracy %g'% accuracy.eval(feed_dict={x:mnist.test.next_batch(200)[0], y_:mnist.test.next_batch(200)[1], keep_prob:1.0}))






# # 测试函数的输出
# a = weight_variable([2, 3])
# b = bias_variable([3, 2])

# sess = tf.InteractiveSession()
# sess.run(tf.initialize_all_variables())
# print(sess.run(a))
# print(b.eval())
