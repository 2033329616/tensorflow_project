# encoding:utf-8

"""
功能:自编码器实现数据的特征提取
作者:<<Tensorflow实战>>第四章例程
版本:1.0
日期:02/25/2018
结果:
Epoch:0001 cost= 14194.550706818
Epoch:0002 cost= 9987.025388068
Epoch:0003 cost= 9153.967202841
Epoch:0004 cost= 7388.280001136
Epoch:0005 cost= 8090.349357386
Epoch:0006 cost= 9026.847461364
Epoch:0007 cost= 7839.241132386
Epoch:0008 cost= 8159.309690909
Epoch:0009 cost= 7257.022417614
Epoch:0010 cost= 7607.130351705
Epoch:0011 cost= 7570.813230114
Epoch:0012 cost= 7628.024593182
Epoch:0013 cost= 7557.681542614
Epoch:0014 cost= 7371.261881250
Epoch:0015 cost= 6821.391518750
Epoch:0016 cost= 7636.978473864
Epoch:0017 cost= 7481.062013068
Epoch:0018 cost= 7835.498840909
Epoch:0019 cost= 8134.941004545
Epoch:0020 cost= 7902.877036364
Total cost: 610323.9
"""

import numpy as np
import tensorflow as tf
import sklearn.preprocessing as prep
from tensorflow.examples.tutorials.mnist import input_data

# xavier参数初始化函数
def xavier_init(fan_in, fan_out, constant=1):
	"""
	功能: 初始化各层间权重,使权重满足:均值=0,方差=2/(Nin+Nout)
	参数: fan_in:输入的节点数, fan_out:输出的节点数, constant:权重的幅度值
	返回: (fan_in,fan_out)形状的向量,各个元素由均匀分布产生
	"""
	low = -1*constant*np.sqrt(6.0/(fan_in + fan_out))
	high = 1*constant*np.sqrt(6.0/(fan_in + fan_out))
	return tf.random_uniform((fan_in, fan_out), minval=low, maxval=high, dtype=tf.float32)

# 定义AGN去噪自编码器的类
class AdditiveGaussionNoiseAutoencoder(object):
	def __init__(self, n_input, n_hidden, transfer_function=tf.nn.softplus, optimizer=tf.train.AdamOptimizer(), scale=0.1):
		"""
		功能:初始化自编码器的类
		参数:n_input:输出的节点数,n_hidden:隐藏层的节点数,transfer_function:隐藏层的激活函数,默认为sotfplus
		    optimizer:优化器,默认为Adam,scale:高斯噪声系数,默认0.1
		返回:无返回值
		"""
		# 变量的初始化
		self.n_input = n_input
		self.n_hidden = n_hidden
		self.transfer = transfer_function
		self.scale = tf.placeholder(tf.float32)   #self.scale使用占位符,是为了在类初始化后可以改变噪声系数,将training_scale的值赋给scale
		self.training_scale = scale
		network_weights = self._initialize_weights()  #调用下划线的类方法(成员函数),所以使用self!!!!!!!!
		# network_weights = _initialize_weights() 
		self.weights = network_weights

		# 自编码器的网络结构定义
		self.x = tf.placeholder(tf.float32, [None, n_input])
		# 隐藏层=输入上叠加噪声后乘w1加b1
		hidden_output = tf.add(tf.matmul(self.x + self.scale*tf.random_normal((n_input,)), self.weights['w1']), self.weights['b1'])
		self.hidden = self.transfer(hidden_output)
		# 输出层=隐藏层*w2+b2
		reconstruction_output = tf.add(tf.matmul(hidden_output, self.weights['w2']), self.weights['b2'])
		self.reconstruction = reconstruction_output

		# 定义损失函数和优化器,平方误差:(输出-输入)**2的和
		self.cost = 0.5*tf.reduce_sum(tf.pow(tf.subtract(self.reconstruction, self.x), 2.0))
		self.optimizer = optimizer.minimize(self.cost)
		init = tf.initialize_all_variables()        # 定义初始化所有的变量参数的变量
		self.sess = tf.Session()                         # 定义对话
		self.sess.run(init)                              # 调用对话初始化变量

	def  _initialize_weights(self):
		"""
		功能:初始化各层的权重w和偏置b, w的形状为:输入节点数*输出节点数, b的形状为:1*输出节点数
		     没有激活函数的层,w和b取0,有激活函数的层间w用已定义的xavier_init来初始化
		     单下划线函数表示私有成员函数,
		"""
		all_weights = dict()
		all_weights['w1'] = tf.Variable(xavier_init(self.n_input, self.n_hidden))
		all_weights['b1'] = tf.Variable(tf.zeros([self.n_hidden], dtype=tf.float32))
		all_weights['w2'] = tf.Variable(tf.zeros([self.n_hidden, self.n_input], dtype=tf.float32))
		all_weights['b2'] = tf.Variable(tf.zeros([self.n_input], dtype=tf.float32))
		return all_weights

	def partial_fit(self, X):
		"""
		功能:传入数据训练编码器
		参数:X:输入的数据,向量
		"""
		# 返回cost便于后续的计算和观察,计算optimizer是为了优化loss来更新参数
		cost, opt = self.sess.run((self.cost, self.optimizer), feed_dict={self.x: X, self.scale: self.training_scale})
		return cost

	def cal_total_cost(self, X):
		"""
		功能:主要用于测试集来评估模型的性能
		参数:X:输入的数据向量
		"""
		cost = self.sess.run(self.cost, feed_dict={self.x: X, self.scale: self.training_scale})
		return cost

	def transform(self, X):
		"""
	 	功能:隐含层的功能是将原始数据抽象出高阶的特征,该函数输出隐含层的特征
	 	参数:X:输入的数据向量
		"""
		return self.sess.run(self.hidden, feed_dict={self.x: X, self.scale: self.training_scale})

	def generate(self, hidden=None):
		"""
		功能:将隐含层得到的高阶特征重构为输入数据,??????????????????????该函数怎么使用,这里的hidden参数和原来的self.hidden参数关系
		参数:hidden:隐含层的输出
		"""
		if hidden == None:
			hidden = np.random.normal(size= self.weights['b1'])
		return self.sess.run(self.reconstruction, feed_dict={self.hidden: hidden})

	def reconstruct(self, X):
		"""
		功能:抽象输入数据的高阶特征,并用其来重构输入数据,reconstruct = transform + generate,transform相当于
			计算图的子图
		参数:X:输入的数据向量
		"""
		return self.sess.run(self.reconstruction, feed_dict={self.x: X, self.scale: self.training_scale})

	def getWeights(self):
		"""
		功能:获取隐含层的权重w1
		"""
		return self.sess.run(self.weights['w1'])

	def getBiases(self):
		"""
		功能:获取隐含层的偏置b1
		"""
		return self.sess.run(self.weights['b1'])


def standard_scale(X_train, X_test):
	"""
	功能:将数据集进行标准化预处理,使其均值为0,标准差为1
	参数:X_train, X_test分别为训练集和测试集的图片向量
	返回:返回标准化的数据向量
	"""
	preprocessor = prep.StandardScaler().fit(X_train)
	X_train = preprocessor.transform(X_train)
	X_test  = preprocessor.transform(X_test)
	return X_train, X_test 

def get_random_block_from_data(data, batch_size):
	"""
	功能:随机抽取block数据用于训练
	参数:data:总的训练集,batch_size每次训练的样本数
	返回:随机选取batch_size大小的样本
	"""
	# 随机产生一个在(0,len(data)-batch_size)范围内的整数
	start_index = np.random.randint(0, len(data) - batch_size)
	# 产生一个batch_size大小的样本子集
	return data[start_index:(start_index + batch_size)]

# 1. 读入数据,并标准化
mnist = input_data.read_data_sets('./MNIST_data', one_hot=True)
X_train, X_test = standard_scale(mnist.train.images, mnist.test.images)  # 对训练集和测试集进行标准变换

# 2. 定义网络常用参数
n_samples = int(mnist.train.num_examples)  # 样本总数
training_epochs = 20                       # 总样本参与训练的轮数
batch_size = 128                           # 每次参与训练的样本数
display_step = 1                           # 每隔一轮显示一次损失函数

# 3. 创建一个AGN编码器

autoencoder = AdditiveGaussionNoiseAutoencoder(n_input= 784, n_hidden= 200, transfer_function=tf.nn.softplus, optimizer=tf.train.AdamOptimizer(learning_rate= 0.001), scale=0.1)
# 4. 开始训练并测试
for epoch in range(training_epochs):
	avg_cost = 0
	total_batch = int(n_samples / batch_size)
	for i in range(total_batch):
		batch_xs = get_random_block_from_data(X_train, batch_size)
		cost = autoencoder.partial_fit(batch_xs)
		avg_cost += cost / n_samples * batch_size

	if epoch % display_step == 0:
		print('Epoch:%04d'% (epoch + 1), 'cost=','{:0.9f}'.format(avg_cost))

print('Total cost: ' + str(autoencoder.cal_total_cost(X_test)))



