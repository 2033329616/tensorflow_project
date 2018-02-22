# encoding:utf-8

"""
功能:测试struct.unpack是否可以将数据出栈
版本:1.0
作者:David
日期:02/20/2018

运行结果:

labels: [0 0 8 ..., 4 5 6]
<class 'numpy.ndarray'> (10008,)
[7 2 1 0 4 1 4 9]
--------------------------------
labels: [7 2 1 ..., 4 5 6]
<class 'numpy.ndarray'> (10000,) 使用struct.unpack('>ii', lbpath.read(8))将8个字节的数据出栈
[7 2 1 0 4 1 4 9]
--------------------------------
"""

import os
import struct
import numpy as np
from scipy.misc import imsave
import matplotlib.pyplot as plt

def load_mnist(path, kind='train'):
	"""Load MNIST data from path
	参数: path:文件路径, kind:文件的类型,训练数据和测试数据
	"""
	#数据的路径,使用字符串相加即可,下面使用两种方法
	images_path = os.path.join(path, '%s-images.idx3-ubyte'% kind)
	labels_path = path + '%s-labels.idx1-ubyte'% kind

	#读入标签,输出一个numpy的向量	
	with open(labels_path, 'rb') as lbpath:
		# magic, num = struct.unpack('>ii', lbpath.read(8))  #使用大端存储,第8字节处开始读入数据,???是否已经该数据已经出栈了
		# print(lbpath)
		labels = np.fromfile(lbpath, dtype=np.uint8)
	# print(magic, num)
	print('labels:', labels)
	print(type(labels), labels.shape)
	print(labels[8:16])
	print('--------------------------------')

	with open(labels_path, 'rb') as lbpath:
		magic, num = struct.unpack('>ii', lbpath.read(8))  #使用大端存储,第8字节处开始读入数据,???是否已经该数据已经出栈了
		# print(lbpath)
		labels = np.fromfile(lbpath, dtype=np.uint8)
	# print(magic, num)
	print('labels:', labels)
	print(type(labels), labels.shape)
	print(labels[0:8])
	print('--------------------------------')
	

load_mnist('./MNIST_data/',  't10k')

