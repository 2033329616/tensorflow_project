# encoding:utf-8

"""
功能:将字节形式存储的数据读入到numpy数组中;将原始的mnist数据另存为jpg格式的图片
     和txt标签
版本:1.0
作者:David
日期:02/20/2018
"""

import os
import struct
import numpy as np
from scipy.misc import imsave
import matplotlib.pyplot as plt

def load_mnist(path, kind='train'):
	"""Load MNIST data from path
	参数: path:文件路径, kind:文件的类型,训练数据和测试数据
	返回:图片和标签的向量
	"""
	#数据的路径,使用字符串相加即可,下面使用两种方法
	images_path = os.path.join(path, '%s-images.idx3-ubyte'% kind)
	labels_path = path + '%s-labels.idx1-ubyte'% kind

	#读入标签,输出一个numpy的向量	
	with open(labels_path, 'rb') as lbpath:
		magic, num = struct.unpack('>ii', lbpath.read(8))  #使用大端存储,读取前8字节的数据,该数据已经出栈!!!
		# print(lbpath)
		labels = np.fromfile(lbpath, dtype=np.uint8)
	# print(magic, num)
	# print('labels:', labels)
	# print(type(labels), labels.shape)
	# print('--------------------------------')

	#读入图片,输出一个numpy的向量
	with open(images_path, 'rb') as imgpath:
		magic, num, rows, cols = struct.unpack('>iiii', imgpath.read(16))  #使用大端存储,读取前16字节的数据,该数据已经出栈!!!
		images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 784)
	# print(magic, num, rows, cols)
	# print('images:', images)
	# print(type(images), images.shape)
	
	# images向量的形状:样本数*图片的特征数(6000*784), labels向量形状: 样本数*标签类(6000*10),6000为训练样本数
	return images, labels

def show_img(images, labels):
	"""显示图片
	参数: images:数据的图片向量, labels:标签的向量
	返回值:无
	"""
	flg, ax = plt.subplots(nrows=2, ncols=5, sharex=True, sharey=True)  #绘制2*5个子图
	ax = ax.flatten()  #铺平ax,将原2*5的ax变为1*10的ax,便于后面的循环遍历
	for i in range(10):
		img = images[labels == i][0].reshape(28, 28)      #取出对应数字的图片向量,并reshape为28*28
		ax[i].imshow(img, cmap='Greys_r', interpolation='nearest')  #显示图片到对应的位置
	ax[0].set_xticks([])       #不显示坐标轴
	ax[0].set_yticks([])
	plt.tight_layout()         #图片分布更加松散
	plt.savefig('MNIST.png')   #保存图片
	# imsave('123.jpg', img)
	plt.show()                 #显示图片

	# print(images[labels ==1]) #取出标签是1的图片

def save_data(path, images, labels, kind='train'):
	"""保存图片和标签
	参数: path:保存图片的路径, images:数据的图片向量, labels:标签的向量, kind:数据集的类型,默认是训练集
	返回值:无
	"""
	numbers = len(images[0:])         #读取图片样本的数目,读取二维数组的行数
	img_dir = ''                      #图片路径
	img_name = ''                     #图片名称
	label_name = ''                   #标签名称
	content = ''                      #标签保存的内容
	# print(numbers)

	#保存图片的部分
	print('start to save the images.')
	for i in range(numbers):
		temp_img = images[i].reshape(28, 28)
		img_dir = path + '/images/%s/'% kind   #图片路径文件夹

		if os.path.exists(img_dir) == False:   #如果没有该路径
			os.mkdir(img_dir)                  #创建该文件夹
			# print('mkdir successfully!')
		img_name =  img_dir + str(labels[i]) + '_%05d.jpg'% i   #图片路径及图片名称
		print(img_name)
		imsave(img_name, temp_img)                              #图片名称为: 类别号_00233.jpg的形式
	print('Images save completely.')

	#保存标签部分
	print('start to save the labels.')
	label_name = path + '/labels/%s.txt'% kind
	# print(label_name)
	with open(label_name, 'w') as file:
		for i in range(numbers):
			content = str(labels[i]) + '_%05d.jpg'% i + ' ' + str(labels[i])
			print(content)
			file.writelines(content + '\n')
	print('Labels save completely.')

#存储train数据集
images, labels = load_mnist('./MNIST_data/')
save_data('./MNIST_data', images, labels)

# 存储test数据集
images, labels = load_mnist('./MNIST_data/', 't10k')
save_data('./MNIST_data', images, labels, 'test')

# show_img(images, labels)
