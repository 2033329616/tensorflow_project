#encoding:utf-8

"""调用自带的数据处理方法,读入MNIST的数据,reshape后显示"""

# download the dataset
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
batch_xs, batch_ys = mnist.train.next_batch(1)


# print(batch_xs.shape)
# print(type(batch_xs))
# print(batch_ys.shape)
# print(type(batch_ys))
# print(batch_xs)
# print(batch_ys)

# batch_xs = np.array([np.random.randint(255) for i in range(784)])  #初始化一个0~255的整型数组
# print(batch_xs.shape)
# print(batch_xs)
image = batch_xs.reshape(28, 28)

plt.imshow(image, cmap='Greys_r')   # Greys表示白底黑字,Greys_r是白字黑底,_r表示逆转
plt.axis('off')
plt.show()