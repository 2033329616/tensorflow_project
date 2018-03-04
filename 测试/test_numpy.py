# import numpy as np
# a = np.zeros([2,3])
# print('a=', '\n', a)
# print(np.shape(a))

# b = np.ones([1,3])
# print('b=', '\n', b)
# print(np.shape(b))

# print(a + b)

# c = np.ones([1,3])
# print('c=', '\n', c)
# bool_result = np.equal(b,c)
# print(np.shape(bool_result))

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)  
print('download has finished!')