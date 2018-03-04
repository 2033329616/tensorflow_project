#encoding:utf-8

# download the dataset
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)  
print(mnist.train.next_batch(1))

import tensorflow as tf
print('dataset download completely.')
# print(mnist)
# 1. defination of the structure of the softmax regression
print('----------------1. defination of the structure of the softmax regression---------------------------------')
x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))         # the same as 1*10 dimension
y = tf.nn.softmax(tf.matmul(x, W) + b)  # the result of the  prediction 
y_ = tf.placeholder(tf.float32, [None, 10])  # the true result in the labels


# 2. defination of the loss function && use one method to optimize the loss
print('----------------2. defination of the loss function && use one method to optimize the loss----------------')
cross_entropy = -tf.reduce_sum(y*tf.log(y))  # defination of the loss function
# the method of the train step
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
# initialize all variables
init = tf.initialize_all_variables()
# create a session to start the model
sess = tf.Session()
sess.run(init)


# 3. start to train the model
print('----------------3. start to train the model--------------------------------------------------------------')
for i in range(6000):
	batch_xs, batch_ys = mnist.train.next_batch(100)
	sess.run(train_step, feed_dict={x:batch_xs, y_:batch_ys})


# 4. evaluate the accuracy of the model
print('----------------4. evaluate the accuracy of the model----------------------------------------------------')
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(accuracy)
# test the accuracy in the test datasets
print(sess.run(accuracy, feed_dict={x:mnist.test.images, y_:mnist.test.labels}))
