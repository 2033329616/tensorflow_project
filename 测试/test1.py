import tensorflow as tf
import numpy as np

# using numpy to generate the phony data
x_data = np.float32(np.random.rand(2, 100))  # 2*100 dimension matrix
y_data = np.dot([0.100, 0.200], x_data) + 0.300

# consturct a linear model
b = tf.Variable(tf.zeros([1]))
w = tf.Variable(tf.random_uniform([1, 2], -1.0, 1.0))
y = tf.matmul(w, x_data) + b

# minimize variance
loss = tf.reduce_mean(tf.square(y - y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

# initialize the variable
init = tf.initialize_all_variables()

# start the graphs
sess = tf.Session()
sess.run(init)

# fitting the plane surface
for step in range(0, 201):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(w), sess.run(b))
