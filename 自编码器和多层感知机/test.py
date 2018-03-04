import tensorflow as tf
import numpy as np

y = np.array([[1, 2, 3],[2,3,4]])
# sum_result = tf.reduce_mean(tf.reduce_sum(y, reduction_indices=[1]))
sum_result = tf.reduce_mean(23.3)

sess = tf.InteractiveSession()
init = tf.initialize_all_variables()
sess.run(init)

print('sum_result=', sum_result.eval())
