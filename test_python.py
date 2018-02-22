
# coding:utf-8

import tensorflow as tf
hello = tf.constant('Hello, TensorFlow!')
sess = tf.Session()
print(sess.run(hello))

# check the version of the current python and tensorflow
import sys
print(sys.version)
print(sys.version_info)
print(sys.path)

print(tf.__version__)
print(tf.__path__)
