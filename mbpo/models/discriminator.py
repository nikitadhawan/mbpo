import tensorflow as tf 
import numpy as np 
import tensorflow.contrib.layers as layers

from functools import partial
from mbpo import tflib as tl

import tensorflow.contrib.slim as slim

conv = partial(slim.conv1d, activation_fn=None)
dconv = partial(slim.conv2d_transpose, activation_fn=None)
fc = partial(layers.fully_connected, activation_fn=None)
relu = tf.nn.relu
lrelu = tf.nn.leaky_relu
elu = tf.nn.elu
batch_norm = partial(slim.batch_norm, scale=True, updates_collections=None)

def discriminator(img):
	bn = partial(batch_norm, is_training=True)
	conv_bn_lrelu = partial(conv, normalizer_fn=bn, activation_fn=lrelu)
	fc_bn_lrelu = partial(fc, normalizer_fn=bn, activation_fn=lrelu)
	fc_bn_elu = partial(fc, normalizer_fn=bn, activation_fn=elu)

	# with tf.variable_scope('D', reuse=tf.AUTO_REUSE):
	# 	y = lrelu(conv(img, 1, 5, 2))
	# 	y = conv_bn_lrelu(y, 64, 5, 2)
	# 	y = conv_bn_lrelu(y, 1024, 5, 2)
	# 	# y = conv_bn_lrelu(y, 1, 5, 2)
	# 	# y = fc_bn_lrelu(y, 1024) # layers.fully_connected(y, 1024, activation_fn=tf.nn.leaky_relu, normalizer_fn=batch_norm)
	# 	logit = fc(y, 1) # layers.fully_connected(y, 1)
	# 	return logit

		# y = lrelu(conv(img, 1, 5, 2))
		# y = conv_bn_lrelu(y, 64, 5, 2)
		# y = fc_bn_elu(y, 2048)
		# y = fc_bn_elu(y, 1024)
		# y = fc_bn_elu(y, 1024)
		# logit = fc(y, 1)
		# return logit

	with tf.variable_scope('D', reuse=tf.AUTO_REUSE):
		y = layers.conv1d(img, 1, 5, 2)
		y = tf.nn.leaky_relu(y)
		y = layers.conv1d(y, 64, 5, 2, activation_fn=tf.nn.leaky_relu, normalizer_fn=layers.batch_norm)
		y = layers.conv1d(y, 256, 5, 2, activation_fn=tf.nn.leaky_relu, normalizer_fn=layers.batch_norm)
		y = layers.fully_connected(y, 1024, activation_fn=tf.nn.leaky_relu, normalizer_fn=layers.batch_norm)
		output = layers.fully_connected(y, 1)
		return output
