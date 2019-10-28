import tensorflow as tf 
import numpy as np 
import tf.contrib.layers as layers

def discriminator(obs):
	with tf.variable_scope('D', reuse=tf.AUTO_REUSE):
		y = layers.conv2d(obs, 1, 5, stride=2)
		y = tf.nn.leaky_relu(y)
		y = layers.conv2d(y, 64, 5, stride=2, activation_fn=tf.nn.leaky_relu, normalizer_fn=layers.batch_norm)
		y = layers.conv2d(y, 256, 5, stride=2, activation_fn=tf.nn.leaky_relu, normalizer_fn=layers.batch_norm)
		y = layers.fully_connected(y, 1024, activation_fn=tf.nn.leaky_relu, normalizer_fn=layers.batch_norm)
		output = layers.fully_connected(y, 1)
		return output