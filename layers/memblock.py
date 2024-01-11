# Merovingen Solver Library
# Author: Dmitry Prozorovsky.
# IOL Office Creative Union.
# MMXXIII

import tensorflow as tf
from keras.layers import Lambda

class MemBlockLayer(tf.keras.layers.Layer):
	def __init__(
			self,
			shape,
			units,
			activation = None,
			trainable = True,
			dtype = tf.float32
			):
		super().__init__()
		mem_init = tf.keras.initializers.GlorotUniform()
		# mem_init = tf.keras.initializers.GlorotNormal()
		# mem_init = tf.keras.initializers.Ones()
		self.shape=shape
		self.units = units
		self._dtype = dtype
		self.trainable = trainable

		self.memblock = tf.Variable(
			name = "memblock",
			initial_value=mem_init(shape=shape, dtype=self._dtype),
			trainable=True)
		
		self.assign = Lambda( lambda x : x, output_shape=self.memblock.shape )
		print(f"Memblock shape: {self.memblock.shape}")

		self.dense = tf.keras.layers.Dense( self.units, activation=None, trainable=self.trainable, dtype=self._dtype)
		self.pool = tf.keras.layers.MaxPooling1D(pool_size=1, strides=1, dtype=self._dtype)
		self.act = activation

	# TODO: Need to add config to the layer
	@tf.function
	def call(self, x, assign):
		if assign:
			print("MEMBLOCK ASSIGNED")
			return self.assign(x)

		print(f"MEMBLOCK SHAPE: {self.memblock.shape}")
		
		out = self.memblock
		out = self.dense(out)
		out = tf.expand_dims(out, axis=-2)
		out = self.pool(out)
		out = tf.squeeze(out, axis=-2)
		out = self.act(out)
		return out
	
	# @tf.function
	# def get(self):
	# 	return self.memblock

	# @property
	# def mem_shape(self):
	#  	return self.memblock.shape

	# @tf.function
	#def assign(self, x):
	#	#if type(x) is tf.Tensor and assign:
	#	#output = self.memblock
	#	print(f"Attemp to assign mem block: {x.shape}")
	#	if type(x) is tf.Tensor:
	#		#mem_in_min = tf.reduce_min(x, axis=0, keepdims=True)
	#		#mem_in_max = tf.reduce_max(x, axis=0, keepdims=True)
	#		#mem_in = tf.concat( [mem_in_min, mem_in_max], axis=-1 )
	#		#mem_in = self.mem_in_dense( x.shape[-1], activation=None)(mem_in)
	#		##mem_in = tf.expand_dims(mem_in, axis=-2)
	#		##mem_in = keras.layers.MaxPooling1D(pool_size=1, strides=1, dtype=precession)(mem_in)
	#		##mem_in = tf.squeeze(mem_in, axis=-2)
	#		#mem_in = tf.keras.layers.Activation(activation='softmax')(mem_in
	#		output = self.memblock.assign( x, read_value=True)
	#		print(f"MEMBLOCK ASSIGNED:{x.shape}")
	#		return output
	#	else:
	#		print(f"MEMBLOCK ASSIGN SKIPPED")
	#	return self.memblock