# Merovingen Solver Library
# Author: Dmitry Prozorovsky.
# IOL Office Creative Union.
# MMXXIII

import tensorflow as tf
from keras import layers
from keras import initializers
from tensorflow.python.keras.utils import tf_utils

ndt_num = 0
# Non Deterministic Transformation
class NDT(layers.Layer):

	def __init__(self,
		name=None, activation=None, #weight_var='weight', mod_weight_var='mod_weight', mod_bias_var='mod_bias',
		shape=(10),
		range=(-0.15, 0.15),
		trainable=True,
		**args):

		self.range = range
		self.activation = activation
		global ndt_num
		ndt_num += 1
		if name is None: name = self.__class__.__name__

		self.trainable  = trainable
		super( self.__class__, self).__init__(name=f"{name}_{ndt_num}", trainable=self.trainable)
		print(f"NDT initialized...")

	def build(self, input_shape):
		print(f"NDT build started...")
		x_shape = input_shape[0]
		
		self.logic_dense = layers.Dense( 32, activation=None, trainable=self.trainable, )
		# self.logic_act = IOLActivation( name=f"{self.name}_logic",frequency=1.0, negative_fix=True,trainable=self.trainable, )#self.activation['fn'](**self.activation['args'])

		self.min_dense = layers.Dense( input_shape[-1], activation=None, trainable=self.trainable, )
		self.max_dense = layers.Dense( input_shape[-1], activation=None, trainable=self.trainable, )

		# self.min_act = IOLActivation( name=f"{self.name}_min",frequency=1.0, negative_fix=False,trainable=self.trainable, )#self.activation['fn'](**self.activation['args'])
		# self.max_act = IOLActivation( name=f"{self.name}_max",frequency=1.0, negative_fix=False,trainable=self.trainable, )#self.activation['fn'](**self.activation['args'])

		self.out_shape = self.compute_output_shape(input_shape)
		print("NDT initialization finished.")

	def get_config(self):
		config = {
			'trainable': self.trainable,
			'activation':self.activation,
		}
		base_config = super(NDT, self).get_config()
		return dict(list(base_config.items()) + list(config.items()))

	@tf_utils.shape_type_conversion
	def compute_output_shape(self, input_shape):
		print(f"NDT output.shape: {input_shape}")
		return [input_shape, input_shape, input_shape]

	@tf.function
	def keep_range(self, inputs, range):
		x = tf.keras.activations.tanh(inputs)
		x = (x + 1.0)*0.5
		x = x*(range[1]-range[0])
		x = x + range[0]
		x = tf.clip_by_value(x, range[0], range[1])
		return x
		
	@tf.function
	def call(self, inputs):
		x = inputs
		logic = self.logic_dense(x)
		logic = self.logic_act(logic)

		minval = self.min_dense(logic)
		minval = self.min_act(minval)
		minval = self.keep_range( minval, self.range )
		
		maxval = self.max_dense(logic)
		maxval = self.max_act(maxval)
		maxval = self.keep_range( maxval, self.range )

		prn = tf.random.uniform(shape=tf.shape(x) , minval=minval, maxval=maxval, dtype=tf.float32)
		
		return [tf.add(x, prn), minval, maxval]
	
if __name__ == "__main__":
	minval = tf.convert_to_tensor( [[-255.0, 0],[0,0]] , dtype=tf.float32)
	maxval = tf.convert_to_tensor( [[255,0], [255,255]], dtype=tf.float32)
	while True:
		prn = tf.random.uniform(shape=(2,2), minval=minval, maxval=maxval, dtype=tf.float32)
		prn = tf.cast(prn+0.5, dtype=tf.int16)
		print(prn)