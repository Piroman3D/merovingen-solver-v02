# Patent pending.
# Author: Dmitry Prozorovsky.
# IOL Office Creative Union.
# MMXXIII

# The code is not for public domain, redistribution is prohibited. Remove the code if you read this NOTICE
# Based on Modulated Convolution algorythm from NVidia, Tensorflow
import tensorflow as tf
import keras
from keras import initializers
from keras import layers
from tensorflow.python.keras.utils import tf_utils

def assert_shape(x, shape):
	if(x.shape != shape):
		print(f"Shape does not match: {x.shape} - {shape}")
		exit()

mod_conv_num = 0
class ModConv2D(layers.Layer):
	def __init__(self,
		filters, kernel_size,
		up=False, down=False, demod=True, #fused_modconv=True,
		name=None, activation=None, #weight_var='weight', mod_weight_var='mod_weight', mod_bias_var='mod_bias',
		trainable=True,
		dtype = tf.float32,
		**args):
		global mod_conv_num

		if name is None: name = self.__class__.__name__
		super( self.__class__, self).__init__(name=f"{name}_{mod_conv_num}", trainable=trainable, dtype=dtype)
		
		self.demod = demod
		self.filters = filters
		self.up = up
		self.down = down
		self._dtype = dtype
		#self.fused_modconv = fused_modconv
		self.kernel_size = kernel_size
		self.activation = activation

		mod_conv_num += 1
		self.trainable  = trainable
		# print(f"ModConv2D initialized {self._dtype}...")

	def build(self, input_shape):
		# print(f"ModConv2D build started...")
		x_shape = input_shape[0]
		style_shape = input_shape[1]
		
		self.in_channels = x_shape[-1]
		print(f"[ModConv2D] {self._dtype} {self.__class__.__name__} {mod_conv_num} : x:{x_shape}, style:{style_shape}")
		
		weights_init = tf.keras.initializers.GlorotNormal()
		self.w = tf.Variable(
			name = "w",
			initial_value = weights_init(
				shape=(self.kernel_size,
				self.kernel_size,
				self.in_channels,
				self.filters),
			dtype=self._dtype),
			trainable=self.trainable)

		self.dense = layers.Dense(self.filters, input_shape=style_shape, activation=None, dtype=self._dtype)
		
		if self.activation is None:
			#print(1)
			#self.w_act = layers.Activation(None, dtype=self._dtype)
			self.w_act = tf.identity
		elif type(self.activation) == str:
			if self.activation.lower() in ['relu', 'swish', 'gelu', 'sigmoid', 'tanh', 'prelu', 'elu', 'selu', "softmax"]:
				#print(2)
				self.w_act = layers.Activation(self.activation)
			elif self.activation.lower() in ['snake']:
				#print(3)
				# self.w_act = tfa.layers.Snake( dtype=self._dtype )
				print("[ERROR] in activation layers!")
				exit()
			else:
				print("[ERROR] in activation layers!")
				exit()
				# print(4)
				# self.w_act = IOLActivation(negative_fix=False, frequency=0.1, dtype=self._dtype)
		else:
			#print(5)
			self.activation['args']['dtype'] = self._dtype
			self.w_act = self.activation['fn'](**self.activation['args'])

		self.out_shape = self.compute_output_shape(input_shape)
		print("ModConv  initialization finished.")

	def get_config(self):
		config = {
			#'frequency': initializers.constant([self.frequency]),
			'trainable': self.trainable,
			'demod': self.demod,
			#'fused_modconv':self.fused_modconv,
			'dtype':self._dtype,
			'filters':self.filters,
			'up':self.up,
			'down':self.down,
			'in_channels':self.in_channels,
			'activation':self.activation,
		}
		base_config = super(ModConv2D, self).get_config()
		return dict(list(base_config.items()) + list(config.items()))

	@tf_utils.shape_type_conversion
	def compute_output_shape(self, input_shape):
		x_shape = input_shape[0]
		style_shape = input_shape[1]
		self.out_shape = (x_shape[0], x_shape[1], x_shape[2], self.filters)
		if self.up:
			self.out_shape = (x_shape[0], x_shape[1]*2, x_shape[2]*2, self.filters)
		if self.down:
			self.out_shape = (x_shape[0], int(x_shape[1]/2), int(x_shape[2]/2), self.filters)
		
		print(f"ModConv output.shape: {self.out_shape}")
		#TODO: Can try to output modified style too...
		return self.out_shape # tf.shape( (x_out_shape, style_shape) )

	@tf.function
	def call(self, inputs):
		x = inputs[0]
		y = inputs[1]

		w = self.w
		s_dense = self.dense(y)
		s_dense = self.w_act(s_dense)
		s = tf.reduce_mean(s_dense, axis=0)
		w = tf.multiply(w, s)
		
		if self.up:
			x = tf.nn.conv2d_transpose(x, w, data_format='NHWC', strides=[1,2,2,1], padding='SAME')
		elif self.down:
			x = tf.nn.conv2d(x, w, data_format='NHWC', strides=[1,2,2,1], padding='SAME')
		else:
			x = tf.nn.conv2d(x, w, data_format='NHWC', strides=[1,1,1,1], padding='SAME')
		
		return [x, s_dense]