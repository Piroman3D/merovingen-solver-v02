# Merovingen Solver Library
# Author: Dmitry Prozorovsky.
# IOL Office Creative Union.
# MMXXIII

import tensorflow as tf
from keras import layers
from keras import initializers

from tensorflow.python.keras.utils import tf_utils

from layers.phyblock import phy, tf_const

iol_act_counter = 0
class IOLActivation(layers.Layer):
	def __init__(self,
		name=None, frequency=0.955, negative_fix=True, sparse=True, trainable=True,
		dtype = tf.float32,
		act=None, **args):
		global iol_act_counter

		#if name is None: name = self.__class__.__name__
		name = self.__class__.__name__
		super( self.__class__, self).__init__(name=f"{name}_{iol_act_counter:02d}", trainable=trainable)
		iol_act_counter += 1

		self.frequency = frequency 
		self.negative_fix = negative_fix
		self.trainable  = trainable
		self._dtype = dtype
		self.sparse = sparse
	
	def build(self, input_shape):
		dims = len(input_shape)
		freq_shape = input_shape[-1]
		if self.sparse:
			self.left = int(freq_shape/2)
			self.right = freq_shape - self.left
			freq_shape = self.left

		if dims == 4:
			var_shape = (1,1,1, freq_shape) # used in image activation functions.
			if self.sparse:
				r_shape = (1,1,1, self.right) # used in image activation functions.
		if dims == 3:
			var_shape = (1,1, freq_shape) #used in image activation functions.
			if self.sparse:
				r_shape = (1,1, self.right) # used in image activation functions.
		if dims == 2:
			var_shape = (1, freq_shape) #used for classification activation.
			if self.sparse:
				r_shape = (1, self.right) # used in image activation functions.
		if dims == 1:
			var_shape = (freq_shape,) #used for classification activation. # Coma is fix for tf 2.8 version...
			if self.sparse:
				r_shape = (self.right,) # used in image activation functions.

		print(f"Initialising IOLActivation [{self.name}][{self._dtype}] with: {var_shape}")
		if(len(var_shape) > 1):
			pass
		
		_frequency_init = initializers.constant([self.frequency])
		if self.sparse:
			_r_init = initializers.GlorotUniform()
			self._r = tf.Variable(
				name = "r",
				initial_value=_r_init(shape=r_shape, dtype=self._dtype),
				trainable=self.trainable)
		
		self._frequency = tf.Variable(
			name = "f",
			initial_value=_frequency_init(shape=var_shape, dtype=self._dtype),
			trainable=self.trainable)

		self._pi		= phy.pi  			# tf_const( "pi",  3.1415926535897932384626433, dtype=self._dtype)
		self._two		= phy.two 			# tf_const( "two", 2.0, dtype=self._dtype)
		self._one 		= phy.one 			# tf_const( "one", 1.0, dtype=self._dtype)
		self._two_pi 	= phy.two_pi		# tf_const( "two_pi",  tf.multiply( self._pi, self._two).numpy(), dtype=self._dtype)
		self._two_pi_inv= phy.two_pi_inv	# tf_const( "two_pi_inv",  tf.truediv( self._one, self._two_pi).numpy(), dtype=self._dtype)
		self.e_weight 	= phy.e_weight 		# tf_const( "e_weight", 	9.1093837015e-31, 				self._dtype) # kg # elector weight
		self.plank 		= phy.plank 		# tf_const( "plank", 		6.62607015e-34, 				self._dtype)
		self.plank_sc 	= phy.plank_sc 		# tf_const( "plank_sc", 	6.62607015e-3, 					self._dtype)
		self.plank_time = phy.plank_time	# tf_const( "plank_time",	5.391247e-44, 					self._dtype)
		self.h 			= phy.h				# tf_const( "h",	6.582119514e-16, 					self._dtype)

		# self._osc_const = tf_const( "osc_const",	86.9317383 * self._two_pi, 					self._dtype)
		self._osc_const = tf_const( "osc_const",	85.875317383 * self._two_pi, 					self._dtype) # Find out what is this constant actually...

	def get_config(self):
		config = {
			'dtype'	   : self._dtype,
			'frequency': initializers.constant([self.frequency]),
			'negativ_fix': self.negative_fix,
			'sparse': self.sparse,
			'trainable': self.trainable,
		}
		base_config = super(IOLActivation, self).get_config()
		return dict(list(base_config.items()) + list(config.items()))

	@tf_utils.shape_type_conversion
	def compute_output_shape(self, input_shape):
		return input_shape

	@tf.function
	def p2_shift(self, x, shift):
		return tf.subtract( tf.nn.elu(x), shift)
	
	@tf.function
	def p2(self, x):
		hz_ceil = tf.math.ceil(self._frequency)
		osc = tf.multiply( tf.math.subtract(hz_ceil, self._frequency), self._osc_const )

		shift_one = tf.subtract(self._one, tf.multiply(hz_ceil, phy.plank*phy.pi_sq2))
		shift_x = tf.math.add(x, shift_one)

		theta = tf.multiply(osc , shift_x)

		output = tf.add( shift_x, ( tf.subtract( shift_one, tf.math.cos( theta ) ) * (shift_one/osc) ) )
		if self.negative_fix:
			return self.p2_shift(output, shift_one)
		else:
			return output
	
	@tf.function
	def sparse_p2(self, x):
		a_sign, b_sign = tf.split( tf.sign(x), [self.left, self.right], axis=-1)
		a,b = tf.split( x, [self.left, self.right], axis=-1)
		
		r_b = tf.math.ceil(self._r)*phy.plank_time + self._one
		r_a = tf.multiply( tf.math.mod( self._r, self._one ) , phy.plank_mass )

		out_left  = self.p2(a + a_sign*tf.reduce_min(r_a))
		out_right = tf.nn.relu( (b*r_b) - r_a*b_sign)
		
		return tf.concat( [out_left, out_right] , axis=-1 )
	
	@tf.function( jit_compile=False )
	def call(self, inputs):

		x = inputs

		if self.sparse:
			x = self.sparse_p2(x) # * self.w
			return x
		else:
			x = self.p2(x) # * self.w
			return x

if __name__ == "__main__":
	pass