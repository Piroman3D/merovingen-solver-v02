# Merovingen Solver Library
# Author: Dmitry Prozorovsky.
# IOL Office Creative Union.
# MMXXIII

# [WARNING] Physical constant does not match the original values from other sources.
# Constants are adjusted for usage with neural network.

import tensorflow as tf
from tensorflow.python.keras.layers import VersionAwareLayers
from tensorflow.python.keras import layers
from tensorflow.python.keras.utils import tf_utils

_phy_constants_ = {}
_tf_phy_constants_ = None

def tf_const( name, value, dtype):
	global _phy_constants_
	
	dtype_name = f"{dtype}".replace("dtype", "").replace("'", "").replace("<", "").replace(">", "").replace(" ", "").replace(":", "")
	key = f"{dtype_name}_{name}"
	if key not in _phy_constants_.keys():	
		_phy_constants_[key] = tf.constant(value, name=name, dtype=dtype )
		print(f"[CONST]: [{name}]: [{dtype_name}] {_phy_constants_[key]} ")
	return _phy_constants_[key]

class PhyBlockLayer(layers.Layer):
	
	def __init__(self, input_shape=None, size=3, normalize=False, dtype=tf.float32, trainable=True, **args):
		name = self.__class__.__name__
		super( self.__class__, self).__init__(name=f"{name}", dtype=dtype, trainable=trainable, **args)
		
		# Phy constants
		self._dtype = dtype
		self.size = size
		self.trainable = trainable
		self._shape = input_shape
		self.normalize = normalize
		self.build(input_shape)
		
	def build(self, input_shape):
		global _tf_phy_constants_
		super().build(input_shape)
		phymult = 1.0e3 #1e3
		# The constants are wrong and should not be used for any purpose!
		print(f"[CONST] Multiplier: {phymult}")
		self.one		= tf_const( "one", 			1.0, self._dtype )
		self.two		= tf_const( "two",			2.0*self.one, self._dtype )
		self.neg 		= tf_const( "neg",			-self.one, 							self._dtype) # Negative one
		
		self.pi 		= tf_const( "pi",			3.1415926535897932384626433, 	self._dtype)
		self.pi_sq2		= tf_const( "pi_sq2",		tf.pow(self.pi, 2.0), 			self._dtype)
		self.pi_inv 	= tf_const( "pi_inv",		self.one/self.pi, 				self._dtype)
		self.two_pi 	= tf_const( "two_pi",  		tf.multiply(self.pi, self.two), self._dtype)
		self.two_pi_inv = tf_const( "two_pi_inv",  	tf.truediv( self.one, self.two_pi), self._dtype)

		self.bolc 		= tf_const( "bolc",			1.380649e-23*phymult,		 	self._dtype) # Bolcman constant
		# self.bolc_sc 	= tf_const( "bolc_sc",		1.380649,			 			self._dtype) # Bolcman constant
		self.eu 		= tf_const( "eu", 			0.57721566490153286060651209, 	self._dtype) # Euler const
		self.e 			= tf_const( "e", 			1.602176634e-19*phymult,		self._dtype) # Kulons, elementar electric charge.
		self.e_weight 	= tf_const( "e_weight", 	9.1093837015e-31*phymult, 		self._dtype) # kg # elector weight
		self.p_weight 	= tf_const( "p_weight", 	0.2314933236e-18*phymult, 		self._dtype) # eV/c^2 # Just guess...
		self.plank 		= tf_const( "plank", 		6.62607015e-34*phymult, 		self._dtype)
		self.plank_sc	= tf_const( "plank_sc", 	6.62607015e-3, 					self._dtype) # Scaled plank constant
		self.h 			= tf_const( "h", 			6.582119514e-16*phymult, 		self._dtype) # eV * s
		self.fi			= tf_const( "fi0", 			self.h/(2.0*self.e),			self._dtype) # h/(2e)  # 2.067833848e-15
		self.G 			= tf_const( "G", 			6.67430e-11*phymult, 			self._dtype) # m^3 * s^-2*kg^-1 # Gravitational constant
		self.plank_time = tf_const( "plank_time",	5.391247e-44*phymult, 			self._dtype) # s ( tf.sqrt( (h/G) / tf.pow( c , 5.0 ) ), x_zeroed ) # seconds ~=5.391247(60)e-44 # Plank time constant
		self.plank_mass = tf_const( "plank_mass",	2.176434e-8*phymult, 			self._dtype) 
		self.plank_watt = tf_const( "plank_watt",	3.62831e51*1e-30, 			self._dtype)  # 1e-30 is adjustment...

		self.c 			= tf_const( "c", 			(299792458.0/(1.006677)), self._dtype) # m/s light speed. # Multiplier is to keep digits of self.pts....
		self.c_inv 		= tf_const( "c_inv", 		(self.one/self.c), self._dtype) # m/s light speed. # Multiplier is to keep digits of self.pts....
		# Electromagnetic constant
		self.e0			= tf_const( "e0",			self.one / (4.0*self.pi*tf.pow( self.c, 2.0)) * 1e-11 * phymult, self._dtype) # 
		# self.e0 		= tf_const( "e0",			8.85418781762039e-12*phymult, 	self._dtype) # [ WARNING: Modern value is different ] dielectric 
		
		# self.four 		= tf_const( "four", 		-9.178861623172e10, self._dtype)
		# Negative value sample
		
		# Postoyannaya tonkoy structuri
		self.pts		= tf_const( "pts", tf.pow(self.e, 2.0) / (2.0*self.e0*self.c*self.h) *phymult, self.dtype ) # Calculated: 7.346079010590691e-21 
		# self.pts		= tf_const( "pts", 7.2973525693e-3, 								   self.dtype ) # Value from Wikipedia
		
		self._shape = [_ for _ in self._shape]
		self._shape[-1] = self.size
		# self._shape = tf.shape(self._shape)

		# Can be moved to globals...
		self.constants = tf.stack( [
			self.pi,
			self.pi_inv,
			self.pi_sq2,
			self.two_pi,
			self.two_pi_inv,
			# self.bolc,
			self.eu,
			self.e_weight,
			self.pts,
			self.plank,
			self.plank_mass,
			self.plank_watt,
			# self.plank_sc,
			self.plank_time,
			self.c,
			# self.c_inv,
			self.fi,
			self.h,
			self.G,
			self.e0,
			# self.four,
			# self.neg,
		], axis=0, name='phy_constants_concat' )

		# Normalize constant # Looks like non normalized constants helps to fit data faster
		# if self.normalize:
		# 	const_min = tf.reduce_min(self.constants)
		# 	const_max = tf.reduce_max(self.constants)
		# 	self.constants = ((self.constants - const_min) / (const_max - const_min) - 0.5)*2.0

		if _tf_phy_constants_ is None:
			self.constants = _tf_phy_constants_ = tf.constant( tf.expand_dims( self.constants, axis=0), self._dtype, name='phy_constants' )
			# self.constants = _tf_phy_constants_ = tf.constant( self.constants.numpy(), self._dtype, name='phy_constants' )
		else:
			self.constants = _tf_phy_constants_
		print(f"Phy constants: { self.constants.shape}")
		
		self.dense = layers.Dense( self.size, activation=None, trainable=self.trainable)
		# Tanh works better for constants selection
		self.act = layers.Activation(activation='sigmoid', dtype=self._dtype )
		# self.call = layers.Lambda( lambda x : tf.add( self.act( self.dense( self.constants )), x), output_shape=self._shape, name=self.name )
		# self.add = layers.Add()

	@tf_utils.shape_type_conversion
	def compute_output_shape(self, input_shape):
		self._shape = [_ for _ in input_shape]
		self._shape[-1] = self.size
		self._shape = tf.shape(self._shape)
		return self._shape

	def get_config(self):
		config = {
			'dtype'		: self._dtype,
			'shape' 	: self.shape,
			'trainable'	: self.trainable,
			'normalize'	: self.normalize,
		}
		base_config = super(PhyBlockLayer, self).get_config()
		return dict(list(base_config.items()) + list(config.items()))

	# @tf.function(jit_compile=False)
	def call(self, inputs):
		# return self.out(inputs)
		# return tf.add( tf.nn.tanh( self.dense( self.constants )) , inputs)
		# return tf.add( self.act( self.dense( self.constants )), inputs)
		# return tf.nn.bias_add( inputs, self.act( self.dense( self.constants ) ) )
		# return self.add( inputs, self.act( self.dense( self.constants ) ))
		# return inputs + self.act( self.dense( self.constants ))
		return tf.add( inputs, self.act( self.dense( self.constants )) )

phy =  PhyBlockLayer( (1,1) , dtype=tf.float32 )
if __name__ == "__main__":

	layer = PhyBlockLayer( (1,1) , dtype=tf.float64 )
	exit()
	precession = tf.float64

	one = tf.constant( 1.0, precession)
	pi = tf.constant( 3.1415926535897932384626433, precession)

	inv_pi = tf.constant( 0.3183098861837907, tf.float64)
	pi_pow2 = tf.constant( tf.pow( pi, 2.0), tf.float64)
	print(f"pi^2 = {pi_pow2.numpy()}")
	pi_pow3 = tf.constant( tf.pow( pi, 3.0), tf.float64)
	print(f"pi^3 = {pi_pow3.numpy()}")
	pi_pow4 = tf.constant( tf.pow( pi, 4.0), tf.float64)
	print(f"pi^4 = {pi_pow4.numpy()}")

	const255 = tf.constant( 67.0, tf.float64)
	print(f"const255 = {const255.numpy()}")
	const999 = tf.constant( 999.0, tf.float64)
	print(f"const999 = {const999.numpy()}")

	div255x9 = tf.constant( tf.truediv( const255, const999), tf.float64)
	print(f"div255x9 = {div255x9.numpy()}")

	division = tf.truediv( one , pi )
	print(f"1/pi = {division.numpy()}")
	division = pi * inv_pi
	print(division.numpy())