# Merovingen Solver Library
# Author: Dmitry Prozorovsky.
# IOL Office Creative Union.
# MMXXIII

import tensorflow as tf

from keras.src.engine.base_layer import Layer
from keras.src.engine.input_spec import InputSpec
from tensorflow.python.keras.constraints import NonNeg
from tensorflow.python.keras.utils import tf_utils
# from tensorflow.python.keras.layers import Lambda
# from layers.phyblock import tf_const

sdense_num = 0
class SpectreDense(Layer):
	# TODO: Add fallbac to standard dense layer
	modes = [
		'fold', 	# Fold weights from 3 to one weight and proces...
		'unfold', 	# Unfold weights from 3 to 9 weights and process input
		'default'	# Default dense implementation with use of tensordot.
	] 
	def __init__(self,
		units, mode='unfold',
		name=None, activation=None,
		trainable=True,
		dtype = tf.float32,
		**args):
		global sdense_num

		if name is None: name = self.__class__.__name__
		super( self.__class__, self).__init__(name=f"{name}_{sdense_num}", trainable=trainable, dtype=dtype)

		self.units = int(units) if not isinstance(units, int) else units
		if self.units < 0:
			raise ValueError(
				"[SpectreDense] Received an invalid value for `units`, expected "
				f"a positive integer. Received: units={units}"
			)

		self.input_spec = InputSpec(min_ndim=2)

		self.units = int(units)
		self.mode = f"{mode}".lower()
		self.activation = activation
		self.trainable = trainable
		self._dtype = dtype

		sdense_num += 1
		

	def build(self, input_shape):
		self.w_init = tf.keras.initializers.GlorotUniform()
		self.b_init = tf.keras.initializers.Zeros()

		w_shape = None
		b_shape = None
		dtype = self.dtype

		self.one = tf.constant( 1.0, dtype=dtype)
		self.half = tf.constant( 0.5, dtype=dtype)
		self.two = tf.constant( 2.0 , dtype=dtype)
		self.inv_byte = tf.constant( 1.0/255.0, dtype=dtype )
		self.vec3_ratio = tf.constant( 0.38828840, dtype=dtype)# 1/255.0 * 99.0
		self.vec3_nrm_ratio = tf.constant(1.0/9999.99, dtype=dtype) # 1/9999.99 * 255.0
		self.inv_vec3_nrm_ratio = tf.constant(9999.99, dtype=dtype) # 1/9999.99 * 255.0
		self.byte_value = tf.constant(255.0 , dtype=dtype)
		self.h100 = tf.constant(100, dtype=dtype)
		self.inv_h100 = tf.constant(0.01, dtype=dtype)
		self.value_vec3 = tf.constant( 9999.99, dtype=dtype) # fractional part can be from 0.0 to 99999999.... 
		
		# TODO: Ranges should be checked
		self.vec3_out_nrm_r = tf.constant(self.h100*(1.0/(99.02344)), dtype=dtype)
		self.vec3_out_nrm_g = tf.constant(self.h100*(1.0/(99.9999)), dtype=dtype)
		self.vec3_out_nrm_b = tf.constant(self.inv_h100*(1.0/(99.9999)), dtype=dtype)

		# if self.mode == 'default':
		w_shape = (input_shape[-1], self.units)
		b_shape = (self.units,)

		constraint = None
		if self.mode == 'unfold':
			self.op = self.unfold # float_to_vec3
			constraint=NonNeg()
		elif self.mode == 'fold':
			self.op = self.fold # vec3_to_float
			constraint=NonNeg()
		else:
			self.op = tf.identity

		if type(self.activation) is str:
			self.act = tf.keras.layers.Activation(self.activation)
		elif self.activation is None:
			self.act = tf.identity
		else:
			self.act = self.activation

		self._w = tf.Variable(
			name = "w",
			initial_value=self.w_init(shape=w_shape, dtype=self._dtype),
			constraint=constraint,
			trainable=self.trainable, dtype=self._dtype)
		
		self._b = tf.Variable(
		 	name = "b",
		 	initial_value=self.b_init(shape=b_shape, dtype=self._dtype),
			constraint=constraint,
		 	trainable=self.trainable, dtype=self._dtype)
		
		self._s = tf.Variable(
			name = "s",
		 	initial_value=self.w_init(shape=(1,), dtype=self._dtype),
		 	trainable=self.trainable, dtype=self._dtype)
		
		print(f"[Spectre Dense] {self.units}]: in:{input_shape} | out: {self.compute_output_shape(input_shape)} | w_shape: {w_shape} | b_shape: {b_shape}")
		# print( self.unfold( [255.0] ) )
		# exit()

	def get_config(self):
		config = {
			'activation': self.activation,
			'trainable'	: self.trainable,
			'units'		: self.units,
			'dtype'		: self._dtype,
		}
		base_config = super(SpectreDense, self).get_config()
		return dict(list(base_config.items()) + list(config.items()))

	@tf.function # vec3_to_float # in: [0 ~ 255.0] out: [-1.0 ~ 1.0]
	def fold(self, x):
		x = tf.multiply( tf.add(x, 0.009) , self.vec3_ratio)
		r,g,b = tf.split(x, 3, axis=-1) # TODO: [WARNING] FIX 0,0,0 value! +1.0 or +0.1... fix
		b = tf.multiply( tf.math.round(b), self.h100 )
		g = tf.math.round(g)
		r = tf.multiply( r , self.inv_h100 )
		x = tf.concat( [ r, g, b] , axis=-1)
		x = tf.reduce_sum(x, axis=-1) # + one
		x = tf.multiply(x , self.vec3_nrm_ratio)
		return x

	@tf.function # float_to_vec3 # in: [-1.0 ~ 1.0] out: [0 ~ 255.0]
	def unfold(self, x):
		# x = tf.clip_by_value( x+3.494599e-09, 3.494599e-09, 1.0)
		x = tf.multiply( x, self.inv_vec3_nrm_ratio) #vec3_nrm_ratio
		x_b = tf.multiply(x, self.vec3_out_nrm_b)
		x_g = tf.multiply(tf.math.mod(x_b, self.one), self.vec3_out_nrm_g) 
		x_r = tf.multiply(tf.math.mod(x, self.one), self.vec3_out_nrm_r) 
		x = tf.concat( [x_r, x_g, x_b] , axis=-1) # concat instead of stack for dense layer
		# tf.print(x)
		return (x/255.0) * 5.0 # *2*(3.14159265)
	
	@tf_utils.shape_type_conversion
	def compute_output_shape(self, input_shape):
		out_units = self.units
		if self.mode == 'unfold':
			out_units = int(self.units*3)
		elif self.mode == 'fold':
			out_units = int(self.units/3)
		else:
			self.op = tf.identity
		return input_shape[:len(input_shape)-1] + (out_units,)

	@tf.function
	def normalize(self, x):
		return x # tf.tanh(tf.abs(x))
	
	@tf.function
	def call(self, inputs):
		w = self.op(self._w)
		b = self.op(self._b) # *self._s

		x = inputs
		x = self.act(x)
		x = tf.tensordot(x, w, axes=1)
		x = tf.nn.bias_add(x , b)*self._s
		
		return x
		
		# if self.quantize:
		# 	w = self._w
		# 	w_a = tf.add( tf.math.ceil(w),  		self._w_sc[0] )
		# 	w_b = tf.add( tf.math.mod( w, 1.0 ) , 	self._w_sc[1] )
		# 	w = tf.concat( [w_a, w_b*0.0], axis=-1 ) 

		# 	b = self._b
		# 	b_a = tf.add( tf.math.ceil(b),			self._b_sc[0] )
		# 	b_b = tf.add( tf.math.mod( b, 1.0 ) , 	self._b_sc[1] )
		# 	b = tf.concat( [b_a, b_b*0.0], axis=-1 ) 
		# else:
		# 	w = self._w
		# 	b = self._b

if __name__ == "__main__":
	data = tf.random.uniform( (4096*32,), -1.0*2.3*3.14159 , 1.0*2.3*3.14159 , dtype=tf.float32 )
	x_train = data/(2.3*3.14159) # * 1024
	y_train = tf.math.sin(data) # * 1024

	units = 9
	dns = SpectreDense
	
	model = tf.keras.Sequential([
		tf.keras.layers.InputLayer((1,) ),
		SpectreDense(units, activation=None),
		SpectreDense(units, activation=None),
		SpectreDense(units, activation=None),
		# SpectreDense( 1, activation=None, mode='default'),
		tf.keras.layers.Activation("tanh")
	])
	
	#model = tf.keras.Sequential([
	#	tf.keras.layers.InputLayer((1,) ),
	#	tf.keras.layers.Dense(units, activation=None),
	#	tf.keras.layers.Dense(units, activation=None),
	#	tf.keras.layers.Dense(units, activation=None),
	#	tf.keras.layers.Dense( 1, activation=None, mode='default'),
	#	tf.keras.layers.Activation("tanh")
	#])

	optimizer = tf.optimizers.Adam(0.01)
	model.compile(optimizer=optimizer, loss=['mse', 'mae'], metrics=['mse', 'mae'])
	model.summary()
	model.fit(x=x_train, y=y_train, batch_size=4096*32, epochs=1024)

	out = model(x_train)
	for layer in model.layers:
		print(layer.get_weights())
	
	print("OUT min: ", tf.reduce_min(out).numpy(), "max: ", tf.reduce_max(out).numpy() )
	print("X min: ", tf.reduce_min(x_train).numpy(), "max: ", tf.reduce_max(x_train).numpy() )
	print("Y min: ", tf.reduce_min(y_train).numpy(), "max: ", tf.reduce_max(y_train).numpy() )
	