# Merovingen Solver Library
# Author: Dmitry Prozorovsky.
# IOL Office Creative Union.
# MMXXIII

import cerebro
import numpy as np
import tensorflow as tf

def emit_constant( constant, x_zeroed):
	constant = tf.constant( constant, dtype=x_zeroed.dtype)
	constant = x_zeroed + constant

	# constant = tf.split(constant, constant.shape[-1], axis=-1)[-1]
	# initializer = tf.keras.initializers.Constant(constant)
	# constant = initializer(shape=x_zeroed.shape)
	
	return constant

class ShallowModel(cerebro.IOLCerebro):

	def build_model(self, layers=[5, 32, 64, 16, 6], activation=None, name='ShallowModel'):
		input_shape = (layers[0])

		if activation is None:
			print(f'[WARNING] activation was not passed to the model fallingback to relu')
			activation = {}
			activation['fn'] = tf.keras.layers.Activation
			activation['args'] = {'activation':'relu'}
		
		self.model = tf.keras.Sequential( name=name )
		self.model.add( tf.keras.layers.InputLayer( input_shape ) )
		
		# self.model.add( keras.layers.BatchNormalization())
		# Skip first layer, as it is used as an input
		
		for n, layer in enumerate(layers[1:]):
			if n == len(layers[1:])-1: break # Skip last layer
			self.model.add( tf.keras.layers.Dense(layer, activation=None))
			
			if n == len(layers[1:])-2:
				print(f'Last layer removed negative_fix') 
				if 'negative_fix' in activation['args']:
					activation['args']['negative_fix'] = False
			self.model.add( activation['fn'](**activation['args']))
		
		self.model.add(tf.keras.layers.Dense(layers[-1], activation=None) )
		self.model.add(tf.keras.layers.Activation(activation='tanh'))

		return self.model

# from activations.iolactivation_v04 import IOLActivation
class ShallowModelFFT(cerebro.IOLCerebro):

	def build_model(self, layers=[5, 32, 64, 16, 6], activation=None, name='ShallowModel'):
		input_shape = (layers[0])

		if activation is None:
			print(f'[WARNING] activation was not passed to the model fallingback to relu')
			activation = {}
			activation['fn'] = tf.keras.layers.Activation
			activation['args'] = {'activation':'relu'}

		input_layer = tf.keras.layers.Input( input_shape )
		x = input_layer
		x_zeroed = x*0.0

		pi = emit_constant( 3.1415926535897932384626433, x_zeroed)
		e = emit_constant( 0.57721566490153286060651209, x_zeroed)
		electron_weight = emit_constant( 9.1093837015e-31, x_zeroed)
		plank = emit_constant( 6.62607015e-34, x_zeroed)
		c = emit_constant( 299792458.0, x_zeroed )

		constants = tf.concat( [pi, e, electron_weight, plank, c], axis=-1 )
		constants = tf.keras.layers.Dense( x.shape[-1], activation='tanh',trainable=True)(constants)
		constants = tf.multiply( constants , tf.constant( 3.1415926535897932384626433, tf.float32 ) )

		x_complex = tf.cast( input_layer, tf.complex64)
		x = tf.signal.fft(x_complex)
		x = tf.concat( [
			tf.cast( input_layer, tf.float32 ),
			tf.cast( tf.math.real(x), tf.float32),
			tf.cast( tf.math.imag(x), tf.float32),
			tf.cast( constants, tf.float32 ),
			], axis=-1 )
		
		skip = tf.reshape( x, (-1, 1, x.shape[-1] ) )
		skip = tf.keras.layers.Conv1D( 32, kernel_size=(3), strides=1, padding='same', activation=None)(skip)
		skip = tf.reshape(skip, (-1, skip.shape[-1] ))
		skip = activation['fn'](**activation['args'])(skip)
		
		skip = tf.keras.layers.Dense(layers[-1], activation=None)(skip) * tf.constant( 0.57721566490153286060651209e-3, tf.float32 )
		skip = activation['fn'](**activation['args'])(skip)
		
		# Inverse FFT for part of the values
		x = tf.keras.layers.Dense( x.shape[-1]*2, activation=None)(x)
		x = activation['fn'](**activation['args'])(x)
		x = tf.split(x, 4, axis=-1 )

		x_ifft = tf.signal.ifft( tf.complex( real=x[1], imag=x[2]))
		x = tf.concat( [
			x[0],
			x[1],
			tf.cast( tf.math.real(x_ifft), tf.float32),
			tf.cast( tf.math.imag(x_ifft), tf.float32),
			], axis=-1 )

		for n, layer in enumerate(layers[1:]):
			if n == len(layers[1:])-1: break # Skip last layer
			x = tf.reshape(x, (-1, 1, x.shape[-1] ) )
			x = tf.keras.layers.Conv1D( layer, kernel_size=(3), strides=1, padding='same', activation=None)(x)
			x = tf.reshape( x, (-1, x.shape[-1] ))
			x = tf.keras.layers.Dense(layer, activation=None)(x)
			
			# 1. summ of conv and dense gives worser results.
			x = activation['fn'](**activation['args'])(x)
		
		x = tf.keras.layers.Dense(layers[-1], activation=None)(x)
		x = tf.keras.layers.Activation(activation='tanh')(x)
		x = tf.multiply( x , tf.constant( 3.1415926535897932384626433, tf.float32 ) ) + skip

		self.model = tf.keras.Model(name=self.name, inputs=input_layer, outputs=x )
		self.model.trainable = self.trainable

		return self.model