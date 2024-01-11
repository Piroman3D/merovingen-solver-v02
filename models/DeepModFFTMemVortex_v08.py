# Merovingen Solver Library
# Author: Dmitry Prozorovsky.
# IOL Office Creative Union.
# MMXXIII

import cerebro
import numpy as np
import tensorflow as tf

from tensorflow import keras
from keras import initializers

def merge_const( constant, x_zeroed):
	constant = tf.add( x_zeroed, constant)
	return constant

from layers.modconv import ModConv2D
from layers.memblock import MemBlockLayer
from layers.phyblock import PhyBlockLayer, tf_const
# from layers.spectredense import SpectreDense

class DeepModFFTMemVortex(cerebro.IOLCerebro):

	def build_model(self, name='ShallowModel',
		
		input_shape  = (256, 256),
		output_shape = (256, 256),
		layers=[3, 32, 64, -1, 64, 16, 6],
		
		activation = None, 
		
		# vortex_range = (-1.0 , 1.0),
		# vortex_shape = (1,),
		# output_range = (-1.0, 1.0),

		phy_size = 3,
		mem_block = True,
		mem_multiplier = 2,

		modulated = True,

		dtype = tf.float64
		):
		
		input_shape = (layers[0])
		max_shape = max(layers)

		if activation is None:
			print(f'[WARNING] activation was not passed to the model falling back to relu')
			activation = {}
			activation['fn'] = tf.keras.layers.Activation
			activation['args'] = {'activation':'relu'}

		LayerDense = tf.keras.layers.Dense

		activation['args']['dtype'] = self.dtype

		input_layer = keras.layers.Input( input_shape, dtype=self.dtype )
		x = input_layer

		# TODO: Spectre should be somewhere here.
		zero 		= tf_const( "zero",			0.0, 							self.dtype)
		one 		= tf_const( "one",			1.0, 							self.dtype)

		x_one, _ 	= tf.split(x, (1, x.shape[-1]-1), axis=-1) # One batch slice
		x_zeroed 	= tf.multiply(x_one, zero, name="x_zeroed") # Zeroed batch slice
		
		# Phy Block start
		phy = PhyBlockLayer( size=phy_size, normalize=False, input_shape=x.shape, dtype=self.dtype )
		constants = phy( x_zeroed )
		# Phy Block end

		# Skip block
		skip = tf.concat( [
			tf.math.reduce_min(x, axis=0, keepdims=True),
			tf.math.reduce_variance(x, axis=0, keepdims=True),
			tf.math.reduce_max(x, axis=0, keepdims=True),
			], axis=-1 )
		skip = LayerDense( max_shape, activation=None, dtype=self.dtype)(skip)
		skip = activation['fn'](**activation['args'])(skip)

		# Memory Block
		mem_sq_layers = 0
		if mem_block:
			mem_sq_layers = (len(layers))
			
			memblock = MemBlockLayer(
				shape = (1, (x.shape[-1]*len(layers)-1)*mem_multiplier),
				units = 6 + mem_sq_layers,
				activation = activation['fn'](**activation['args']),
				dtype=self.dtype
				)
			
			mem_sq = memblock(x_zeroed, assign=False)
			mem_sq = mem_sq[0]

			#if modulated:
			mem_sq, mem_sq_layers = tf.split(mem_sq, [ 6, mem_sq_layers] )
			print(f"Memblock shape: {mem_sq.shape} | Memlayers: {mem_sq_layers.shape}")
				
		complex_dtype = tf.complex64
		if self.dtype == tf.float64: complex_dtype = tf.complex128
		
		# Start of the model here.
		x_input = input_layer
		
		# FFT -> ACT(DENSE) -> IFFT
		x_complex = tf.cast( x_input, complex_dtype)
		x = tf.signal.fft(x_complex)
		x = tf.concat( [
			x_input,
			tf.math.real(x) * phy.plank_sc,
			tf.math.imag(x),
			constants,
			], axis=-1, name="fuse" )

		# Inverse FFT for part of the values
		# Prepare layers for convolution and ffts
		x = LayerDense( (x.shape[-1]*4), activation=None, dtype=self.dtype)(x) 
		x = activation['fn'](**activation['args'])(x)
		
		# Mem Fuse and FFT
		x = tf.split(x, 4, axis=-1 )
		x_ifft = tf.signal.ifft( tf.complex( real=x[2], imag=x[3]))
		if mem_block:
			x = tf.concat( [
				x[0] + (mem_sq[0] * phy.e_weight),
				x[1] - (mem_sq[1] * phy.c_inv),
				tf.math.real(x_ifft) * (mem_sq[2] * phy.e_weight ),
				tf.math.imag(x_ifft) * (mem_sq[3] * phy.plank_time ),
				x[2] * (mem_sq[4]) * phy.p_weight,
				x[3] * (mem_sq[5]) * phy.e0,
				], axis=-1, name="ifft_concat" )
		else:
			x = tf.concat( [
				x[0],
				x[1],
				tf.math.real(x_ifft),
				tf.math.imag(x_ifft),
				x[2],
				x[3],
				], axis=-1, name="ifft_concat" )

		# First layer
		if mem_block:
			x = x + (mem_sq_layers[-2]*(tf.sign(x)*phy.h)) # Precession adjustments
		x = tf.expand_dims(x, axis=1)
		x = keras.layers.Conv1D(filters=int(x.shape[-1]/2), kernel_size=1, strides=1, activation=None, dtype=self.dtype)(x)
		x = tf.squeeze(x, axis=1)
		if mem_block:
			x = x + (mem_sq_layers[-1]*(tf.sign(x)*phy.h )) # Precession adjustments
		x = activation['fn'](**activation['args'])(x)

		style = None
		print(f"[LOG] Main block")
		print(x.shape)
		for n, layer in enumerate(layers[1:]):
			if n == len(layers[1:])-1: break # Skip last layer
			# Vortex : TODO: Vortex layer removed.
			skip_x = LayerDense(layer, activation=None, dtype=self.dtype)(x)

			if modulated:
				if style is None:
					style_in = tf.math.reduce_variance(x, keepdims=True, axis=-1)
					print(f"const shape: {constants.shape}")
					print(f"StyleIn shape: {style_in.shape}")
					style = tf.concat( [constants, ], axis=-1)

				style = activation['fn'](**activation['args'])(style)*(phy.one+mem_sq_layers[n])
				if len(x.shape) == 2:
					x = tf.expand_dims(tf.expand_dims(x, axis=0), axis=0)
				mod_kernel = 1#  min( 3, min(x.shape[-1], x.shape[-2]) ) # Kernel size of 1 for linear data
				x, style = ModConv2D(filters=layer, kernel_size=mod_kernel, activation=None, dtype=self.dtype)([x,style])
				x = tf.reshape( x, (-1, layer )) # TODO: Fix dims extention when will be switching to 2D...
				
				x = activation['fn'](**activation['args'])(x)
				print(f"ModConv {x.shape} {x.dtype} | {mod_kernel} | x:{x.shape} | style:{style.shape}")
			
				style = LayerDense(layer, activation=None, dtype=self.dtype)(style)
				style = activation['fn'](**activation['args'])(style)
			else:
				x = keras.layers.Conv1D(filters=layer, kernel_size=1, activation=None, dtype=self.dtype)(x)
				x = activation['fn'](**activation['args'])(x)

			x = x + skip_x
		
		# Final layer
		if x.shape != layers[-1]:
			x = LayerDense(layers[-1], activation=None, dtype=self.dtype)(x)

		if skip.shape[-1] != x.shape[-1]:
			skip = LayerDense(x.shape[-1], activation=None, dtype=self.dtype)(skip)
			skip = activation['fn'](**activation['args'])(skip)
		
		x = tf.math.sinh(x)
		x = tf.add(x, tf.multiply( skip, tf.sign(x)*phy.e_weight) )
		
		# Mem In and Mem bias
		mem_input = x
		if mem_block:
			mem_in = tf.concat( [
				tf.math.reduce_variance(mem_input, axis=0, keepdims=True),
				memblock.memblock
				], axis=-1 )
			
			mem_in = tf.expand_dims(mem_in, axis=0)
			mem_in = keras.layers.MaxPooling1D(pool_size=1, strides=1, dtype=self.dtype)(mem_in)
			mem_in = tf.squeeze(mem_in, axis=0)
			mem_in = LayerDense( memblock.shape[-1], use_bias=False, activation=None)(mem_in) 
			mem_in = tf.keras.layers.Activation('softmax')(mem_in)
			
			mem_add = memblock( memblock.memblock + (mem_in), assign=True)

			print("[MemoryLayer] IN: ", mem_in)
			print("[MemoryLayer] ADD: ",mem_add)
			print("[MemoryLayer] Block: ",memblock.memblock)
			# x = x + tf.reduce_min(mem_add, axis=-1, keepdims=False)*0.0
			
			mem_add = tf.math.reduce_min(mem_add)
			x = tf.add(x, tf.sign(x)*mem_add*phy.plank_time)
		
		# Final layer scaling. Per channel
		# x = tf.split(x, x.shape[-1], axis=-1)
		# for n in range( len(x) ):
		# 	w = tf.Variable( 31415.9265, trainable=True)
		# 	w_bias = (tf.math.ceil(w)*tf.constant(1.0/31415.9265))*phy.h
		# 	w_mult = tf.multiply( tf.math.mod( w, one ) , phy.pi )
		# 	x[n] = tf.add(x[n], tf.sign(x[n])*w_bias) 
		# 	x[n] = tf.multiply(x[n], w_mult)
		# x = tf.concat(x, axis=-1)
		
		w = tf.Variable( 31415.9265, trainable=True)
		w_bias = (tf.math.ceil(w)*tf.constant(1.0/31415.9265))*phy.h
		w_mult = tf.multiply( tf.math.mod( w, one ) , phy.pi )
		x = tf.add(x, tf.sign(x)*w_bias) 
		x = tf.multiply(x, w_mult)

		self.model = tf.keras.Model(name=self.name, inputs=input_layer, outputs=x )
		self.model.trainable = self.trainable

		return self.model