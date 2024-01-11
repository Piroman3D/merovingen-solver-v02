# Merovingen Solver Library
# Author: Dmitry Prozorovsky.
# IOL Office Creative Union.
# MMXXIII

import tensorflow as tf

class HistogramLoss(tf.losses.Loss):
	min = -1.0
	max = 1.0
	bins = 10

	def __init__(self, min=-1.0, max=1.0, bins=16, minmax=True):
		super().__init__()
	
		self.min = min
		self.max = max
		self.bins = bins
		self.minmax = True
	
	@tf.function
	def call(self, y_true, y_pred):
		# hist_true = tf.histogram_fixed_width( y_true, value_range=[self.min, self.max], nbins=self.bins)
		# hist_pred = tf.histogram_fixed_width( y_pred, value_range=[self.min, self.max], nbins=self.bins)
	
		hist_true = tf.cast( tf.histogram_fixed_width( y_true, value_range=[self.min, self.max], nbins=self.bins), dtype=tf.float32 )
		hist_pred = tf.cast( tf.histogram_fixed_width( y_pred, value_range=[self.min, self.max], nbins=self.bins), dtype=tf.float32 )
		hist_true = tf.truediv(hist_true, tf.reduce_max(hist_true) )
		hist_pred = tf.truediv(hist_pred, tf.reduce_max(hist_pred) )

		loss = tf.losses.mean_squared_error(hist_true, hist_pred)
		loss += tf.losses.mean_squared_error( tf.reduce_min(y_true) , tf.reduce_min(y_pred) )
		loss += tf.losses.mean_squared_error( tf.reduce_mean(y_true) , tf.reduce_mean(y_pred) )
		loss += tf.losses.mean_squared_error( tf.reduce_max(y_true) , tf.reduce_max(y_pred) )

		# summ_true = tf.math.reduce_sum(tf.abs(y_true))
		# summ_pred = tf.math.reduce_sum(tf.abs(y_pred))
		# loss += tf.losses.mean_squared_error( summ_true , summ_pred )

		return loss

# [WARNING] Probably PSNR loss here is wrong and require 1.0 - psnr loss.
# Fixed in updated version.
class PSNRLoss(tf.losses.Loss):

	def __init__(self, max=1.0):
		super().__init__()
		self.max = tf.constant(max)
	
	@tf.function
	def call(self, y_true, y_pred):
		y_pred = tf.convert_to_tensor(y_pred)
		y_true = tf.cast(y_true, y_pred.dtype)
		return 10*tf.math.log( (self.max**2)/tf.losses.mean_squared_error( y_pred, y_true) , axis=-1) / 2.303

class CubicLoss(tf.losses.Loss):

	def __init__(self, min=-1.0, max=1.0, bins=16, minmax=True):
		super().__init__()
	
	@tf.function
	def call(self, y_true, y_pred):
		y_pred = tf.convert_to_tensor(y_pred)
		y_true = tf.cast(y_true, y_pred.dtype)

		diff = y_pred - y_true
		return tf.reduce_mean( tf.abs( tf.math.pow( diff, 3.0 )) , axis=-1 )