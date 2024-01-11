# Merovingen Solver Library
# Author: Dmitry Prozorovsky.
# IOL Office Creative Union.
# MMXXIII

import datetime
import time
# from typeguard import typechecked

import tensorflow as tf

# Save best model weights during model fit
class SaveBestWeights(tf.keras.callbacks.Callback):
	def __init__(self, metric='val_loss', max=False):
		self.metric = metric
		self.max = max
		if self.max:
			self.best = float('-inf')
		else:
			self.best = float('inf')
	
	def on_epoch_end(self, epoch, logs=None):
		# print(logs)
		metric = logs[self.metric]
		print(f"\n[metrics]: {metric} | {self.best}")
		if self.max:
			if metric > self.best:
				print(f"Saving best weights: {metric}")
				self.best = metric
				self.weights = self.model.get_weights()
		else:
			if metric < self.best:
				print(f"Saving best weights: {metric}")
				self.best = metric
				self.weights = self.model.get_weights()

	# Load best saved weights
	def load(self):
		self.model.set_weights(self.weights)
	
	def print(self):
		print(f"best: {self.best} | weights:")
		print(self.weights)

# From tfa utils. Function to limit training time
class TimeStopping(tf.keras.callbacks.Callback):

	# @typechecked
	def __init__(self, seconds: int = 86400, verbose: int = 0):
		super().__init__()

		self.seconds = seconds
		self.verbose = verbose
		self.stopped_epoch = None

	# Due to model different compilation time at first epochs
	# the first epoch is skipped when counting time to make final results more correct to compare.
	# To consider time for the first batch need to precompile jit function
	# But currently no way was found to do it.
	def on_train_begin(self, logs=None):
		self.stopping_time = time.time() + self.seconds

	def on_epoch_end(self, epoch, logs={}):
		#if epoch == 0:
		#	self.stopping_time = time.time() + self.seconds

		if time.time() >= self.stopping_time:
			self.model.stop_training = True
			self.stopped_epoch = epoch

	def on_train_end(self, logs=None):
		if self.stopped_epoch is not None and self.verbose > 0:
			formatted_time = datetime.timedelta(seconds=self.seconds)
			msg = "Timed stopping at epoch {} after training for {}".format(
				self.stopped_epoch + 1, formatted_time
			)
			print(msg)

	def get_config(self):
		config = {
			"seconds": self.seconds,
			"verbose": self.verbose,
		}

		base_config = super().get_config()
		return {**base_config, **config}