# Merovingen Solver Library
# Author: Dmitry Prozorovsky.
# IOL Office Creative Union.
# MMXXIII

import tensorflow as tf
from tensorflow import keras

import random 
import numpy as np

# TODO: Custom data generator,
class DataGen(keras.utils.Sequence):

	def __init__(self, inputs, outputs, batchSize, minRand, maxRand):
		self.inputs = inputs
		self.outputs = outputs
		self.batchSize = batchSize
		self.minRand = minRand
		self.maxRand = maxRand

	#if you want shuffling
	def on_epoch_end(self):
		indices = np.array(range(len(self.inputs)))
		np.random.shuffle(indices)
		self.inputs = self.inputs[indices]
		self.outputs = self.outputs[indices] 

	def __len__(self):
		leng,rem = divmod(len(self.inputs), self.batchSize)
		return (leng + (1 if rem > 0 else 0))

	def __getitem__(self,i):
		start = i*self.batchSize
		end = start + self.batchSize

		x = self.inputs[start:end] * random.uniform(self.minRand, self.maxRand)
		y = self.outputs[start:end]

		return x,y