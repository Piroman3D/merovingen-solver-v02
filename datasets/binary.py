import os
import numpy as np
import tensorflow as tf
from utils.debug import success, warning, error
byte2float = (1/255)*2.0

def uint8_to_scalar(data):
	data = data * byte2float # (1/255)*2 # Got 0 ~ range.
	data = (data - 1.0)
	print(f"min: {np.min(data)} max: {np.max(data)} mean: {np.mean(data)}")
	return data

def uint8_to_pi(data):
	data = uint8_to_scalar(data) * 3.1415926535897932384626433
	print(f"min: {np.min(data)} max: {np.max(data)} mean: {np.mean(data)}")
	return data 

def uint8_to_plank(data):
	data = uint8_to_scalar(data) * 6.62607015e-3
	print(f"min: {np.min(data)} max: {np.max(data)} mean: {np.mean(data)}")
	return data

def tensor_info(x):
	return f"min: {tf.reduce_min(x):.5f} | max: {tf.reduce_max(x):.5f} | mean: {tf.reduce_mean(x):.5f}"

def normalize(x):
	_min = tf.reduce_min(x)
	return (tf.multiply( tf.truediv( x - _min, tf.reduce_max(x)-_min ), tf.constant(2, x.dtype) ) - tf.constant(1, x.dtype)).numpy()

class binary():
	x_path = None
	x_shape = (-1, 5)
	y_path = None
	y_shape = (-1, 6)

	#x_values = None
	#y_values = None
	
	train_ratio = 0.6
	test_ratio = 0.2
	
	shuffle = True

	process_x = [],
	process_y = [],

	data ={
		'x_train': None,
		'y_train': None,
		'x_test': None,
		'y_test': None,
		'x_validate': None,
		'y_validate': None,
	}

	loaded = False
	def __init__(self,
		x_path = './x.bin', x_shape = (-1, 5),
		y_path = './y.bin', y_shape = (-1, 6),
		train_ratio = 0.6, test_ratio = 0.2,
		shuffle = True,
		no_validation_split = False,
		process_x = [],
		process_y = [],
	):
		self.x_path = x_path
		self.y_path = y_path
		self.x_shape = x_shape
		self.y_shape = y_shape

		self.train_ratio = train_ratio
		self.test_ratio = test_ratio
		self.shuffle = shuffle
		self.no_validation_split = no_validation_split

		self.loaded = False

		self.process_x = process_x
		self.process_y = process_y
		
		# print(self.__class__.__name__)
		# print(self.process_x)
		# print(self.process_y)
		return

	def fload(self, path):
		values = None
		if ".bin" in path:
			values = np.fromfile(path, dtype='<f')
		elif ".npy" in path:
			values = np.load(path)
		print(f"	Loaded: {values.shape} from: {path}")
		return values

	def load(self):
		if not self.loaded:
			print(f'[Loading]:')
			print(f'	x = {self.x_path} [ valid: {os.path.exists(self.x_path)} ]')
			print(f'	y = {self.y_path} [ valid: {os.path.exists(self.y_path)} ]')

			x_values, y_values = None, None

			if type(self.x_path) is str:
				x_values = self.fload(self.x_path)
				#print(f"Loaded {x_values.shape}: {self.x_path}")
			elif type(self.x_path) is list:
				x_values = []
				[x_values.extend( self.fload(x) ) for x in self.x_path]
				x_values = np.array(x_values)

			if type(self.y_path) is str:
				y_values = self.fload(self.y_path)
				#print(f"Loaded {y_values.shape}: {self.y_path}")
			elif type(self.y_path) is list:
				y_values = []
				[y_values.extend( self.fload(y) ) for y in self.y_path]
				y_values = np.array(y_values)

			# print(f"Reshaping X: {x_values.shape} -> {self.x_shape}")
			# x_values = np.reshape( x_values, self.x_shape)
			# print(f"Reshaping Y: {y_values.shape} -> {self.y_shape}")
			# y_values = np.reshape( y_values, self.y_shape)
			
			print(f"Preprocessing data: {self.process_x}")

			if len(self.process_x)>0:
				for process in self.process_x:
					print(f"X processed: {process.__name__}")
					x_values = process(x_values)
			if len(self.process_y)>0:
				for process in self.process_y:
					print(f"Y processed: {process.__name__}")
					y_values = process(y_values)

			SAMPLES = min(x_values.shape[0], y_values.shape[0])
			x_values = x_values[:SAMPLES]
			y_values = y_values[:SAMPLES]

			if self.no_validation_split:
				print("[ATTENTION] : No validation split. copying values to validation dataset.")
				self.data['x_train'] = self.data['x_test'] = self.data['x_validate'] = x_values
				self.data['y_train'] = self.data['y_test'] = self.data['y_validate'] = y_values
			else:
				TRAIN_SPLIT =  int(self.train_ratio * SAMPLES)
				TEST_SPLIT = int(self.test_ratio * SAMPLES + TRAIN_SPLIT)
				self.data['x_train'], self.data['x_test'], self.data['x_validate'] = np.split(x_values, [TRAIN_SPLIT, TEST_SPLIT])
				self.data['y_train'], self.data['y_test'], self.data['y_validate'] = np.split(y_values, [TRAIN_SPLIT, TEST_SPLIT])

			success(f"[Dataset Loaded]:")
			success(f"	x_train:    {len(self.data['x_train'])} shape: {self.data['x_train'].shape} | {tensor_info(self.data['x_train'])}")
			success(f"	y_train:    {len(self.data['y_train'])} shape: {self.data['y_train'].shape} | {tensor_info(self.data['y_train'])}")
			success(f"	x_test:     {len(self.data['x_test'])} shape: {self.data['x_test'].shape} | {tensor_info(self.data['x_test'])}")
			success(f"	y_test:     {len(self.data['y_test'])} shape: {self.data['y_test'].shape} | {tensor_info(self.data['y_test'])}")
			success(f"	x_validate: {len(self.data['x_validate'])} shape: {self.data['x_validate'].shape} | {tensor_info(self.data['x_validate'])}")
			success(f"	y_validate: {len(self.data['y_validate'])} shape: {self.data['y_validate'].shape} | {tensor_info(self.data['y_validate'])}")
		
		self.loaded = True
		return self.data
	
	def __call__(self):
		return self.load()