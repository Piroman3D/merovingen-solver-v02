# Merovingen Solver Library
# Author: Dmitry Prozorovsky.
# IOL Office Creative Union.
# MMXXIII

import tensorflow as tf
from tensorflow import keras

import os
import sys
import time

#Add shared scripts folder to the system path, so they can be imported.
sys.path.insert(0, f'{os.path.normpath( os.path.dirname(os.path.abspath(__file__))) }')

# Model/Layer base class to inherit and override model definition.

model_directory = os.path.abspath('./cerebro/')

class IOLCerebro:

	model = None

	trainbale = True
	#training_momentum = 1e-4

	@property
	def trainable_weights(self):
		# TODO: Count parameters
		return self.model.trainble_weights
	@property
	def non_trainable_weights(self):
		# TODO: Count parameters
		return self.model.non_trainable_weights
	
	optimizer = None

	def prepare_checkpoints_paths(self):
		self.checkpoint_file = "ckpt"

		self.checkpoint_dir = f"{model_directory}/checkpoints/{self.name}"
		self.checkpoint_dir = os.path.normpath(self.checkpoint_dir)

		self.checkpoint_prefix = os.path.join(self.checkpoint_dir, self.checkpoint_file)
		self.checkpoint_prefix = os.path.normpath(self.checkpoint_prefix)

		self.checkpoints_path = os.path.normpath( f"{self.checkpoint_dir}/{self.checkpoint_file}" )
		self.checkpoints_path = os.path.normpath(self.checkpoints_path)

		if os.path.exists(self.checkpoints_path) == False:
			os.makedirs(self.checkpoints_path, exist_ok=True)

		self.latest_checkpoint_path = tf.train.latest_checkpoint(self.checkpoint_dir)
		if self.latest_checkpoint_path != None:
			self.checkpoints_path =  os.path.normpath(self.latest_checkpoint_path)

		print( f'Checkpoints: {self.checkpoint_dir} \n {self.checkpoint_prefix} \n {self.checkpoints_path} \n {self.latest_checkpoint_path}' )

	def __init__(self, input_shape, trainable, name=None, restore=False, summary=False, skip_optimizer=False, lrate=1e-4, decay_steps=1000, optimizer=None, dtype=tf.float32, **args) -> None:
		if name is None: name = self.__class__.__name__
		self.name = name
		self.dtype = dtype
		self.lrate = lrate

		self.trainable = trainable
		self.input_shape = input_shape
		print(f"Initializating model: [{self.name}][{self.input_shape}] : trainable:{self.trainable } | args: {args}")

		self.prepare_checkpoints_paths()
		if 'name' not in args.keys():
			args['name'] = self.name

		self.model = self.build_model(**args)
		self.model.summary()
		
		self.output_shape = self.model.output_shape

		if self.trainable == False:
			skip_optimizer = True

		if skip_optimizer:
			self.checkpoints = tf.train.Checkpoint(
				network=self.model,
			)
		else:
			if optimizer is None:
				model_lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
					self.lrate,
					decay_steps=decay_steps,
					decay_rate=0.96,
					staircase=True
				)
				self.optimizer = tf.keras.optimizers.Adam( learning_rate=model_lr_schedule )
			else:
				self.optimizer = optimizer

			self.checkpoints = tf.train.Checkpoint(
				network=self.model,
				network_optimizer=self.optimizer
			)


		if restore:
			self.restore()  # Restore model

		if summary:
			self.model.summary()
		return None
	
	def build_model(**args):
		print(
			f"[ERROR] Unimplemented abstract function model. You need to overrider this function to generate your model."
		)
		raise 'Attemp to use abstract [build_model] function'
		return self.model

	"""
	Checkppoint operations to restore and save model state.
	"""
	checkpoint_file = None
	checkpoint_dir = None
	checkpoint_prefix = None
	checkpoints_path = None
	latest_checkpoint_path = None

	def restore(self):
		if self.latest_checkpoint_path != None:
			print(
				f"Restoring latest checkpoints [{self.name}] [{self.latest_checkpoint_path}] ...."
			)
			result = self.checkpoints.restore(
				self.latest_checkpoint_path
			)#.expect_partial()
			print(
				f"Checkpoints [{self.name}] restored [{result}]...."
			)
			#time.sleep(5)
		else:
			print(f"Checkpoints [{self.name}] was not found model is clean...")
			#time.sleep(5)
		# return True

	def save_optimized(self, mode='tflite'): # modes ['tflite', 'pb'] 
		converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
		print(f'IOLCerebro: {self.name} converted constructed converter from model.')
		
		converter.optimizations = [tf.lite.Optimize.DEFAULT] #Eight bit quantization
		
		tflite_model = converter.convert()

		print(f'IOLCerebro: {self.name} TF model converted to TFLite model')

		output_path = os.path.join(model_directory, f'{self.name}.{mode}')

		with open(output_path, 'wb') as f:
			f.write(tflite_model)
		
		model_saved = os.path.exists(output_path)
		if model_saved:
			print(f'[SUCCESS] {self.name} saved to: {output_path}')
		else:
			print(f'[ERROR] {self.name} failed to save: {output_path}')
			return False
		
		print(f'Running TF Lite inference test: {output_path}')
		import tflite_runtime.interpreter as tflite
		interpreter = tf.lite.Interpreter(model_path=output_path)
		print(f'Interpreter mode loaded: {output_path}')

		input_details = interpreter.get_input_details()
		output_details = interpreter.get_output_details()
		print(f'TFLite: {self.name}\nInput: {input_details}\nOutput:{output_details}\n')

		return True

	def compile(self, *args, **kwrgs):
		return self.model.compile(*args, **kwrgs)

	def fit(self, *args, **kwrgs):
		return self.model.fit(*args, **kwrgs)
	
	def summary(self):
		return self.model.summary()

	def save(self):
		print(f"Saving checkpoints [{self.name}] to [{self.checkpoints_path}] ....")
		saved = False
		if self.checkpoints == None: raise '[ERROR] Model checkpoints are None'
		if self.checkpoint_prefix == None: raise '[ERROR] Model checkpoint_prefix are None'
		try:
			self.checkpoints.save(file_prefix=self.checkpoint_prefix)
			print(f"Save checkpoints [{self.name}] to [{self.checkpoints_path}]")
			saved = True
			
			# try:
			#	 print(f'Saving TF Light model:')
			#	 converter = tf.lite.TFLiteConverter.from_saved_model(self.checkpoints_path) # path to the SavedModel directory
			#	 tflite_model = converter.convert()
			#	 # Save the model.
			#	 with open(f'./src/cerebro/{self.name}.tflite', 'wb') as f:
			#		 f.write(tflite_model)
			# except:
			#	 print(f'Failed to save TFLight model.')

			return self.checkpoints_path
		except Exception as e:
			print(f"Failed to save checkpoints [{self.name}] to [{self.checkpoints_path}] .... \n Exception: {e}")
			return False

	# Make the class callable so it can be called the same way as keras layers or models.
	@tf.function
	def __call__(self, inputs):
		# return self.model(self)
		return self.model(inputs)


if __name__ == "__main__":
	pass
