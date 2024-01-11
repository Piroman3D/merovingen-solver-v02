# Merovingen Solver Library
# Author: Dmitry Prozorovsky.
# IOL Office Creative Union.
# MMXXIII

# The code is not for public domain, redistribution is prohibited. Remove the code if you read this NOTICE
print("(c) Prozorovskiy Dmitry")

import sys, os, time
sys.path.insert(0, './models')
sys.path.insert(0, './datasets')
sys.path.insert(0, './activations')
sys.path.insert(0, './utils')

import tensorflow as tf
from tensorflow import keras

print("GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
gpus = tf.config.list_physical_devices('GPU')

import traceback
import signal
IRQ = True
def signal_handler(sig, frame):
	global IRQ
	print('\n[SIGNAL] Ctrl+C , exiting...')
	IRQ = False
	exit()
	return
signal.signal(signal.SIGINT, signal_handler)

if gpus:
	try:
		# Currently, memory growth needs to be the same across GPUs
		for gpu in gpus:
			tf.config.experimental.set_memory_growth(gpu, True)
			logical_gpus = tf.config.list_logical_devices('GPU')
			print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
	except RuntimeError as e:
		# Memory growth must be set before GPUs have been initialized
		print(e)

import numpy
from datetime import datetime 

# Core configuration to run
from config import activations, models

from plot_activations import plot_activations
from plot_training import plot_train_info, plot_summary
from utils.debug import success, warning, error, wait
from utils.callbacks import TimeStopping, SaveBestWeights

def train_model( name, model, activation, plot_skip=0,
	#act_plot = None,
	**args ):
	
	if model is None: print(f"[WARNING] Model {name} is None"); return None;
	if model['class'] is None: print(f"[WARNING] Model {name} has None class description"); return None
		
	act_name = 'default'
	if activation is not None:
		act_name = activation['name']
		print(f'Using custom activations: {act_name} for {name}')
		model['args']['activation'] = activation
	
	model['args']['name'] = name

	try: _model_ = model['class'](**model['args'])
	except Exception as e:
		error(f"[ERROR] \nFailed to create model\n	model: {name}\n{e}")
		print(traceback.format_exc())
		wait(".", 100)
		return None
	if 'lr_decay_steps' not in model.keys(): model['lr_decay_steps'] = 100
	
	lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
			model['lr'], # initial_learning_rate
			decay_steps = model['lr_decay_steps'] * model['epochs'],
			decay_rate = model['lr_decay'], # 0.998
			staircase=False,
			)
	optimizer = keras.optimizers.Adam(learning_rate=lr_schedule) # epsilon=1e-8
	# optimizer = keras.optimizers.experimental.SGD(learning_rate=lr_schedule)
	opt_name = optimizer.__class__.__name__

	try:
		print(f"{name} keys: {model.keys()}")
		_model_.compile(
			optimizer = optimizer,
			loss = model['loss'],
			loss_weights = model['loss_weights'],
			metrics = ['mae', 'mse', tf.keras.metrics.CosineSimilarity() ],
			jit_compile = False,
			)

		print(f'Model {name} compiled...')
	except Exception as e:
		print(f"[ERROR] {name} compilation failed.")
		print(traceback.format_exc()); return None
		return None
	
	# Prepare training data
	for dataset_n, (dataset_name, dataset) in enumerate(model['datasets'].items()):

		# start_time = datetime.now()
		# tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=f"./logs_{name}")
		try:
			if not dataset['enabled']: print(f"[WARNING] Skipped training: {dataset_name}"); continue
			# print(f"Starting traing: {name} for {dataset_name} dataset")
			data = dataset['data']()
			if data is None: print(f"[WARNING] Model {name} training data is None"); return None
			shuffle = True
			if 'shuffle' in dataset.keys(): shuffle = dataset['shuffle']
			if model['train']:
				#model['trainableWeights'] = numpy.sum([numpy.prod(w.shape) for w in _model_.trainable_weights])
				#model['totalWeights'] = model['trainableWeights'] + model['nonTrainableWeights']
				callbacks = []

				model['trainable_weights'] = tf.reduce_sum([tf.reduce_prod(v.get_shape()) for v in _model_.model.trainable_weights]).numpy()
				model['dataset'] = dataset_name
				success(f"\n[Starting Traing] :")
				success(f"	model:      {name}")
				success(f"	weights:    {model['trainable_weights']}")
				success(f"	class:      {model['class'].__name__}")
				success(f"	activation: {activation['name']}")
				
				if "time_limit" in model.keys():
					success(f"	time limit: {model['time_limit']} seconds")
					callbacks.append( TimeStopping(model["time_limit"], verbose=1) )

				# best_weights_clb = SaveBestWeights()
				# callbacks.append(best_weights_clb)
				# success(f"	trainableWeights: {model['trainableWeights']}")
				# success(f"	nonTrainableWeights: {model['nonTrainableWeights']}")
				# success(f"	totalWeights: {model['totalWeights']}")
				success(f"	dataset:    {dataset_name}\n")
				with tf.device('/GPU:0'):
					dry_run = _model_(data['x_test'][:3])
					#dry_run = _model_.fit(
					#	data['x_test'][:3], data['y_train'][:3],
					#	epochs=1, batch_size=model['batch'],
					#	validation_data=( data['x_test'][:3], data['y_test'][:3]),
					#	)
					print(dry_run.numpy())
					success("Dry run completed, starting fit...")
					start_time = datetime.now()
					train_info = _model_.fit(
						data['x_train'], data['y_train'],
						epochs=model['epochs'], batch_size=model['batch'],
						validation_data=( data['x_test'], data['y_test']),
						shuffle=shuffle,
						callbacks=callbacks,
						#callbacks=[model_checkpoint_callback, tensorboard_callback]
						#callbacks=[keras.callbacks.LearningRateScheduler(schedule, verbose=1)],
						)
					success(f"Model {name} {activation['name']} finished training.")
					# best_weights_clb.load() # Takes to long to save and load
					# success(f"Model {name} {activation['name']} best weights loaded...")
					# train_summary[f"{name}_{activation['name']}"] = train_info
			else:
				train_info = None

			
			if 'save' in model.keys() and model['save']:
				try:
					saved_path = _model_.save()
					success(f'[SAVED] {name} {saved_path}')
				except Exception as e:
					error(f'[FAILED SAVE] {name} \n{e}')
					print(traceback.format_exc()); return None
			else:
				print(f'[SAVE SKIPPED] {name}')

			if 'post_actions' in model.keys() and model['post_actions'] is not None:
				for action in model['post_actions'].keys():
					print(f"Post action: {action}")
					try:
						action = model['post_actions'][action]
						action['args']['model'] = _model_
						action['fn'](**action['args'])

					except Exception as e:
						print(f"Action failed: {e}")
						exc_type, exc_obj, exc_tb = sys.exc_info()
						fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
						print(exc_type, fname, exc_tb.tb_lineno)

			if 'tests' in model.keys() and model['tests'] is not None:
				for item in model['tests']:
					print(f"{item} -> ", end="")
					pred = _model_(item)
					if pred.shape == item.shape:
						print(f"{pred} [{tf.losses.mean_squared_error(item, pred)}] ", end="\n")
					else:
						print(f"{pred}", end="\n")


		except Exception as e:
			error(f'[ERROR] {name} fit model failed:\n{e}')
			error(traceback.format_exc()); return None
			
		end_time = datetime.now()
		if train_info is not None:
			success(f'[SUCCESS] Model {name} finished training: {end_time-start_time}...')
			train_info.history['batch'] = model['batch']
			train_info.history['start_time'] = start_time
			train_info.history['end_time'] = end_time
			train_info.history['duration'] = train_info.history['end_time'] - train_info.history['start_time']
			train_info.history['train_epochs'] = len(train_info.history['loss'])
			train_info.history['epoch_time'] = (train_info.history['duration'].total_seconds() * 1000.0 ) / float(train_info.history['train_epochs'])
			train_info.history['batch_time'] = float(train_info.history['epoch_time'] ) / float(model['batch'])
			
			train_info.history['trainable_count'] = tf.reduce_sum([tf.reduce_prod(v.get_shape()) for v in _model_.model.trainable_weights]).numpy()
			train_info.history['non_trainable_count'] = tf.reduce_sum([tf.reduce_prod(v.get_shape()) for v in _model_.model.non_trainable_weights]).numpy()

			# Add data and predicted data to train hostory to plot the distribution.
			train_info.history['data'] = data
			train_info.history['data_pred'] = {'x':[],'y':[]}
			min = numpy.min(data['x_train'])
			max = numpy.max(data['x_train'])
			shape = _model_.model.input_shape
			shape = (model['batch'],) + shape[1:]
			for n in range(5):
				x = numpy.random.uniform(min*2.0, max*2.0, size=shape)
				y = _model_.model(x)
				train_info.history['data_pred']["x"].append(x)
				train_info.history['data_pred']["y"].append(y)
			train_info.history['data_pred']['x'] = numpy.concatenate( train_info.history['data_pred']['x'], axis=0)
			train_info.history['data_pred']['y'] = numpy.concatenate( train_info.history['data_pred']['y'], axis=0)
			# print(train_info.history['data']["x_train"].shape)
			# print(train_info.history['data_pred']["x"].shape)
			# exit()
			output = plot_train_info( model, name, activation, train_info, plot_skip )
		# nonTrainableParams = np.sum([np.prod(v.get_shape()) for v in model.non_trainable_weights])
		else:
			print("No training data was generated.")
			output = None
		return output
	return None


# Train models with multiple activations
def train_models():
	
	print(f'Starting models training loop...')
	trained_models = {}
	failed_models = {}
	train_data = {}

	for model_n, (model_name, model) in enumerate( models.items() ):
		if model is None: continue
		if 'name' in model.keys():
			model_name = model['name']
		
		if not model['enabled']: print(f"Skipped modeel training: {model_name}");continue

		if model['multiple_activations_support']:
			for n, (act_name, activation) in enumerate(activations.items()):
				if not activation['enabled']: print(f'Activation: {act_name} skipped.'); continue
				
				# _model_name_ = f"{model_name}_{act_name}"
				try:
					print(f'Training model: {model_name} [ {act_name}...')
					activation['name'] = act_name
					train_data[model_name] = train_model(model_name, model, activation)
					trained_models[model_name] = model
					if train_data[model_name] is None:
						print(f'[WARNING] Model {model_name} has no training output...')
					else:
						print(f'Model {model_name} [ {act_name} | Finished training...')
				except Exception as e:
					print(f'[ERROR] Exception during traing of {model_name}:\n{e}');
					print(traceback.format_exc())
					failed_models[model_name] = str(e)
					continue
		else:
			try:
				print(f'Training model: {model_name}...')
				train_data[model_name] = train_model(model_name, model, None)
				trained_models[model_name] = model
				if train_data[model_name] is None:
					print(f'[WARNING] Model {model_name} has no training output...')
				else:
					print(f'Model {model_name} finished training...')
			except Exception as e:
				print(f'[ERROR] Exception during traing of {model_name}:\n{e}')
				print(traceback.format_exc())
				failed_models[model_name] = str(e)
				continue
	# print(f"Ttraining finished: {len(train_summary.keys())} models: {train_summary.keys()}")
	# print(train_summary.keys())
	# print(train_summary.values())
	plot_summary()
	return

def dry_run():
	print("Starting initialization...")
	model = tf.keras.Sequential([
		tf.keras.layers.Flatten( input_shape=(8,) ),
		tf.keras.layers.Dense(16, activation='relu'),
		tf.keras.layers.Reshape( (-1, 16) ),
		tf.keras.layers.Conv1D( 8 , kernel_size=1, strides=1),
		tf.keras.layers.Reshape( (-1, 8) )
	])
	x_train = numpy.random.uniform( -1.0, 1.0, size=(1024, 8))
	y_train = numpy.random.uniform( 0.0, 0.0, size=(1024, 8))
	# x_valid = numpy.random.uniform( -1.0, 1.0, size=(1024, 8))
	# y_valid = numpy.random.uniform( -1.0, 1.0, size=(1024, 8))
	optimizer = keras.optimizers.Adam(learning_rate=0.001)
	model.compile(optimizer=optimizer, loss=tf.keras.losses.MSE)
	info = model.fit( x_train, y_train, batch_size=1024, epochs=1)
	success(f"Initization finished: {info.history['loss'][-1]}")
	return

if __name__ == "__main__":

	# 0. Dry run to initialize functions.
	dry_run()

	# 1. Plot activations
	plot_activations(activations)
	# exit()
	
	# 2. Train models with multiple activations
	train_models()
	exit()
	
	pass