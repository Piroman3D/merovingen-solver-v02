# Merovingen Solver Library
# Author: Dmitry Prozorovsky.
# IOL Office Creative Union.
# MMXXIII

import os
import sys
import numpy
import tensorflow as tf
try:
	import progressbar
except:
	progressbar = None
	print("[WARNING] progressbar2 module not found")

from datetime import datetime
from matplotlib import pyplot as plt

fontsize = 10

chart_history_info = {}

def plot_chart(name, label, axn, x, y, start, end, color=None, ymax=None, limits =[[-2.0, 2.0],[-1.1, 1.1]] ):
	# v_axis = 0.001 # history: 0.2, 0.05
	axn.title.set_text(f'{name}')
	if end is None:
		end = len(y)
	
	autofit = False
	if ymax is None:
		y_max = numpy.max(y[start:end])+0.001
		autofit = True

	axn.axis([start, end, numpy.min(y[start:end])-0.01, ymax ])
	for axis in ['top','bottom','left','right']: axn.spines[axis].set_linewidth(0.3)
	axn.set_xlabel('')
	axn.set_ylabel('')
	# axn.set_yscale('log')
	axn.grid(True, linewidth=0.2)
	
	if color is None: color = "b."
	if label is None: label = "chart"
	axn.plot(x[start:end], y[start:end], color, label=label, linestyle='solid', linewidth=0.25, markersize=0.3)
	
	# plot zero line
	axn.plot(x[start:end], [0.0 for _ in x[start:end]], "#252525", label=label, linestyle='solid', linewidth=0.20, markersize=0.1)
	if autofit:
		axn.autoscale(enable=True, axis='both', tight=None)
	else:
		axn.axis([start, end, numpy.min(y[start:end])-0.01, ymax ])
		axn.autoscale(enable=True, axis='x', tight=None)

	axn.axis('on')
	# plt.yticks( [0.0, 0.00001, 0.0001, 0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0] )
	axn.yaxis.set_major_formatter(lambda x, pos: f"{(x):.4g}")
	if limits is not None:
		axn.set_xlim(limits[0])
		axn.set_ylim(limits[1])

def plot_sample_data(ax, info, name='Data'):
	ax.title.set_text(f'{name}')
	x_train = info.history['data']['x_train']
	y_train = info.history['data']['y_train']
	x_pred = info.history['data_pred']['x']
	y_pred = info.history['data_pred']['y']

	_min_x = min( numpy.min(x_train), numpy.min(x_pred) )
	_max_x = max( numpy.max(x_train), numpy.max(x_pred) )
	_min_y = min( numpy.min(y_train), numpy.min(y_pred) )
	_max_y = max( numpy.max(y_train), numpy.max(y_pred) )

	ax.axis([ _min_x, _max_x, _min_y, _max_y ])
	for axis in ['top','bottom','left','right']: ax.spines[axis].set_linewidth(0.3)
	ax.set_xlabel('')
	ax.set_ylabel('')
	# axn.set_yscale('log')
	ax.grid(True, linewidth=0.2)

	ax.scatter(x_train, y_train, s=0.05, c="#3B44F6", label='reference')
	ax.scatter(x_pred, y_pred, s=0.05, c="#3EC70B", label='predicted')

	ax.autoscale(enable=True, axis='both', tight=None)
	ax.axis('on')
	ax.legend( loc="lower left", bbox_to_anchor=(0.0, -0.4))
	
	limits =[[-2.0, 2.0],[-1.1, 1.1]]
	ax.set_xlim(limits[0])
	ax.set_ylim(limits[1])
	# ax.yaxis.set_major_formatter(lambda x, pos: f"{(x):.4g}")
	return ax

def plot_model_info(ax, model_name, model, activation, train_info):
	print(train_info.history.keys())
	summary = []
	loss_names = []
	if type(model["loss"]) is list:
		for item in model['loss']:
			if hasattr(item, "__class__"):
				item_name = item.__class__.__name__
				if item_name == 'function':
					item_name = item.__name__
				loss_names.append(item_name)
	else:
		loss_names = str(model['loss'])

	loss_names = [x.lower().replace("mean_absolute_error", "mae").replace("mean_squared_error", "mse").replace("kl_divergence", "kl").replace("cosinesimilarity", "cos") for x in loss_names]
	loss_names = ",".join(loss_names)
	epochs = range(1, len(train_info.history['loss']) + 1)
	model_name = str(model_name).replace('_', ' ')
	def field(target, name, value):
		target.append(f"{f'{name}:':10s} {value}")
	field(summary, f"Summary", "")
	summary.append("")
	field(summary, f"name", 		f"{model_name} [{str(activation['name']).replace('_', ' ')}]")
	# field(summary, f"activation",	f"{str(activation['name']).replace('_', ' ')}")
	field(summary, f"dataset", 		f"{model['dataset']}" )
	field(summary, f"epochs",  		f"{len(train_info.history['loss'])}" )
	field(summary, f"batch",  		f"{train_info.history['batch']}" )
	field(summary, f"lr",  			f"{model['lr']}" )
	field(summary, f"weights",  	f"{train_info.history['trainable_count']} | {train_info.history['non_trainable_count']}" )
	if 'shape' in model['args']:
		field(summary, f"shape",  	f"{model['args']['shape']}" )
	field(summary, f"duration",		f"{str(train_info.history['duration'])}" )
	field(summary, f"epoch/ms",		f"{train_info.history['epoch_time']:.2f} epoch/ms" )
	# field(summary, f"batch/ms",		f"{train_info.history['batch_time']:.2f} batch/ms" )
	field(summary, f"loss", 		f"{loss_names}" )
	summary.append("")
	field(summary, f"           training:       validation", "")
	# field(summary, f"accuracy",	f"{numpy.max(accuracy):.6f}")
	field(summary, f"loss",	 	f"{numpy.min(train_info.history['loss']):.6f}        {numpy.min(train_info.history['val_loss']):.6f}")
	field(summary, f"mse",		f"{numpy.min(train_info.history['mse']):.6f}        {numpy.min(train_info.history['val_mse']):.6f}")
	field(summary, f"mae", 		f"{numpy.min(train_info.history['mae']):.6f}        {numpy.min(train_info.history['val_mae']):.6f}")
	if 'cosine_similarity' in train_info.history.keys():
		field(summary, f"cos",		f"{numpy.max(train_info.history['cosine_similarity']):.6f}        {numpy.max(train_info.history['val_cosine_similarity']):.6f}")
	summary.append("")

	summary = "\n".join(summary)
	txt = ax.text( 0.0, 0.95, summary, horizontalalignment='left', verticalalignment='top', size=fontsize, clip_on=False)
	ax.grid(False)
	ax.axis('off')

def configure_plot():
	plt.rcParams.update({'font.size': 7})
	plt.rcParams.update({'figure.autolayout':True})
	plt.rc('font', size=fontsize)          # controls default text sizes
	plt.rcParams.update({'font.family': 'monospace'})
	plt.rc('axes', titlesize=fontsize)     # fontsize of the axes title
	plt.rc('axes', labelsize=fontsize)     # fontsize of the x and y labels
	plt.rc('xtick', labelsize=fontsize)    # fontsize of the tick labels
	plt.rc('ytick', labelsize=fontsize)    # fontsize of the tick labels
	plt.rc('legend', fontsize=fontsize)    # legend fontsize
	plt.rc('figure', titlesize=fontsize)   # fontsize of the figure title
	plt.axis("off")

def prepare_canvas( nrows, ncols, figsize = (8.27, 11.69)):
	plt.cla()
	plt.clf()
	
	# plt.style.use('dark_background')
	configure_plot()
	
	return plt.subplots( nrows=nrows, ncols=ncols, figsize=figsize, layout="constrained")
	
def plot_summary():
	global chart_history_info

	cols = 3
	rows = int(round( len(chart_history_info.keys())/cols))+1
	rows = rows + 2
	configure_plot()
	fig = plt.figure( figsize = (8.27, 11.69), constrained_layout=True)
	gs = fig.add_gridspec(rows, cols)

	ax_summary = fig.add_subplot(gs[0, :])
	ax_start = fig.add_subplot(gs[1, :])
	ax_end = fig.add_subplot(gs[2, :])

	# Plot validation data and reference data comparison.
	row = col = 0
	row = 3
	for n, (name, data) in enumerate(chart_history_info.items()):
		if col >= cols: row = row+1; col = 0
		ax = fig.add_subplot( gs[ min(row, rows-1), col] )
		ax = plot_sample_data(ax, data, name=name)
		ax.get_legend().remove()
		col = col + 1

	summary = []
	duration = None
	for model_name, info in chart_history_info.items():
		duration = info.history['end_time'] - info.history['start_time']

	summary.append(f"Summary:\n" )
	summary.append(f"Duration: {duration}\n")
	summary.append(f"Model:                validation loss:" )


	colors = ["#27374D", "#3EC70B", "#3B44F6", "#A149FA", "#F90716", "#FF5403", "#FFCA03", "#FFF323"]
	it = 0
	_printed = []
	for model_name, info in chart_history_info.items():
		_name = model_name.replace("_", " ")
		_min_val_loss = numpy.min( info.history['val_mse'] )
		summary.append(f"    {_name:23s}: {_min_val_loss}")
		
		epochs = []
		duration = info.history['end_time'] - info.history['start_time']
		total_epochs = len(info.history['val_mse'])
		for n in range(total_epochs):
			epochs.append( (float(n)/float(total_epochs)) * float(duration.total_seconds()) )
		start = 0
		end = len(info.history['val_mse'])
		half = int(end/2.0)
		plot_chart("validation mse", model_name, ax_start, epochs, info.history['val_mse'], start, half ,color = colors[it], limits=None)
		plot_chart("validation mse", model_name, ax_end, epochs, info.history['val_mse'], half, end ,color = colors[it], ymax=0.025, limits=None)
		plot_chart("validation mse", 'zero', ax_end, epochs, [0.0 for _ in info.history['val_mse']], half, end ,color = colors[it], ymax=0.025, limits=None)
		it = it + 1

	plt.rc('legend', fontsize=int(fontsize/2)) # legend fontsize
	handles, labels = ax_start.get_legend_handles_labels()
	by_label = dict(zip(labels,handles))
	ax_start.legend( by_label.values(), by_label.keys() ) # loc="upper left", bbox_to_anchor=(-1.0, -1.0)
	for n, handle in enumerate( ax_start.get_legend().legendHandles ):
		handle.set_color(colors[n])

	summary = "\n".join(summary)
	txt = ax_summary.text( 0.0, 0.95, summary, horizontalalignment='left', verticalalignment='top', size=fontsize, clip_on=False)
	ax_summary.grid(False)
	ax_summary.axis('off')

	plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.2)
	fig.tight_layout()

	ctime = datetime.now().strftime("%Y-%m-%d-%H-%M")
	output_path =  f"./traininfo/Summary_{ctime}.png"
	plt.savefig(output_path, dpi=280)
	print(f"Training summary saved: {output_path}")
	return

def plot_train_info(model, model_name, activation, train_info, SKIP=1 ):
	global chart_history_info

	ABSOLUTE_PATH = os.path.dirname(os.path.abspath(__file__))

	# if EPOCHS > 9: SKIP = 5
	# if EPOCHS > 40: SKIP = 20
	# if EPOCHS > 100: SKIP = 50
	# if EPOCHS > 1000: SKIP = 100

	fig, ax = prepare_canvas(4, 2)

	epochs = []
	duration = train_info.history['end_time'] - train_info.history['start_time']
	total_epochs = len(train_info.history['loss'])
	for n in range(total_epochs):
		epochs.append( (float(n)/float(total_epochs)) * float(duration.total_seconds()) )

	# Model information
	plot_model_info(ax[0][0], model_name, model, activation, train_info)

	# Plot reference data and predicted data and append data for summary ptint
	plot_sample_data(ax[0][1], train_info)
	key = f"{model_name}_{activation['name']}"
	chart_history_info[key] = train_info
	# charts
	idx = 1
	start = 0
	end = len(train_info.history['loss'])
	half = int(end/2.0)
	#mse
	if 'mse' in train_info.history.keys():
		plot_chart("mse", "training"  ,ax[idx][0], epochs, train_info.history['loss'],   start, half, "b.", limits=None)
		plot_chart("mse", "validation",ax[idx][0], epochs, train_info.history['val_mse'],start, half, "g.", limits=None)
		plot_chart("mse", "training"  ,ax[idx][1], epochs, train_info.history['loss'],   half, end, "b.", limits=None)
		plot_chart("mse", "validation",ax[idx][1], epochs, train_info.history['val_mse'],half, end, "g.", limits=None)
		idx = idx +1
	#mae
	if 'mae' in train_info.history.keys():
		plot_chart("mae", "training"  ,ax[idx][0], epochs, train_info.history['mae'], 	start, half, "b.", limits=None)
		plot_chart("mae", "validation",ax[idx][0], epochs, train_info.history['val_mae'],start,half, "g.", limits=None)
		plot_chart("mae", "training"  ,ax[idx][1], epochs, train_info.history['mae'], 	half, end, "b.", limits=None)
		plot_chart("mae", "validation",ax[idx][1], epochs, train_info.history['val_mae'],half,end, "g.", limits=None)
		idx = idx +1
	# accuracy
	if 'accuracy' in train_info.history.keys():
		plot_chart("accuracy", "training"  ,ax[idx][0], epochs, train_info.history['accuracy'], 	start, half, "b.", limits=None)
		plot_chart("accuracy", "validation",ax[idx][0], epochs, train_info.history['val_accuracy'],start, half, "g.", limits=None)
		plot_chart("accuracy", "training"  ,ax[idx][1], epochs, train_info.history['accuracy'], 	half, end, "b.", limits=None)
		plot_chart("accuracy", "validation",ax[idx][1], epochs, train_info.history['val_accuracy'],half, end, "g.", limits=None)
		idx = idx +1
	# cosine_similarity
	if 'cosine_similarity' in train_info.history.keys():
		plot_chart("cosine_similarity", "training"  ,ax[idx][0], epochs, train_info.history['cosine_similarity'], 	start, half, "b.", limits=None)
		plot_chart("cosine_similarity", "validation",ax[idx][0], epochs, train_info.history['val_cosine_similarity'],start, half, "g.", limits=None)
		plot_chart("cosine_similarity", "training"  ,ax[idx][1], epochs, train_info.history['cosine_similarity'], 	half, end, "b.", limits=None)
		plot_chart("cosine_similarity", "validation",ax[idx][1], epochs, train_info.history['val_cosine_similarity'],half, end, "g.", limits=None)
		idx = idx +1

	# fig.tight_layout()
	plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.2)

	ctime = datetime.now().strftime("%Y-%m-%d-%H-%M")
	output_path =  f"./traininfo/{model_name}_{activation['name']}_{ctime}.png"
	plt.savefig( output_path, dpi=280)
	# plt.show(block=False)
	# Photo( os.path.abspath(output_path) )
	print(f"Training [{model_name}] summary saved: {output_path}")
	return fig, ax

# default dataset is x_values
def save_model(model_name, model, path, data ):
	h5_path = os.path.join( path, f"{model_name}.h5")
	model.save(h5_path)
	print(f'TF model saved: {h5_path}')
	print(f'Optimizing the model...')

	# def representative_data_gen():
	#		 for n in range(100):
	#			 print(f'sampling quantization: { model.input_shape }')
	#			 sample_shape = []
	#			 for dim in model.input_shape:
	#				 if dim is None:
	#					 sample_shape.append(1)
	#				 else:
	#					 sample_shape.append(dim)
	#			 sample_shape = tuple(sample_shape)
	#			 data = np.random.uniform( sample_shape )
	#			 yield [ np.array(data, dtype=np.float32, ndmin=2) ]

	def representative_data_gen():
		representative_tensors = tf.data.Dataset.from_tensor_slices(data['x_train']).batch(1).take(len(data['x_train']))
		bar = None
		if progressbar is not None:
			bar = progressbar.ProgressBar(max_value=len(representative_tensors))
		print(f'Iterating representative dataset')
		
		for n, input_value in enumerate(representative_tensors):
			yield_value = [ tf.dtypes.cast(input_value, tf.float32) ]
			#print(f'.', end='')
			# Model has only one input so each data point has one element.
			if bar is not None:
				bar.update(n)
			yield yield_value
		if bar is not None:
			bar.finish()
		print("representative_data_gen finished")

	# Model Optimization -------------------------------------------------------------------------
	# Convert the model to the TensorFlow Lite format with quantization
	converter = tf.lite.TFLiteConverter.from_keras_model(model)
	# Set the optimization flag.
	converter.allow_custom_ops = False
	converter.optimizations = [tf.lite.Optimize.DEFAULT]
	#converter.target_spec.supported_types = [tf.float16] # Not supported by TEENSY:Failed to populate a temp TfLiteTensor struct from flatbuffer data! Node DEQUANTIZE
	#TODO: Optimize model with converter to run with INT8 and INT8 operations.
	#converter.representative_dataset = representative_dataset
	#converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
	#converter.inference_input_type = tf.uint8	# or tf.uint8 # Quantizes only input?
	#converter.inference_output_type = tf.uint8	# or tf.uint8 # Quantizes only input?

	converter.representative_dataset = representative_data_gen

	# converter.target_spec.supported_ops = [
	#	 tf.lite.OpsSet.TFLITE_BUILTINS, # enable TensorFlow Lite ops.
	#	 #tf.lite.OpsSet.SELECT_TF_OPS # enable TensorFlow ops.
	# ]
	tflite_model = converter.convert()
	# Model Optimization -------------------------------------------------------------------------
	print(f'Checking quantized model: ')
	interpreter = tf.lite.Interpreter(model_content=tflite_model)
	#input_type = interpreter.get_input_details()[0]['dtype']
	#print(f'Input details: {interpreter.get_input_details()[0]}')
	#print('input: ', input_type)
	#output_type = interpreter.get_output_details()[0]['dtype']

	#print('output: ', output_type)
	#Run inference test of the converted model.
	print(f'Testing TFLite model:')
	interpreter.allocate_tensors()	# Needed before execution!
	input = interpreter.get_input_details() #Get input tensor
	print(f'Input: {input}')
	output = interpreter.get_output_details() #Get output tensor
	print(f'Output: {output}')
	mse = tf.keras.losses.MeanSquaredError()
	average_mse = 0.0

	# bar = None
	# if progressbar:
	#	bar = progressbar.ProgressBar(max_value=progressbar.UnknownLength)
		
	for n in range(0, len(data['x_test'][:1000]) ):
		input_data = tf.convert_to_tensor( data['x_test'][n] )
		input_data = tf.expand_dims(input_data, aximarkersize=0) #Add batch dimension.
		reference_data = tf.convert_to_tensor( data['y_test'][n] )
		reference_data = tf.expand_dims(reference_data, aximarkersize=0) #Add batch dimension.

		# TFLite inference
		interpreter.set_tensor(input[0]['index'], input_data)
		interpreter.invoke()
		tflite_predictions = interpreter.get_tensor(output[0]['index'])
		
		# Original model inference
		model_predictions = model.predict(input_data)

		model_test_mse = mse( reference_data, tflite_predictions).numpy()
		if n == 0:
			average_mse = model_test_mse
		else:
			average_mse = (average_mse + model_test_mse)/2.0

		print(f'Input tensor: {input_data}')
		print(f'    -> TFLite: {tflite_predictions} | mse.: {model_test_mse} | avg_mse: {average_mse}')
		print(f'    -> Model:  {model_predictions}')

	# Save the model to disk
	tf_lite_path = os.path.join(path, f"{model_name}.tflite")
	open( tf_lite_path, "wb").write(tflite_model)
	print(f'tflite saved: {tf_lite_path}')
	# Generate C files
	from tensorflow.lite.python.util import convert_bytes_to_c_source
	source_text, header_text = convert_bytes_to_c_source(tflite_model, f"{model_name}", include_path=f"{model_name}.h")

	header_path = os.path.join( path, f"{model_name}.h")
	open( header_path, 'w').write(header_text)
	print(f'header saved: {header_path}')

	cpp_path = os.path.join( path, f"{model_name}.cpp")
	open( cpp_path, 'w').write(source_text)
	print(f'cpp saved: {cpp_path}')