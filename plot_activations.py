# Merovingen Solver Library
# Author: Dmitry Prozorovsky.
# IOL Office Creative Union.
# MMXXIII

import numpy
import tensorflow as tf
import matplotlib.pyplot as plt
from utils.debug import success

fontsize = 7
def plot_activation(name, act, x, default_y, drange, ax):
	success(f"Running plot: {name} [{type(act['fn'])}]")
	if act['fn'].__class__.__name__ == 'function':
		activation = act['fn']
	else:
		activation = act['fn'](**act['args'])
		
	with tf.GradientTape() as g:
		g.watch(x)
		y = activation(x)
		
		with tf.GradientTape() as g2:
			g2.watch(y)
			#z = activation(y)

	derivative = g.gradient(y, x)

	ax.set_title(f'{name}', fontsize=fontsize)
	ax.axis(  [-drange, drange, -drange, drange])
	ax.set_xlabel('x', fontsize=fontsize)
	ax.set_ylabel('y', fontsize=fontsize)
	ax.grid(True)
	ax.grid(linewidth=0.1)

	major_ticks = numpy.arange(-drange, drange, 1.0)
	minor_ticks = numpy.arange(-drange, drange, 0.5)

	ax.set_xticks(major_ticks)
	ax.set_xticks(minor_ticks, minor=True)
	ax.set_yticks(major_ticks)
	ax.set_yticks(minor_ticks, minor=True)
	ax.set_aspect('equal')
	ax.set_xticklabels(major_ticks, fontsize=4)
	ax.set_yticklabels(major_ticks, fontsize=4)

	if y.shape[-1] < 1:
		ax.plot(x, default_y, 'yellowgreen', label='x', linewidth=0.3)
		ax.plot(x, y, 'mediumblue', label='act. fn', linewidth=0.3)
		ax.plot(x, derivative, 'darkmagenta', label='act. fn\' ', linewidth=0.3)
	else:
		# TODO: Make plot for multi channel activations.
		channels = y.shape[-1]
		y = tf.split(y, channels, axis=-1)
		# default_y = tf.split(default_y, y.shape[-1])
		derivative = tf.split(derivative, channels, axis=-1)
		# for n in range(channels):
		ax.plot(x, default_y, 'yellowgreen', label='x', linewidth=0.3)
		ax.plot(x, y, 'mediumblue', label='act. fn', linewidth=0.3)
		ax.plot(x, derivative, 'darkmagenta', label='act. fn\' ', linewidth=0.3)
	
	ax.grid(True)
	ax.axis('on')
	#ax[i].plot(x, derivative2, 'violet', label='act. fn\'\' ', linewidth=0.3)
	plt.setp(ax.spines.values(), linewidth=0.05) # Border lines
	# lgd = ax.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0))
	return ax #, lgd

def plot_activations(activations):
	drange = 5.0
	fn_step = 0.001

	# Prepare plot input data for activation plot
	x = tf.convert_to_tensor( numpy.arange(-drange, drange, step=fn_step) , dtype=tf.float32 )
	default_y = x = tf.convert_to_tensor( numpy.arange(-drange, drange, step=fn_step) , dtype=tf.float32)
	
	# Prepare plot canvas
	cols = 3
	rows = int(round( len(activations)/cols))+1
	figsize=figsize=( 3*(cols), len(activations.items()) )
	fig, ax = plt.subplots(rows, cols, figsize=figsize)

	fontsize = 7
	title = fig.suptitle("Activation functions summary", y=1.0, size=fontsize)
	plt.rcParams.update({'font.size': fontsize})
	plt.rcParams.update({'figure.autolayout':True})
	plt.rc('font', size=fontsize)          # controls default text sizes
	plt.rcParams.update({'font.family': 'monospace'})
	plt.rc('axes', titlesize=fontsize)     # fontsize of the axes title
	plt.rc('axes', labelsize=fontsize)     # fontsize of the x and y labels
	plt.rc('xtick', labelsize=fontsize)    # fontsize of the tick labels
	plt.rc('ytick', labelsize=fontsize)    # fontsize of the tick labels
	plt.rc('legend', fontsize=fontsize)    # legend fontsize
	plt.rc('figure', titlesize=fontsize)   # fontsize of the figure title
	
	for j in range(cols):
		for i in range(rows): 
			ax[i][j].grid(False)
			ax[i][j].axis('off')

	# Plot activations to canvas
	activation_plots = {}
	row = col = 0
	for n, (act_name, activation) in enumerate(activations.items()):
		if row >= rows: col = col+1; row = 0
		# print(row, col)
		ax[row][col] = plot_activation(act_name, activation, x, default_y, drange, ax[row][col])
		activation_plots[act_name] = ax[row][col]
		row = row + 1

	print(f'Plot activations complete...')
	fig.tight_layout()
	output_path = './traininfo/activations_x_y_dx.png'
	plt.savefig( output_path, dpi=280, bbox_extra_artists=(title,), bbox_inches='tight')
	print(f'Finished activation layers summary generation.')
	print(f'Output summary: {output_path}')