# Merovingen Solver Library
# Author: Dmitry Prozorovsky.
# IOL Office Creative Union.
# MMXXIII

import numpy
import tensorflow as tf
from spectre.spectre import float_to_vec3, convert_image
from utils.debug import success

fontsize = 7

def setup_plot():
	# Prepare plot canvas
	cols = 1
	rows = int(1)
	# figsize = ( 3*(cols)*5, 1*5)
	# fig, ax = plt.subplots(rows, cols, figsize=figsize)
	
	# figsize = ( 3*(cols)*5, 1*5)
	fig, ax = plt.subplots(rows, cols)

	fontsize = 7
	title = fig.suptitle("Spectre", y=1.0, size=fontsize)
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

	return ax

def test_fn(x):
	return x*2.0

def plot_spectre(ax):
	success(f"Running spectre plot...")

	x0 = numpy.linspace(0, 999999, 999999)
	x1 = numpy.linspace(-1.0, 1.0, 4096)
	
	x = x1
	default_y = x
	x = tf.convert_to_tensor(x, dtype=tf.float32)
	default_y = tf.convert_to_tensor(default_y)
	with tf.GradientTape() as g:
		g.watch(x)
		y = float_to_vec3(x)
		# y = test_fn(x)
		#with tf.GradientTape() as g2:
		#	g2.watch(y)
		#	#z = activation(y)
	
	derivative = g.gradient(y, x)

	if derivative is None:
		print("[ERROR] No derivative computed.")
		# exit()
		derivative = x*0.0

	print(f"Sprectre derivative [{derivative.shape}]: {derivative}")
	y = (y/(255.0*0.5))-1.0
	print(f"Output [{y.shape}] : {y}")
	y = tf.split(y, 3, axis=-1)
	drange = 1.2

	ax.set_title(f'spectre', fontsize=fontsize)
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

	ax.plot(x, default_y, 'yellowgreen', label='x', linewidth=0.3)
	ax.plot(x, y[1], 'mediumblue', label='act. fn', linewidth=0.3)
	# ax.plot(x, y[1], 'mediumblue', label='act. fn', linewidth=0.3)
	# ax.plot(x, y[2], 'mediumblue', label='act. fn', linewidth=0.3)
	ax.plot(x, derivative, 'darkmagenta', label='act. fn\' ', linewidth=0.3)

	ax.grid(True)
	ax.axis('on')
	#ax[i].plot(x, derivative2, 'violet', label='act. fn\'\' ', linewidth=0.3)
	plt.setp(ax.spines.values(), linewidth=0.05) # Border lines
	# lgd = ax.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0))
	return ax #, lgd

if __name__ == "__main__":

	# Convert test image
	path = "./spectre/birch.png"
	convert_image(path)
	print(f"[SPECTRE] Image saved to: {path}.xxx.png ")
	
	import time
	time.sleep(3.0)

	# Plot spectre convertion graphs
	try:
		import matplotlib.pyplot as plt
		ax = setup_plot()
		plot_spectre(ax)
		plt.show()
	except:
		print("[Exception] During plotting spectre graphs")

	
