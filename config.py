# Merovingen Solver Library
# Author: Dmitry Prozorovsky.
# IOL Office Creative Union.
# MMXXIII

import tensorflow as tf
from utils.losses import HistogramLoss, PSNRLoss
from layers.snake import Snake
from layers.iolactivation_v08 import IOLActivation

from datasets.binary import binary, uint8_to_scalar, uint8_to_pi, uint8_to_plank, normalize
from models.ShallowDense import ShallowModel, ShallowModelFFT
from models.DeepModFFTMemVortex_v08 import DeepModFFTMemVortex

training_time = 0.5*60

activations = {
	"IOL_V09":{
		"enabled": True,
		"fn"   : IOLActivation,
		"args" : {'frequency':0.955, 'negative_fix':True, 'sparse':True, 'trainable':True,},
		#'dtype':tf.float64
	},
	"snake":{
		"enabled": True,
		"fn"   : Snake,
		"args" : {},
	},
	"relu":{
		"enabled": True,
		"fn"   : tf.keras.layers.Activation,
		"args" : {'activation':'relu'},
	},
	"elu":{
		"enabled": False,
		"fn"   : tf.keras.layers.Activation,
		"args" : {'activation':'elu'},
	},
	"gelu":{
		"enabled": False,
		"fn"   : tf.keras.layers.Activation,
		"args" : {'activation':'gelu'},
	},
	"selu":{
		"enabled": False,
		"fn"   : tf.keras.layers.Activation,
		"args" : {'activation':'selu'},
	},
	"sigmoid":{
		"enabled": False,
		"fn"   : tf.keras.layers.Activation,
		"args" : {'activation':'sigmoid'},
	},
	"swish":{
		"enabled": False,
		"fn"   : tf.keras.layers.Activation,
		"args" : {'activation':'swish'},
	},
	"tanh":{
		"enabled": False,
		"fn"   : tf.keras.layers.Activation,
		"args" : {'activation':'tanh'},
	},
	"sinh":{
		"enabled": False,
		"fn"   : tf.math.sinh,
		"args" : {},
	},
	"cosh":{
		"enabled": False,
		"fn"   : tf.math.cosh,
		"args" : {},
	},
	"atanh":{
		"enabled": False,
		"fn"   : tf.math.atanh,
		"args" : {},
	},
	#"softmax":{
	#	"enabled": False,
	#	"fn"   : tf.keras.layers.Activation,
	#	"args" : {'activation':'softmax'},
	#},
	# "None":{"enabled": False,"fn" :tf.keras.layers.Activation, "args" : {'activation':'tanh'},},
}

models = {
	'DeepModFFTMemVortex': {
		'name': "Model_v02",
		'enabled': True,				# Enable / Disable network for fit procedure
		'class'	: DeepModFFTMemVortex,	# Network class to process test
		'train'	: True,					# Enable training mode
		'save'	: False,				# Save weights after the fit is complete
		'args'	: {
			'restore': False,			# Restore previous training results and continue training
			'layers': [1, 16, 16, 1],	# Network layers sequence | ex.: 'layers': [3, 64, -1, 64, 3], # Affects the complexity of the data that can be learned. -1 for Vortex layer (excluded from current code)
			'input_shape': (None, 4), 	# Depricated parameter, required by the parent class of the modell
			'modulated':True,			# Modulated convolution with additional input
			'mem_block':True,			# Memory block with backpropagation and self learning
			'mem_multiplier': 10,		# Multiplaer for memblock size ( length(layers)*mem_multiplier )
			'trainable': True,	
			'dtype': tf.float32
		},
		'time_limit'	:int(training_time),
		'lr'			: 0.00065,
		'lr_decay'		: 0.98, # 0.998,
		'lr_decay_steps': 200,
		'epochs'		: 2000,		# 2500
		'batch'			: 4096*12,	# 4096
		'loss'	: [ 
			tf.keras.losses.Huber(),
			tf.keras.losses.MSE,
			tf.keras.losses.CosineSimilarity(),
			HistogramLoss(),
			PSNRLoss() # Replacement for MSE loss
			],
		'loss_weights' : [ 0.3, 0.1, 0.1, 0.2, 0.3 ], # 0.4, 0.1, 0.3, 0.1, 0.2 
		'multiple_activations_support': True,
		'datasets':{
			'sin_function':{
				'enabled':True,
				'data': binary(
				x_path = './datasets/sin_function_x.bin', x_shape = (-1,1),
				y_path = './datasets/sin_function_y.bin', y_shape = (-1,1),
				no_validation_split = False,
				shuffle = True,
				process_x = [normalize],
				process_y = [normalize],
				)
			},
		}
	},

	# Simple densely connected model for result comparison at fit simple data.
	# Suitable for fast fit and deploy to C code after quantization.
	'ShallowDense': {
		'name': "ModelSimple",
		'enabled': True,
		'train'	: True,
		'save'	: False,
		'class'	: ShallowModel,
		'args'	: {
			'input_shape': (1),
			'layers': [1, 32, 32, 1], # [1, 32, 32, 1]
			'trainable': True,
		},
        'time_limit':int(training_time),
		'lr'	: 0.00025,
		'lr_decay': 0.98, #0.998,
		'lr_decay_steps': 200,
		'epochs': 2000,	#2500
		'batch': 4096*1024,	#4096
		'loss'	: [ 
			tf.keras.losses.Huber(),
			tf.keras.losses.MAE,
			tf.keras.losses.MSE,
			HistogramLoss(), #tf.keras.losses.KLD, HistogramLoss()
			tf.keras.losses.CosineSimilarity()
			],
		'loss_weights' : [ 0.4, 0.1, 0.3, 0.1, 0.2 ],
		'multiple_activations_support': True,
		'datasets':{
			'sin_function':{
				'enabled':True,
				'data': binary(
				x_path = './datasets/sin_function_x.bin', x_shape = (-1,1),
				y_path = './datasets/sin_function_y.bin', y_shape = (-1,1),
				no_validation_split = False,
				shuffle = True,
				process_x = [normalize],
				process_y = [normalize],
				)
			},
		}
	},
}
# end
