# Merovingen Solver Library
# Author: Dmitry Prozorovsky.
# IOL Office Creative Union.
# MMXXIII

import random
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import json
import os
import sys

#Add shared scripts folder to the system path, so they can be imported.
sys.path.insert(0, f'{os.path.normpath( os.path.dirname(os.path.abspath(__file__))) }')

import debug
#sprint = print # system print
debugprint = debug.print # Debug print information with time stamp

import os
import time
import types
import base64
from io import BytesIO

import PIL
import progressbar

#msle = tf.keras.losses.MeanSquaredLogarithmicError()
msle = tf.keras.losses.MeanSquaredLogarithmicError() # Does not fit in the shadows, but fit in the lights....
ms = tf.keras.losses.MeanSquaredError()

def check_image_extension(path):
	if path.lower().endswith('png') : return True
	if path.lower().endswith('jpg') : return True
	if path.lower().endswith('jpeg') : return True
	return False

#retrieves or sets file information, via creating description ini file.
file_info_cache = {}
def file_info(path, key=None, value=None, disable_cache=False):
	
	config_path = f'{path}.ini'
	#print(f'Getting file info: {config_path}')

	contents = None

	if disable_cache == False:
		if key == None and value == None:
			if path in file_info_cache.keys():
				return file_info_cache[path]

	exists = os.path.exists(config_path)

	if exists:
		try: #if the file contents are not JSON or broken..
			with open(config_path, 'r') as file:
				contents = file.read()
				contents = json.loads(contents)
		except:
			contents = {}
	else:
		contents = {}

	if key == None and value == None:
		if disable_cache == False:
			file_info_cache[path] = contents
		return contents
	elif key != None and value != None:
		contents[key] = value
		with open(config_path, 'w') as file:
			file.write( json.dumps(contents) )

	debugprint(f'File info [{config_path}] saved: \n {contents}')
	return contents

cached_paths = {}
def load_folder(path, max_count=500, shuffle=False, exclude=[], filters=None,
	size=None, color_space='rgb', color_range=(-1,1), linear=True,
	method='cover', preserve_aspect_ratio=True , adjust_size=None,
	shuffle_seed=None, topdown=False, only_paths=False,
	disable_cache=False, raw=False
	):
	
	# Skip path dependencies on multiple platforms.
	# path = unifyPath(path)

	if isinstance( path, list):
		output = []
		for _path in path:
			_path_content = load_folder(_path, max_count=max_count, only_paths=only_paths, color_range=color_range, color_space=color_space, size=size, exclude=exclude, linear=linear)
			output.extend(_path_content)
		
		debugprint(f'Merged paths content: {len(output)} items.')
		return output

	global cached_paths

	index = 0
	files_cache = []
	checked_count = 0

	if path in cached_paths.keys() and disable_cache == False:
		#restore the list of files if we have already parsed this folder.
		files_cache = cached_paths[path]
	else:
		#Parse txt file containing paths.
		if path.lower().endswith('.txt'):
			debugprint(f'Reading path cache file: {path}')
			file_handler = open(path, "r", encoding="utf-8")
			lines = file_handler.readlines()
			file_handler.close()
			_lines_count = len(lines)
			
			n = 0
			item_path = None
			for n, line in enumerate(lines):
				item_path = line.rstrip('\n')
				files_cache.append(item_path)
			
			print(f'File [{path}] cached [{n} of {_lines_count}: {item_path}')
		else:
			# parsing the folder
			for root, dirs, files in os.walk(path, topdown=topdown):
				print(f'[cached: {len(files_cache)} | checked: {checked_count}] Loading path: {root}')
				should_break = False
				#debugprint(files)
				#if len(files_cache) > max_count-1 and max_count != -1:
				#	break
				for name in files:
					checked_count = checked_count+1
					item_path = os.path.join(root, name).lower()
					
					should_load = check_image_extension(item_path)

					
					if not should_load:
						continue

					#Exlude folder names from exlude = []
					if exclude is not None:
						if len(exclude)>0 and should_load:
							for exclude_item in exclude:
								path_string = item_path.lower()
								path_string = path_string.replace('\\', '|')
								path_string = path_string.replace('/', '|')
								if f'|{exclude_item.lower()}|' in path_string:
									should_load=False
									continue
					
					if should_load:
						files_cache.append(item_path)

						if shuffle == False and max_count != -1:
							if len(files_cache) == max_count:
								debugprint(f'Max count reached: {len(files_cache)} of {max_count}')
								should_break = True

					if should_break:
						debugprint(f'Folder parsing stopped...')
						break
				if should_break:
					debugprint(f'Folder parsing stopped...')
					break
		#store path in cache, so no need to parse them again if requesting same path.
		cached_paths[path] = files_cache

	#Shuffle paths.
	if shuffle:
		if shuffle_seed == None:
			shuffle_seed = np.random.seed()
		random.Random(shuffle_seed).shuffle(files_cache)

	#Show how many files will be loaded.
	output_data = []
	if max_count < 0:
		max_count =  len(files_cache)
	required_file_count = min( len(files_cache), max_count )

	debugprint(f'Enumerating cache: {len(files_cache)}')
	bar = progressbar.ProgressBar(max_value=required_file_count, redirect_stdout=True)
	for n, item_path in enumerate(files_cache):
		
		if len(output_data) > max_count-1 and max_count != -1:
			break

		#if index > max_count-1 and max_count != -1:
		#	break

		#Should load, with file_info filter should be moved here.
		should_load = True
		if filters != None:
			info = file_info(item_path, disable_cache=disable_cache)
			#debugprint(f'{item_path} : {info}')
			#debugprint(file_info)
			for key, value in filters.items():
				#debugprint( f'{str( info.get(key) )} != {str(value)}' )
				if str( info.get(key) ) != str(value):
					should_load = False
					continue

		if should_load:
			if only_paths == False:
				image_tensor = load( item_path ,
									size=size, method=method, preserve_aspect_ratio=preserve_aspect_ratio, adjust_size=adjust_size,
									color_space=color_space, linear=linear,  color_range=color_range,
									raw=raw
									)
				if image_tensor != None:
					output_data.append(image_tensor)
					if len(image_tensor.shape) == 4:
						debug.draw_tensor4(image_tensor, 'loaded_image')
					elif len(image_tensor.shape) == 3:
						debug.draw_tensor3(image_tensor, 'loaded_image')

					debugprint(f"Loaded image [{len(output_data)}] [{color_space}]: {item_path}")
				else:
					debugprint(f"Failed to load [{len(output_data)}] [{color_space}]: {item_path}")
			else:
				output_data.append(item_path)

		#index = index+1
		bar.update( len(output_data) )
			
	bar.finish()

	if len(output_data) < 1:
		debugprint(f'Warning: Image loader did not load any images from: {path}')
	return output_data

def grid(x, size=6):
	x = tf.stack(x, axis=0)
	t = tf.unstack(x[:size * size], num=size*size, axis=0)
	rows = [tf.concat(t[i*size:(i+1)*size], axis=0) for i in range(size)]
	image = tf.concat(rows, axis=1)
	return image

def save(tensor, path, scale=True, raw=False):
	
	#path = unifyPath(path)
	
	if raw:
		tf.keras.preprocessing.image.save_img(path, tensor, scale=False)
		if os.path.exists(path):
			debugprint(f'Saved: {path}')
			return path
		else:
			debugprint(f'Failed save: {path}')
			return None

	if len(tensor.shape) == 4:
		if tensor.shape[0] == 1:
			tensor = tf.squeeze(tensor , axis=0)
		else:
			debugprint(f'Error: saving multibatch tensor: {tensor.shape} . Skipped.')
			return None

	tensor = tensor - debug.global_color_range[0]
	tensor = tensor / ( debug.global_color_range[1]-debug.global_color_range[0] )

	if debug.global_color_linear == True:
		tensor = tf.pow( tf.abs(tensor), 1.0/2.2 ) #Positive linear space to sRGB conversion.

	tensor = tensor*255

	tensor = tf.clip_by_value( tensor, 0, 255, name=None)
	tensor = tf.cast(tensor , tf.uint8)

	tf.keras.preprocessing.image.save_img(path, tensor, scale=False)
	if os.path.exists(path):
		debugprint(f'Saved: {path}')
		return path
	else:
		debugprint(f'Failed save: {path}')
		return None
	
	return None


############################################### Image Randomization - Shout be processed be shift  and lerp
def random_rotate(x, angle=25.0):
	"""
	Randomly rotates an image within the bounds (-angle, angle)
	"""
	if random.uniform(0.0, 1.0) > 0.5:
		_angle = (random.uniform(-1.0, 1.0) * angle * np.pi) / 180.0
		x = tfa.image.rotate(x, _angle, fill_mode='reflect', interpolation='bilinear')
	#image = tf.image.central_crop(image, central_fraction=0.8)
	return x

def random_crop(x, factor=0.8):
	"""
	Randomly crop
	"""
	original_shape = ( x.shape[0], x.shape[1] )
	crop = random.uniform(factor, 1.0)
	if crop != 1.0:
		x = tf.image.central_crop(x, central_fraction=crop)
		x = tf.image.resize(x, size=(original_shape), antialias=True, method='bilinear' )
	return x

def random_light(x):
	"""
	Applys random augmentations related to lighting
	"""
	x = tf.image.random_brightness(x, 0.05,)
	x = tf.image.random_contrast(x, 0.9, 1.1,)
	return x

def random_noise(x):
	if random.uniform(0.0, 1.0) > 0.2:
		noise =  ( tf.random.normal( [ x.shape[0], x.shape[1], x.shape[2] ] )+1  )
		x = x + (noise * tf.random.uniform(shape=[], minval=-0.05, maxval=0.05, dtype=tf.float32))
	return x
############################################### Image Randomization - Shout be processed be shift  and lerp

def greyscale(x):
	return tf.image.rgb_to_grayscale(x)

def load_fast(path, expand_dims=True, shift=True, randomize=['flip', 'rotate', 'crop', 'noise', 'light'], size=None):
	
	if not os.path.exists(path):
		print(f'[WARNING] File does not exists: {path}')
		return None

	try:
		image = tf.keras.preprocessing.image.load_img(path, target_size=size, interpolation='bilinear')
		image = tf.keras.preprocessing.image.img_to_array(image)
		image = tf.cast(image, tf.float32)
	except Exception as e:
		print(f'Broken file: {path}')
		return None

	image = image/255.0
	if randomize != None:
		if 'flip' in randomize:
			image = tf.image.random_flip_left_right(image)
		if 'noise' in randomize:
			image = random_noise(image)
		if 'rotate' in randomize:
			image = random_rotate(image)
		if 'crop' in randomize:
			image = random_crop(image)
		if 'light' in randomize:
			image = random_light(image)

	image = tf.pow( tf.abs(image), 2.2 ) #srg2linear
	image = tf.clip_by_value(image, 0, 1)
	
	if shift:
		image = (image - 0.5)*2.0

	if expand_dims:
		image = tf.expand_dims(image, axis=0)

	return image

def image_base64(npyImage):
	pil_image = PIL.Image.fromarray(npyImage)
	#buff = BytesIO()
	#pil_img.save(buff, format="PNG")
	#new_image_string = base64.b64encode(buff.getvalue()).decode("utf-8")

	buffered = BytesIO()
	pil_image.save(buffered, format="PNG")
	return str( base64.b64encode( buffered.getvalue() ), encoding='utf8')

def load(path,
	invert=False, color_space='rgb', color_range=(-1,1), interpolation='bicubic', linear=True,
	preserve_aspect_ratio=True, size=None, adjust_size=None,
	desaturate=False, method='fit', raw=False , asBase64=False, set_global=True):
	color_space = color_space.lower()
	
	#path = unifyPath(path)

	if not os.path.exists(path):
		print(f'[WARNING] File does not exists: {path}')
		return None

	if set_global:
		debug.set_global_color_space(color_space)
		debug.set_global_color_range(color_range)
		debug.set_global_color_linear(linear)
	try:
		if path.lower().endswith('npy'):
			tensor = np.load(path)
			tensor = tf.convert_to_tensor(tensor)
			return tensor

		if raw or asBase64:
			#print(size)
			image = tf.keras.preprocessing.image.load_img(path, target_size=size , interpolation=interpolation)
			if asBase64:
				buffered = BytesIO()
				image.save(buffered, format="PNG")
				return str( base64.b64encode( buffered.getvalue() ), encoding='utf8')
			else:
				image = tf.convert_to_tensor(image)
			return image
		else:
			if preserve_aspect_ratio:
				image = tf.keras.preprocessing.image.load_img(path, target_size=None, interpolation='bicubic')
			else:
				image = tf.keras.preprocessing.image.load_img(path, target_size=size, interpolation='bicubic')

		image = tf.keras.preprocessing.image.img_to_array(image)
		#image = np.array([image])
		image = tf.cast(image, tf.float32)
		image = tf.expand_dims(image, axis=0)
		#print(f'Loaded image shape: {image.shape}')


	except Exception as e:
		# Probably faced PNG bomb during decompression of the data. 
		debugprint(f'Raw image processing function failed opening: {path}')
		debugprint(e)
		return None
	#image = tf.expand_dims(image, axis=0)

	#if method.lower() == 'fit':
	if (image.shape[1], image.shape[2]) != size:
		debugprint(f"Resize: {image.shape} -> {size}")
		image = tf.image.resize(image, size , method=interpolation,  antialias=True, preserve_aspect_ratio=preserve_aspect_ratio)

	if adjust_size != None:
		image = adjsut_size(image, adjust_size)

	#Load image in base64 mode to display on the webpage.
	#range: -1 ~ 1
	image = image/255
	image = tf.clip_by_value(image, 0, 1) #Make sure input image is alwasy in 0 ~ 1 range

	#All input images are always in sRGB space, so convert them to linear if needed.
	if linear:
		image = srgb_2_linear(image)

	if invert:
		image = 1-image

	if 'rgb' in color_space or color_space=='multispace':
		rgb_image = image

	if 'hsv' in color_space or color_space=='multispace':
		hsv_image = tf.image.rgb_to_hsv(image)

	if color_space == 'rgb':
		image = rgb_image

	if color_space == 'lms':
		image = lms_image

	if color_space == 'hsv':
		image = hsv_image

	if color_space == 'rgbhsv':
		image = tf.concat( [rgb_image, hsv_image], axis=-1)

	if color_space == 'multispace':
		sobel = tf.image.sobel_edges(rgb_image)
		sobel = sobel**2 
		sobel = tf.math.reduce_sum(sobel,axis=-1) # sum all magnitude components
		sobel = tf.sqrt(sobel)/5
		#sobel = sobel/50
		image = tf.concat( [rgb_image, sobel, hsv_image], axis=-1)

	#If we need desaturated image no matter what the space is, we are looking for average values.
	if desaturate:
		image = desaturate(image)

	
	#map image to the output range
	if color_range == (-1, 1):
		image = (image - 0.5)*2.0
	else:
		image = lerp( color_range[0], color_range[1], image)
	
	#lms color space is out of 0 - 1 range.
	#if color_space != 'lms':
	#image = tf.clip_by_value(image, color_range[0], color_range[1])
	#image_size = (image.shape[1], image.shape[2])

	# if size != None and size != image_size:
	#	 #image = tf.image.resize(image, adjusted_shape , method='bicubic',  antialias=True, preserve_aspect_ratio=False)

	return image

#input must be in range [0 - 255]
def tensor_2_base64(image):
	image = tf.clip_by_value( image, 0, 255, name=None)
	image = tf.cast(image , tf.uint8)
	image = tf.squeeze(image , axis=0)
	image = tf.image.encode_png(image , compression=-1)
	image = base64.b64encode( image.numpy() )
	image = str(image, encoding='utf8')
	return image

def merge_batch_images(tensor, res=64):
	batch_size = tensor.shape[0]
	tensor = tf.image.resize(tensor, (res, res) )
	#assert rows * cols == batch_size
	rows = cols = int(batch_size/2)
	canvas = np.zeros(shape=[res * rows, res * cols, tensor.shape[-1] ], dtype=tensor.dtype)
	for row in range(rows):
		y_start = row * res
		for col in range(cols):
			x_start = col * res
			index = col + row * cols
			canvas[y_start:y_start + res, x_start:x_start + res, :] = tensor[index, :, :, :]
	return canvas

def adjsut_size(image, size):
	count_x = int( image.shape[1]/size[0]  )
	count_y = int( image.shape[2]/size[1] )
	if count_x == 0:
		count_x = 1
	if count_y == 0:
		count_y = 1

	adjusted_shape = (size[0]*count_x, size[1]*count_y)

	debugprint(f'Adjusting image shape: {image.shape} to {adjusted_shape}')
	#if adjust_size != (size[0] , size[1]):
	image = tf.image.resize(image, adjusted_shape, antialias=True, method='bicubic', preserve_aspect_ratio=False)
	return image


#def rgb_dimensions(image):
#	rgb, other = tf.split( image , [3, image.shape[-1]-3], axis=-1 )
#	return rgb

# Image data augmentation function
def randomize(
	image,
	cutout = 0,
	flip = 0,
	brightness = 0,
	rotation = 0,
	contrast = 0,
	saturation = 0,
	noise = 0,
	hue = 0,
	gamma = 0,
	gain=0,
	light=0,
	hue_noise=0,
	clip = True, # we must clip the image! We need to keep it in [-1,1] range to keep it as correct image data.
	#cutout_shape = ( 32, 32 )
	seed = None
	):
	shift = 0.05 #Shift is required for image operations. Negative image space is not supported. As well as 0 value in images does not work correctly
	#Always shift the image, even the range is 0 - 1 , nothing bad happens if range is more than 2 for example. But bad things happen if the range is -1 ~ 1
	output = shift_positive(image) 
	output = tf.clip_by_value(output, 0, 1.0) + shift
	# 0 == random.randint(0,1)
	if seed == None:
		seed = int(random.randrange(0, 9999999, 1) )

	#if cutout != 0:
	#	output = tfa.image.random_cutout( output, mask_size = cutout, constant_values = -1, seed=seed+1 )

	if flip != 0:
		output = tf.image.flip_left_right(output)

	#if rotation != 0:
	#	if rotation == None:
	#		rotation = random.uniform(-1, 1) / 5
	#	output = tfa.image.rotate( output , rotation, interpolation = 'BILINEAR' , fill_mode='reflect')

	if light != 0:
		#original_input = tf.convert_to_tensor( output.numpy() )
		light_noise_scale = int( lerp(5, output.shape[1]/10, random.uniform(0, 1) ) )
		light_noise =  ( tf.random.normal( [ output.shape[0], output.shape[1], output.shape[2], 1 ], seed=seed+2  )+1  )
		light_noise = tf.image.resize(light_noise, ( light_noise_scale, light_noise_scale ) , method='bicubic',  antialias=True)
		light_noise = tf.image.resize(light_noise, ( output.shape[1], output.shape[2] ) , method='bicubic',  antialias=True)
		light_noise = tf.image.resize(light_noise, ( output.shape[1], output.shape[2] ) , method='bicubic',  antialias=True)

		light_gamma = random.uniform(-0.1, 2)
		light_gain = random.uniform(-0.1, 0.5)

		light_noise = tf.image.adjust_gamma(light_noise, gamma=2.5, gain=0.8 )
		light_noise = tf.image.resize( light_noise , size=(output.shape[1], output.shape[2]) )
	   
		light_mask = tf.image.resize(output, ( int(output.shape[1]/10), int(output.shape[2]/10) ) , method='bicubic',  antialias=True) - 0.5
		light_mask = tf.clip_by_value(light_mask, 0, 1)
		light_mask = tf.image.adjust_saturation(  light_mask, 0 )
		light_mask = tf.image.adjust_gamma(  light_mask, gamma=1.2, gain=5  )
		light_mask = tf.clip_by_value(light_mask, 0, 1)
		light_mask = tf.image.resize( light_mask , size=(output.shape[1], output.shape[2]) )

		light_mod = tf.clip_by_value( (light_noise*light_mask)-0.5 , 0, 1)

		adjusted_image = tf.image.adjust_gamma( output, gamma=1+random.uniform(-0.1, 2), gain=1+random.uniform(-0.1, 2) )
		adjusted_image = tf.clip_by_value( adjusted_image , 0, 1)

		output = lerp(output, adjusted_image, light_mod*light )
		output = output

	if brightness != 0:
		output = tf.image.random_brightness( output, brightness, seed=seed)
		
	if contrast != 0:
		output = tf.image.random_contrast( output, 1-contrast, 1+contrast, seed=seed)

	if saturation != 0:
		#output = tf.image.adjust_saturation(output , saturation)
		output = tf.image.random_saturation(output, 1-saturation, 1+saturation, seed=seed)

	if hue != 0:
		output = tf.image.random_hue( output, hue, seed=seed)

	if gamma != 0 or gain != 0:
		gamma_value = random.uniform(-gamma, gamma)
		gain_value = random.uniform(-gain, gain)
		output = tf.image.adjust_gamma(output, gamma=1+gamma_value, gain=1+gain_value )

	if noise != 0:
		noise_scale_A = int( lerp(2, output.shape[1]/10, random.uniform(0, 1) ) )
		noise_scale_B = int( lerp(2, output.shape[1]/10, random.uniform(0, 1) ) )

		uniform_noise_A =  ( tf.random.normal([ output.shape[0], output.shape[1], output.shape[2], output.shape[3] ])+1 )/2
		uniform_noise_B =  ( tf.random.normal([ output.shape[0], output.shape[1], output.shape[2], output.shape[3] ])+1 )/2

		uniform_noise_A = tf.image.resize(uniform_noise_A, ( noise_scale_A, noise_scale_A ) , method='bicubic',  antialias=True)
		uniform_noise_B = tf.image.resize(uniform_noise_B, ( noise_scale_B, noise_scale_B ) , method='bicubic',  antialias=True)

		uniform_noise_A = tf.image.resize(uniform_noise_A, ( output.shape[1], output.shape[2] ) , method='bicubic',  antialias=True)
		uniform_noise_B = tf.image.resize(uniform_noise_B, ( output.shape[1], output.shape[2] ) , method='bicubic',  antialias=True)

		#noise_saturation = 0.5
		uniform_noise = (uniform_noise_A+uniform_noise_B)/2
		overlay_noise = tf.image.random_saturation( uniform_noise, 0, 1, seed=seed+6 ) 
		overlay_noise = overlay_noise - 1

		noise_value = random.uniform(0,noise)
		output = output + (overlay_noise*noise_value)

	if hue_noise != 0:
		output = tf.image.rgb_to_hsv(output)
		noise_scale = int( lerp(output.shape[1]/20, output.shape[1]/15, random.uniform(0, 1) ) )

		uniform_noise =  ( tf.random.normal([ output.shape[0], output.shape[1], output.shape[2], 1 ])+1 )/2
		uniform_noise = tf.image.resize(uniform_noise, ( noise_scale, noise_scale ) , method='bicubic',  antialias=True)
		uniform_noise = tf.image.resize(uniform_noise, ( output.shape[1], output.shape[2] ) , method='bicubic',  antialias=True)
		uniform_noise = tf.image.adjust_gamma(uniform_noise, gamma=2.5, gain=1 )

		output = output + (uniform_noise*hue_noise)
		output = tf.clip_by_value(output, 0, 1)
		output = tf.image.hsv_to_rgb(output)

	output = output - shift
	output = shift_negative(output)
	#if clip == True:
	
	output = tf.clip_by_value(output, -1.0, 1.0)

	return output

@tf.function
def shift_negative(image):
	return (image-0.5)*2.0

@tf.function
def shift_positive(image):
	return (image+1.0)/2.0

@tf.function
def hsv(image):
	image = shift_positive(image)
	image = tf.image.rgb_to_hsv(image)
	image = shift_negative(image)
	image = tf.clip_by_value(image, -1.0, 1.0)

	#image = shift_positive(image)
	#image = tf.image.hsv_to_rgb(image)
	#image = shift_negative(image)

	return image

def flip(image, flip):
	#flip = tf.convert_to_tensor(flip)
	#debugprint(flip.shape)
	#assert image.shape[0] == flip.shape[0]
	flips = flip.numpy()
	#debugprint(flips)
	matricies = []
	for flip in flips:
		if flip[0] == 1.0:
			#Horizontal flip matrix
			matricies.append( [ -1, 0, image.shape[1] - 1, 0, 1, 0.0, 0.0, 0.0] )
		else:
			#Normal matrix, no transform
			matricies.append( [ 1, 0, 0, 0, 1, 0.0, 0.0, 0.0] )
		#debugprint(f"flip: {flip[0]}")
	
	matrix = tf.stack(matricies, axis=0)
	#debugprint(matrix.shape)
	#debugprint(  )
	#matrix =
	#flip_matrix = 

	#debugprint(f"image shape: {image.shape} | flip_shape: {flip.shape}")
	return tfa.image.transform(image, matrix)

def blur( image, factor):
	original_shape = image.shape
	output = tf.image.resize(image, ( int( image.shape[1]*factor), int( image.shape[2]*factor ) ) , method='nearest',  antialias=True)
	output = tf.image.resize(output, (original_shape[1], original_shape[2]) , method='nearest',)
	return output

def extract_edges(image, factor=0.8, desaturate=True):
	#debug.draw_tensor4(image, 'input' )
	image_blur = blur(image , factor=factor)
	output = tf.abs( (image_blur - image) )
	#output = difference * image
	return output

#@tf.function
def lerp(x,y,alpha):
	return ( (1.0-alpha) * x) + ((alpha) * y) 


def load_segmentation_model(model_config):
	from thirdparty.image_segmentation_keras.keras_segmentation.models.all_models import \
		model_from_name

	model = model_from_name[model_config['model_class']](
		model_config['n_classes'],
		input_height=model_config['input_height'],
		input_width=model_config['input_width']
		)
	debugprint("loaded weights ", f"Weights: {model_config['checkpoints']}")
	model.load_weights(model_config['checkpoints'])
	return model

@tf.function
def desaturate(image , saturation=0.0 ):
	desaturated = tf.math.reduce_sum(image, axis=-1, keepdims=False )/image.shape[-1]
	desaturated = tf.expand_dims(desaturated, axis=-1)
	desaturated = tf.repeat( desaturated, image.shape[-1] , axis=-1)
	if saturation != 0.0:
		desaturated = lerp(desaturated, image, saturation)
	return desaturated

segmentation_model = None
#random.seed(0)

@tf.function
def distance(a, b):
	debugprint(f'UNIMPLEMENTED FUNCTION DISTANCE!!!')
	#output = tf.reduce_summ( distance( a-b) )
	#desaturated = tf.math.reduce_sum(image, axis=-1, keepdims=False )/image.shape[-1]
	#desaturated = tf.expand_dims(desaturated, axis=-1)
	#desaturated = tf.repeat( desaturated, image.shape[-1] , axis=-1)
	#if saturation != 0.0:
	#	desaturated = lerp(desaturated, image, saturation)
	return a

segmentation_model = None

class_colors = None
def calc_segmentation(image, visualize=True, debugDraw=True):
	from thirdparty.image_segmentation_keras.keras_segmentation.data_utils.data_loader import get_image_array
	from thirdparty.image_segmentation_keras.keras_segmentation.predict import visualize_segmentation

	global class_colors
	global segmentation_model
	if class_colors == None:
		class_colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in range(200)]

	segmentation_model_config = {
		"input_height": 473,
		"input_width": 473,
		"n_classes": 150,
		"model_class": "pspnet_50",
		"checkpoints": r"./thirdparty/image_segmentation_keras/checkpoints/pspnet50_ade20k.h5"
	}

	if segmentation_model is None:
		segmentation_model = load_segmentation_model(segmentation_model_config)

	debugprint(f'Segmenting image: {image.shape}')
	input_image = tf.image.resize(image, (segmentation_model_config['input_width'], segmentation_model_config['input_height']) )

	input_image = shift_positive(input_image)
	input_image = input_image * 255

	segmentation_images = []
	#input_images = input_image
	#input_image

	tensors = tf.split(input_image, num_or_size_splits=input_image.shape[0], axis=0)
	debugprint(f'Segmenting {len(tensors)} images...')

	for tensor in tensors:

		segmentation = segmentation_model.predict( tensor )[0]
		#segmentation = segmentation.reshape((segmentation_model_config['input_height'],  segmentation_model_config['input_width'], segmentation_model_config['n_classes'])).argmax(axis=2)
		segmentation = segmentation.reshape( (segmentation_model.output_height,  segmentation_model.output_width, 150) ).argmax(axis=2)
		debugprint(f'Raw segmentation shape: {segmentation.shape}')
		segmentation = visualize_segmentation(segmentation, n_classes=150,colors=class_colors,prediction_width=473,prediction_height=473)

		#debugprint(f'Finished segmentation, with colors: {segmentation.shape}')
		#segmentation = visualize_segmentation( segmentation, n_classes=segmentation_model_config['n_classes'], colors=class_colors )
		#segmentation = get_colored_segmentation_image(segmentation, n_classes=150, colors=class_colors)

		segmentation = tf.convert_to_tensor(segmentation)
		segmentation = tf.expand_dims(segmentation , axis=0)/127.5 - 1
		
		segmentation_images.append(segmentation)
	
	segmentation = tf.concat(segmentation_images, axis=0)
	segmentation = tf.image.resize(segmentation, (image.shape[1], image.shape[2] ) )
	
	if debugDraw:
		debug.draw_tensor4(image, 'segmentation_input')
		debug.draw_tensor4(segmentation, 'segmentation_output')

	#debugprint(f'Segmentation shape: {segmentation.shape}')

	return segmentation

def build_segmentation(images ,visualize=True):
	debugprint(f'Starting building segmentation for {len(images)} images')
	output = []
	for image in images:
		image_segmentation = calc_segmentation(image,visualize=visualize)
		output.append(image_segmentation)
	return output

def resize(image, factor=1.0, size=None, method='bicubic', antialias=True):

	if type(image) is list :
		list_output = []
		for item in image:
			list_output.append( resize(item, factor=factor, size=size, method=method, antialias=antialias) )
		return list_output

	output = image
	if size is None:
		if factor != 1.0:
			output = tf.image.resize(image, ( int( image.shape[1]*factor), int( image.shape[2]*factor ) ) , method=method,  antialias=antialias)
		else:
			output = image
	else:
		output = tf.image.resize(image, size , method=method,  antialias=antialias)

	return output

# style_session = {}
# style_graph = {}
# style_preds = {}
# style_placeholder = {}
# def style(image, style_name, factor=1.0, desaturate_input = False, desaturate_output = True,
#	 device_name='/GPU:0', styles_path=r'\\eta\data\db\styles' , checkpoint_file_name='fns.ckpt'
#	 ):

#	 if type(image) is list :
#		 list_output = []
#		 for item in image:
#			 list_output.append( style(item, style_name,factor, desaturate_input, desaturate_output) )
#		 return list_output

#	 global style_session
#	 global style_graph
#	 global style_preds
#	 global style_placeholder
	
#	 if factor!=1:
#		 image = resize(image, factor)
#	 if desaturate_input:
#		 image = desaturate(image)
#		 image = tf.concat( [image,image,image], axis=-1)

#	 input_image = image.numpy()

#	 style_key = f"{style_name}_{image.shape[1]}_{image.shape[2]}"

#	 #if style_session is None:
#	 if style_key not in style_graph:
#		  style_graph[style_key] = tf.Graph()

#	 if style_key not in style_session:
#		 with style_graph[style_key].as_default():
#			 from thirdparty.fast_style_transfer.src import transform
#			 soft_config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
#			 soft_config.gpu_options.allow_growth = True

#			 style_session[style_key] = tf.compat.v1.Session(config=soft_config)
#			 style_placeholder[style_key] = tf.compat.v1.placeholder(tf.float32, shape=image.shape, name='img_placeholder')
#			 style_preds[style_key] = transform.net(style_placeholder[style_key])
#			 saver = tf.compat.v1.train.Saver()
#			 saver.restore(style_session[style_key], os.path.join( styles_path, style_name, checkpoint_file_name) )

	
#	 output = style_session[style_key].run(style_preds[style_key], feed_dict={style_placeholder[style_key]:input_image})

#	 output = tf.cast( output , tf.float32)
#	 output = (output / 127.5) - 1
#	 output = tf.clip_by_value(output, -1, 1)

#	 if factor!=1:
#		 output = resize(output, 1.0/factor)

#	 if desaturate_output:
#		 output = desaturate(output)
#		 output = tf.concat( [output,output,output], axis=-1)

#	 #output = tf.convert_to_tensor( output.eval() )
#	 debugprint(f'Style output shape: {output.shape}')
#	 #debugprint(f'Style output value: {output}')
#	 #debugprint(f'Going to save output to: {output_path}')
#	 debug.draw_tensor4(output, f'styled_{style_name}' )
#	 #tf.keras.preprocessing.image.save_img( output_path, _preds[0], data_format='channels_last', scale=True)
#	 return output
#	 # for j, path_out in enumerate(1):


#TODO: Try this function, understand if stretching channels same way as resolution sould help.
def stretch_channels(image, channels=6, blackout=True , method='nearest'): #Not nearest give shift in color destribution.

	#input_shape = image.shape
	#debugprint(f'input_shape: {input_shape}')
	if blackout:
		blackout_image = tf.ones_like(image)*-1
		image = tf.concat( [blackout_image, image, blackout_image] , axis = -1)
	
	image = tf.transpose(image, [0, 3, 1, 2])
	image = tf.image.resize(image, (channels, image.shape[2] ), method='nearest', antialias=True )
	image = tf.transpose(image, [0, 2, 3, 1])
	#debugprint(image.shape)

	return image

@tf.function
def color_mask(image, select_color, selection_multiplier = 1, selection_power = 1):
	#debugprint(f'Selection color: shape: {select_color}')
	output = distance(image, select_color)
	output = tf.expand_dims(output, axis=-1)
	output = 1-output
	
	#output = tf.clip_by_value( output , 0.0 ,1.0)
	#if selection_multiplier != 1.0:
	#	output = output*selection_multiplier
	#	output = tf.clip_by_value( output , 0.0 ,1.0)
	#if selection_power != 1.0:
	#	output = tf.pow( (output*selection_multiplier) , selection_power)
	#	output = tf.clip_by_value( output , 0.0 ,1.0)
	return output

def replace_color(image, select_color, replacement_color, color_range=[-1,1] ):
	select_color = tf.convert_to_tensor(select_color) #-0.5)*2
	#select_color = (select_color-0.5)*2 
	select_color = color_like(select_color, image)

	replacement_color = tf.convert_to_tensor(replacement_color)
	#replacement_color = (select_creplacement_colorolor-0.5)*2
	replacement_color = color_like(replacement_color, image)

	#shift_positive(image)
	mask = color_mask(image, select_color)
	#return mask

	color_distance = distance(select_color, replacement_color)
	color_distance = tf.expand_dims(color_distance, axis=-1)

	image_lerp = lerp(image, replacement_color, color_distance)

	output = lerp(image, image_lerp, mask)

	return output

def color_like( color, reference , colorSpace = 'linear'):
	#if colorSpace.lower() == 'srgb':
	#	color = ColorLinear2sRGB(color)

	output = tf.ones_like(reference)
	output = output*color

	return output

#from spectre import waves, spectre_masks , selection_mask_powers
#import copy #required for deep copying of the reference colors for further replacement

#Input must be positive 0~1
def srgb_2_linear(x):
	x = tf.pow( tf.abs(x), 2.2 )
	return x

#Input must be positive 0~1
def linear_2_srgb(x):
	x = tf.pow( tf.abs(x), 1.0/2.2 )
	return x

# #	   |R|   | 0.4124564 0.3575761 0.1804375 |
# # XYZ = |G| x | 0.2126729 0.7151522 0.0721750 |
# #	   |B|   | 0.0193339 0.1191920 0.9503041 |

# #	   |X|   |  0.4002 0.7076 −0.0808 |
# # LMS = |Y| x | −0.2263 1.1653  0.0457 |
# #	   |Z|   |  0.0000 0.0000  0.9180 |

# #	 | 0.31399022 0.63951294 0.04649755 |   
# # T = | 0.15537241 0.75789446 0.08670142 | = XYZ x LMS
# #	 | 0.01775239 0.10944209 0.87256922 |   
# T_XYZLMS = tf.convert_to_tensor( [
#	 [0.31399022, 0.63951294, 0.04649755],
#	 [0.15537241, 0.75789446, 0.08670142],
#	 [0.01775239, 0.10944209, 0.87256922],
# ])

# # LMS space to LinearRGB
# #			|  5.47221206 -4.64196010  0.16963708 |
# # Tinverse = | -1.12524190  2.29317094 -0.16789520 |
# #			|  0.02980165 -0.19318073  1.16364789 |
# Tinverse = tf.convert_to_tensor( [
#	 [5.47221206, -4.64196010,  0.16963708],
#	 [-1.12524190,  2.29317094, -0.16789520],
#	 [0.02980165, -0.19318073,  1.16364789],
# ])

# #Missing Long waves # missing REd ?
# Protanopia = tf.convert_to_tensor( [
#	 [0.00000000, 1.05118294, -0.05116099],
#	 [0.00000000, 1.00000000,  0.00000000],
#	 [0.00000000, 0.00000000,  1.00000000],
# ])

# #	 | 1.00000000 0.00000000 0.00000000 |
# # D = | 0.95130920 0.00000000 0.04866992 |
# #	 | 0.00000000 0.00000000 1.00000000 |
# #missing the Middle cone #Missing green
# Deuteranopia = tf.convert_to_tensor( [
#	 [1.00000000, 0.00000000, 0.00000000],
#	 [0.95130920, 0.00000000, 0.04866992],
#	 [0.00000000, 0.00000000, 1.00000000],
# ])

# #	 |  1.00000000 0.00000000 0.00000000 |
# # T = |  0.00000000 1.00000000 0.00000000 |
# #	 | -0.86744736 1.86727089 0.00000000 |
# Tritanopia = tf.convert_to_tensor( [
#	 [1.00000000, 0.00000000, 0.00000000],
#	 [0.00000000, 1.00000000, 0.00000000],
#	 [-0.86744736, 1.86727089, 0.00000000],
# ])

# NormalVision = tf.convert_to_tensor( [
#	 [1.00000000, 0.00000000, 0.00000000],
#	 [0.00000000, 1.00000000, 0.00000000],
#	 [0.00000000, 0.00000000, 1.00000000],
# ])

# #		 |R|   | 0.212656 |
# #  Mono = |G| x | 0.715158 |
# #		 |B|   | 0.072186 |
# Monochromacy = tf.convert_to_tensor( [
#	 [0.212656, 0.715158, 0.072186]
# ])

#Input must be positive in range: [ 0 ~ 1 ] and be in linear color space
@tf.function
def rgb_2_lms(image):
	XYZ = tf.convert_to_tensor( [
		[0.4124564, 0.3575761, 0.1804375],
		[0.2126729, 0.7151522, 0.0721750],
		[0.0193339, 0.1191920, 0.9503041],
	])
	LMS = tf.convert_to_tensor( [[
		[0.4002, 0.7076, -0.0808],
		[-0.2263, 1.1653,  0.0457],
		[0.0000, 0.0000,  0.9180],
	]])

	image = tf.matmul(image, XYZ)
	image = tf.matmul(image, LMS)
	return image

#Input must be positive in range: [ 0 ~ 1 ] and be in linear color space
@tf.function
def lms_2_rgb(image):
	XYZ = tf.convert_to_tensor( [
		[0.4124564, 0.3575761, 0.1804375],
		[0.2126729, 0.7151522, 0.0721750],
		[0.0193339, 0.1191920, 0.9503041],
	])
	LMS = tf.convert_to_tensor( [[
		[0.4002, 0.7076, -0.0808],
		[-0.2263, 1.1653,  0.0457],
		[0.0000, 0.0000,  0.9180],
	]])

	image = tf.matmul(image, tf.linalg.inv(LMS) )
	image = tf.matmul(image, tf.linalg.inv(XYZ) )
	return image

#Input must be positive in range: [ 0 ~ 1 ] and be in lms color space
@tf.function
def spectral_shift(image, transform, noise=0.0):
	image_source_channels = image.shape[-1]

	#image = shift_positive(image)
	#image = srgb_2_linear(image)
	#image = rgb_2_lms(image)
	channels = transform.shape[0]
	if image_source_channels != channels :
		image = stretch_channels(image, channels=channels, blackout=False)

	if transform != None:
		image = tf.matmul(image, transform)

	if noise != 0.0:
		image = image + tf.random.normal( image.shape )*noise

	if image_source_channels != channels:
		image = stretch_channels(image, channels=image_source_channels, blackout=False)

	#image = lms_2_rgb(image)
	#image = linear_2_srgb(image)
	#image = shift_negative(image)
	return image


def tiles(image, size=(64,64) , method='list' ):

	count_x = int( image.shape[1]/size[0]  )
	count_y = int( image.shape[2]/size[1] )

	tiles_count = (count_x, count_y)

	adjusted_shape = (size[0]*count_x, size[1]*count_y)

	if adjusted_shape != (image.shape[1], image.shape[2] ):
		image = tf.image.resize(image, adjusted_shape, antialias = True, method='bicubic' )

	debugprint(f'Image size: {image.shape} | required_shape: {adjusted_shape} | Tiles count {tiles_count} ')

	buckets = []
	for x in range(count_x):
		for y in range(count_y):
			x_pos = x*size[0]
			y_pos = y*size[1]
			#index = x_pos*y_pos

			bucket =  tf.image.crop_to_bounding_box( image, x_pos, y_pos, size[0],  size[1])
			buckets.append(bucket)
			pass
	
	if method != 'list':
		buckets = tf.concat(buckets, axis=0 )
	#tile_rows = tf.reshape(image, [image.shape[0], -1, size[1], image.shape[2]])
	#serial_tiles = tf.transpose(tile_rows, [1, 0, 2, 3])
	#return tf.reshape(serial_tiles, [-1, size[1], size[0], image.shape[2]])
	return buckets

#@tf.function
#def compose_tile(tile, backdrop, x_offset, y_offset):
#	ones = tf.ones( ( tile.shape[0], tile.shape[1], tile.shape[2], 1 ), dtype=tf.bool )
#	tile_mask = tf.image.pad_to_bounding_box( ones, x_offset, y_offset, backdrop.shape[1],  backdrop.shape[2] )
#	padded_tile = tf.image.pad_to_bounding_box( tile, x_offset, y_offset, backdrop.shape[1],  backdrop.shape[2] )
#	return tf.where( tf.equal(tile_mask, tf.constant(True) ) , padded_tile , backdrop)

#@tf.function
def compose_tile(tile, backdrop, x_offset, y_offset):
	ones = tf.ones( ( tile.shape[0], tile.shape[1], tile.shape[2], 1 ), dtype=tf.bool )
	#ones = tf.constant( True, dtype=tf.bool, shape=( tile.shape[0], tile.shape[1], tile.shape[2], 1 ) )
	return tf.where( tf.equal(
		tf.image.pad_to_bounding_box( ones, x_offset, y_offset, backdrop.shape[1], backdrop.shape[2] ) #tile_mask
		, tf.constant(True)) ,  # condition
		tf.image.pad_to_bounding_box( tile, x_offset, y_offset, backdrop.shape[1], backdrop.shape[2] ), #padded_tile
		backdrop)

def from_tiles(tiles, shape):
	
	tile_shape = tiles[0].shape

	count_x = int( shape[1]/tile_shape[1] )
	count_y = int( shape[2]/tile_shape[2] )
	#adjusted_shape = (tile_shape[0]*count_x, tile_shape[1]*count_y)

	output = tf.zeros(shape)
	count = len(tiles)
	#replacement = tf.constant(True)
	#ones = tf.ones( ( tile_shape[0], tile_shape[1], tile_shape[2], 1 ), dtype=tf.bool )
	indicies = [0] * count
	debugprint(f'Merging tiles with shape: {tile_shape}: {count} | ({count_x},{count_y}) | ')
	for x in range(count_x):
		x_offset = tile_shape[1]*x
		for y in range(count_y):
			y_offset = tile_shape[2]*y
			index = y + (x * count_y)
			indicies[index] = ( x_offset, y_offset )
			#index = y + (x * count_y)
			#output = compose_tile(tiles[index], output, x_offset, y_offset)

	for index in range( int(count/2) ):
		forward_index = index
		oposite = count-1-index
		#Move in oposite directions to combine the object faster.
		output = compose_tile(tiles[index], output, indicies[index][0], indicies[index][1] )
		output = compose_tile(tiles[oposite], output, indicies[oposite][0], indicies[oposite][1] )

	# for x in range(count_x):
	#	 x_offset = tf.constant( tile_shape[1]*x , dtype=tf.int32 )
	#	 for y in range(count_y):
	#		 y_offset =  tf.constant( tile_shape[2]*y , dtype=tf.int32 )
	#		 index = y + (x * count_y)
	#		 output = compose_tile(tiles[index], output, x_offset, y_offset)

	#debugprint(f'output shape: {output.shape}')
	return output


def drop(tensor, value=-1, rate=0.2):
	mask = tf.ones( shape=(tensor.shape[0], tensor.shape[1], tensor.shape[2], 1 ) , dtype=tf.dtypes.float32 )*2
	mask = tf.nn.dropout(mask, rate = rate)
	mask = tf.repeat( mask, tensor.shape[-1] , axis=-1, name=None)

	condition = tf.equal(mask, 0)
	mask = tf.where(condition, 0.0, 1.0)
	tensor = tf.where( condition , value, tensor)

	#mask = lerp(empty_data, 1.0, mask)
	mask = tf.clip_by_value(mask, 0.0, 1.0)
	return tensor, mask


if __name__ == '__main__':
	
	#input_images = load_folder(r'\\eta\data\db\visualization\architecture\\', max_count=-1,
	#shuffle=False, size=(1024,1024),
	#filters = {'content':'garbage'}
	#)
	print('IOL Image: functional library to work on images in python. Keep it simple, - opening images and folders with images. But convinient and multithreaded for fast access.')
	pass

#Bucket rendering test
if __name__ == '__bucket_rendering__':
	input_images = load_folder(r'./input', max_count=10, shuffle=False, size=(1024,1024), method='fit', adjust_size=(128,128) )

	method = 'shuffle_tiles'
	method = 'sequential'
	#infrared_masks = copy.deepcopy(spectre_masks)
	while True:
		#dropout_rate = 
		#drop_depth = 

		for n, input_image in enumerate(input_images):

			image = input_image
			bucket_size = (128,128)
			for y in range( int(debug.param(f'image_passes' , value=2 , ui=[1, 30,1] )) ):
				image_tiles = tiles( image, size=bucket_size, method='list' )
				
				indicies = []
				for x in range(len(image_tiles)):
					indicies.append(x)
				
				if method == 'shuffle_tiles':
					random.shuffle(indicies)

				for x in indicies:
					for d in range( int(debug.param(f'bucket_iterations' , value=2 , ui=[1, 30,1] )) ):
						image_tiles[x] , _ = drop(image_tiles[x], rate=float( debug.param(f'dropout_rate' , value=0.4 , ui=[0.0, 0.8, 0.001] ) ) )
						#image_tiles[x] = tile
					
					start_time = time.time()
					image = from_tiles(image_tiles, input_image.shape)
					debugprint(f'compose time: {time.time()-start_time}')
					debug.draw_tensor4( image, f'input_image_{n}' )

#During training, the preprocessor ? No, there is no need in preprocessor, the network is going to drop part of content on input by default, and redraw it.
# So we feed to the image, nice good image, than the network is trying to redraw it by pixels with several attemps. 

#Spectral image check
if __name__ == '__spectral_change__':

	#input_images = load_folder(r'\\eta\data\db\visualization\test\\', max_count=10 , method='fit', size=(256,256), adjust_size=(128,128), shuffle=True)
	input_images = load_folder(r'./input', max_count=10, exclude=['skip'], shuffle=False, size=(1024,1024), method='fit', adjust_size=(128,128) )

	#infrared_masks = copy.deepcopy(spectre_masks)
	while True:

		for n, input_image in enumerate(input_images):

			spectral_saturation = debug.param(f'spectral_saturation' , value=1.0 , ui=[-1, 1, 0.001] )
			spectral_image_darkness = debug.param(f'spectral_image_darkness' , value=0.0 , ui=[-1, 1, 0.001] )

			debug.param('separator_1',  value=0.0 , ui=[0.0, 1, 1])
			rgb_matrix = tf.convert_to_tensor( [
				debug.param(f'rgb_matrix_1' , value=[1.0 , 0.0, 0.0 ] , ui=[-2.0, 4.0, 0.001] ),
				debug.param(f'rgb_matrix_2' , value=[0.0 , 1.0, 0.0 ] , ui=[-2.0, 4.0, 0.001] ),
				debug.param(f'rgb_matrix_3' , value=[0.0 , 0.0, 1.0 ] , ui=[-2.0, 4.0, 0.001] ),
				])
			
			debug.param('separator_2',  value=0.0 , ui=[0.0, 1, 1])
			lms_matrix = tf.convert_to_tensor( [
				debug.param(f'lms_matrix_1' , value=[1.0 , 0.0, 0.0 ] , ui=[-2.0, 4.0, 0.001] ),
				debug.param(f'lms_matrix_2' , value=[0.0 , 1.0, 0.0 ] , ui=[-2.0, 4.0, 0.001] ),
				debug.param(f'lms_matrix_3' , value=[0.0 , 0.0, 1.0 ] , ui=[-2.0, 4.0, 0.001] ),
				])

			debug.param('separator_3',  value=0.0 , ui=[0.0, 1, 1])
			minus_lms_matrix = tf.convert_to_tensor( [
				debug.param(f'minus_matrix_1' , value=[0.0 , 0.0, 0.0 ] , ui=[-2.0, 4.0, 0.001] ),
				debug.param(f'minus_matrix_2' , value=[0.0 , 0.0, 0.0 ] , ui=[-2.0, 4.0, 0.001] ),
				debug.param(f'minus_matrix_3' , value=[0.0 , 0.0, 0.0 ] , ui=[-2.0, 4.0, 0.001] ),
				])

			green_mask_multiplier = debug.param('green_mask_multiplier',  value=1.0 , ui=[-1.0, 1, 0.01])
			
			hsv_selection = debug.param('hsv_selection',  value=[0.0, 0.0, 0.0], ui=[-2.0, 20.0, 0.01])
			selection_input_saturation = debug.param('selection_input_saturation',  value=1.0, ui=[-10.0, 10.0, 0.01])
			selection_mask_power = debug.param('selection_mask_power',  value=1.0, ui=[-10.0, 10.0, 0.01])

			selection_color_min = debug.param('selection_color_min',  value=0.0 , ui=[-2.0, 2.0, 0.01])
			selection_color_max = debug.param('selection_color_max',  value=1.0 , ui=[-2.0, 2.0, 0.01])
			#green_mask_rb_weight = debug.param('green_mask_rb_weight',  value=1.0 , ui=[-1.0, 10.0, 0.01])

			hsv_multiplier = debug.param('hsv_multiplier',  value=[1.0, 1.0, 1.0], ui=[-2.0, 20.0, 0.01])

			shift_depth = debug.param(f'shift_depth' , value=1 , ui=[1, 10, 1] )
			
			spectral_image = input_image
			#minus_image = input_image
			for x in range( shift_depth ):

				if x == 0:
					#green_mask = shift_positive(spectral_image)
					#r,g,b = tf.split(spectral_image, 3 , axis=-1 )
					hsv_image = desaturate(spectral_image, selection_input_saturation)
					hsv_image = tf.image.rgb_to_hsv(shift_positive(hsv_image))
					#hsv_image = tf.pow(hsv_image, hsv_power)

					hm, sm, vm = tf.split(hsv_multiplier, 3, axis=0)
					h,s,v = tf.split(hsv_image, 3, axis=-1)
					h = h * hm
					s = s * sm
					v = v * vm

					hsv_image = tf.concat( [h,s,v] , axis=-1 )

					selection_mask = color_mask(hsv_image, hsv_selection )
					selection_mask = tf.clip_by_value(selection_mask, 0.0 , 1.0)
					selection_mask = tf.pow(selection_mask, selection_mask_power)
					selection_mask = tf.clip_by_value(selection_mask, 0.0 , 1.0)
					selection_mask = lerp(selection_color_min, selection_color_max , selection_mask)
					
					#green_mask = (g - green_mask_rb_weight*b - green_mask_rb_weight*r)
					#green_mask = shift_positive(green_mask)
					#green_mask = tf.concat( [green_mask,green_mask,green_mask], axis=-1)
					#green_mask = tf.pow(green_mask, green_mask_power)
					#green_mask = green_mask * green_mask_multiplier
					#green_mask = tf.clip_by_value( green_mask , 0, 1)
					#green_mask = lerp(dark_green_power, light_green_power, green_mask)
					debug.draw_tensor4( shift_negative(selection_mask), f'selection_mask_{n}' )

					#spectral_image = shift_positive(spectral_image)
					#spectral_image = tf.pow( tf.clip_by_value( spectral_image , 0, 99) , green_mask)
					#spectral_image = shift_negative(spectral_image)

				vision_image = shift_positive(spectral_image)
				vision_image = rgb_2_lms(vision_image)
				vision_image = spectral_shift(vision_image, lms_matrix)
				vision_image = lms_2_rgb(vision_image)
				vision_image = shift_negative(vision_image)

				#plus_image = spectral_shift(spectral_image, plus_lms_matrix)
				minus_image = shift_positive(spectral_image)
				minus_image = rgb_2_lms(minus_image)
				minus_image = spectral_shift(minus_image, minus_lms_matrix)
				minus_image = lms_2_rgb(minus_image)
				#minus_image = shift_negative(minus_image)
				debug.draw_tensor4( shift_negative(minus_image), f'minus_image_{n}' )

				spectral_image = vision_image - minus_image
				spectral_image = shift_positive(spectral_image)
				spectral_image = tf.matmul(spectral_image, rgb_matrix)
				spectral_image = shift_negative(spectral_image)
				spectral_image = lerp( spectral_image , -1.0 , spectral_image_darkness)
			
			#spectral_image = tf.image.per_image_standardization(spectral_image)
			#spectral_image = shift_negative(spectral_image)
			
			#spectral_image = vision_image - minus_image# - minus_image # + plus_image

			spectral_image = desaturate( spectral_image, spectral_saturation )

			debug.draw_tensor4( input_image, f'input_{n}' )
			#debug.draw_tensor4( spectral_image , f'spectral_{n}' )
			
			r,g,b = tf.split(input_image, 3 , axis=-1 )
			ir, ig, ib = tf.split(spectral_image, 3 , axis=-1 )

			out_r = lerp(ir, r, debug.param(f'red_pos' , value=0.0, ui=[0.0, 1.0, 0.001] ) )
			out_g = lerp(ig, g, debug.param(f'green_pos' , value=0.0, ui=[0.0, 1.0, 0.001] ) )
			out_b = lerp(ib, b, debug.param(f'blue_pos' , value=0.0, ui=[0.0, 1.0, 0.001] ) )

			restored_image = tf.concat( [out_r, out_g, out_b] , axis=-1 )
			debug.draw_tensor4( restored_image, f'restored_image_{n}' )
	exit()