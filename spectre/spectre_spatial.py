# Merovingen Solver Library
# Author: Dmitry Prozorovsky.
# IOL Office Creative Union.
# MMXXIII

import time
import random
import tensorflow as tf

# import spectre_error
err_max_vec3 = [
	0, 0, 0, 2, 0, 0, 3, 0, 2, 0, 0, 3, 0, 2, 0, 0, 3, 0, 0, 3, 0, 2, 0, 0, 3, 0, 2, 0, 0, 3, 0, 2, 0, 0, 2, 0, 0, 3, 0, 2, 0, 0, 3, 0, 2, 0, 0, 3, 0, 2, 0, 0, 2, 0, 0, 3, 0, 2, 0, 0, 3, 0, 2, 0, 0, 2, 0, 2, 0, 0, 2, 0, 0, 3, 0, 2, 0, 0, 3, 0, 2, 0, 0, 2, 0, 2, 0, 0, 2, 0, 0, 3, 0, 2, 0, 0, 2, 0, 2, 0, 0, 2, 0, 0, 3, 0, 2, 0, 0, 2, 0, 2, 0, 0, 2, 0, 2, 0, 0, 2, 0, 0, 3, 0, 2, 0, 0, 2, 0, 2, 0, 0, 2, 0, 2, 0, 0, 2, 0, 0, 2, 0, 2, 0, 0, 2, 0, 2, 0, 0, 2, 0, 2, 0, 0, 2, 0, 0, 2, 0, 2, 0, 0, 2, 0, 2, 0, 0, 2, 0, 1, 0, 0, 2, 0, 0, 2, 0, 2, 0, 0, 2, 0, 1, 0, 0, 2, 0, 0, 2, 0, 2, 0, 0, 2, 0, 2, 0, 0, 2, 0, 1, 0, 0, 2, 0, 0, 2, 0, 2, 0, 0, 2, 0, 1, 0, 0, 2, 0, 1, 0, 0, 2, 0, 0, 2, 0, 2, 0, 0, 2, 0, 1, 0, 0, 2, 0, 1, 0, 0, 2, 0, 0, 2, 0, 1, 0, 0, 2, 0, 1, 0, 0, 2, 0, 1
]

err_max_vec2 = [
	0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
]

err_label = tf.constant([x for x in range(256)], dtype=tf.int32)
err_max_vec3 = tf.constant(err_max_vec3, dtype=tf.int32)
err_table_vec3 = tf.lookup.StaticHashTable(
	tf.lookup.KeyValueTensorInitializer(err_label, err_max_vec3, key_dtype=tf.int32, value_dtype=tf.int32), 0
)

# dtype = tf.float64
dtype = tf.float32
# dtype = tf.float16

one = tf.constant( 1.0, dtype=dtype)
half = tf.constant( 0.5, dtype=dtype)
two = tf.constant( 2.0 , dtype=dtype)
three = tf.constant( 3.0 , dtype=dtype)
# drange = tf.constant( 3.14159 , dtype=dtype)

max_value_vec2 = tf.constant( 999.0, dtype=dtype)
max_value_vec3 = tf.constant( 99.0, dtype=dtype) # Input is expected to be uint vector of 3 values.
max_value_vec4 = tf.constant( 99.0, dtype=dtype) # Input is expected to be float in range [-1.0, 1.0]

bytes_count = tf.constant(256.0 , dtype=dtype)
byte_value = tf.constant(255.0 , dtype=dtype)
half_byte = tf.math.multiply( byte_value, half )

minus_mask_vec2 = tf.constant(1001, dtype=dtype)
minus_mask_vec3 = tf.constant(10101, dtype=dtype)
minus_mask_vec4 = tf.constant(1010101,  dtype=dtype)

vec2_mult = tf.constant([ 1000.0, 1.0], dtype=dtype)
vec3_mult = tf.constant([ 10000, 100.0, 1.0], dtype=dtype)

ratio_vec2 = tf.math.truediv(max_value_vec2, bytes_count)
ratio_vec3 = tf.math.truediv(max_value_vec3, bytes_count)
ratio_vec4 = tf.math.truediv(max_value_vec4, bytes_count)

inv_half_byte = tf.math.truediv( one, half_byte)

value_vec2 = tf.constant( 999999.0, dtype=dtype)
value_vec3 = tf.constant( 999999.0, dtype=dtype)
value_vec4 = tf.constant( 99999999.0, dtype=dtype)

byte_ratio_vec2 = tf.math.truediv( byte_value, max_value_vec2)
byte_ratio_vec3 = tf.math.truediv( byte_value, max_value_vec3)
byte_ratio_vec4 = tf.math.truediv( byte_value, max_value_vec4)

nrm_ratio_vec2 = tf.math.truediv( two, value_vec2)
nrm_ratio_vec3 = tf.math.truediv( two, value_vec3)
nrm_ratio_vec4 = tf.math.truediv( two, value_vec4)

# Fuses vec3 to one float with 99 items per channel.
def vec2_to_float(x):
	#vec = x
	x = tf.convert_to_tensor(x, dtype=dtype)
	x = tf.multiply(x, ratio_vec2)
	
	x = tf.math.ceil(x + one) * vec2_mult

	x = tf.math.reduce_sum(x, axis=-1)
	x = x - minus_mask_vec2
	x = ( x * nrm_ratio_vec2 ) - one

	return x

# Fuses vec3 to one float with 99 items per channel.
# Expects input 0 ~ 255 amd output -1.0 ~ 1.0
def vec3_to_float(x):
	x = tf.convert_to_tensor(x, dtype=dtype)
	x = tf.multiply(x, ratio_vec3)
	
	x = tf.math.ceil(x + one) * vec3_mult

	x = tf.math.reduce_sum(x, axis=-1)
	x = x - minus_mask_vec3

	x = ( x * nrm_ratio_vec3 ) - one
	return x#, compound

# Function expects input in range of uint8 
def uint_error_fix(x):
	# L1 Error fix
	noise = tf.random.uniform( x.shape, minval = 0.0, maxval = 1.0, dtype=tf.float16 )
	mask = tf.cast(
		err_table_vec3.lookup(tf.cast(x, tf.int32 ), err_max_vec3 ),
		dtype=tf.float16)

	noise = noise * mask
	noise = tf.math.ceil(noise)
	noise = tf.cast(noise, x.dtype)
	#print(noise)
	x = x - noise  # L1 error has a negative value

	return x

def float_to_vec2(x, err_fix=True, cast=False):
	x = (x + one) * half
	x = tf.math.ceil( x * value_vec2 )
	r_f = tf.math.floor(x * tf.constant(0.001, dtype=dtype) )
	g_f = x
	
	r = r_f
	g = g_f - (r_f * tf.constant(1000.0, dtype=dtype) )

	x = tf.stack([r, g], axis=-1) * byte_ratio_vec2

	if cast:
		x = tf.math.ceil(x) ## Works faster and total error is lower. 
		x = tf.cast(x, tf.int32)

	return x

def float_to_vec3(x, cast=True, err_fix=True):
	x = (x + one) * half
	x = tf.math.round( x * value_vec3 )
	
	r_f = tf.math.floor(x * tf.constant(0.0001, dtype=dtype) )
	g_f = tf.math.floor(x * tf.constant(0.01, dtype=dtype) )
	b_f = x
	
	r = r_f
	g = g_f - (r_f * tf.constant(100.0, dtype=dtype) )
	b = b_f - (r_f * tf.constant(10000.0, dtype=dtype) ) - (g * tf.constant(100.0, dtype=dtype) )

	x = tf.stack([r, g, b], axis=-1) * byte_ratio_vec3

	if cast:
		x = tf.math.ceil(x) ## Works faster and total error is lower. 
		x = tf.cast(x, tf.int32)

	if err_fix:
		x = uint_error_fix(x)
	
	return x

err_min = []
err_max = []
err_prob = []

for t in range(3):
	err_min.append([])
	err_max.append([])
	err_prob.append([])
	for n in range(256):
		err_min[t].append( tf.convert_to_tensor(0, dtype=tf.int32))
		err_max[t].append( tf.convert_to_tensor(0, dtype=tf.int32))
		err_prob[t].append( tf.convert_to_tensor(0, dtype=tf.int32))

# Convert [-1.0 ~ 1.0 ] to [ 0.0 ~ 255.0]
def float_to_byte(x, cast=False): 
	return ( x + one ) * half_byte

# Convert [ 0.0 ~ 255.0] to [-1.0 ~ 1.0 ] 
# @tf.function
def byte_to_float(x):
	return ( tf.cast( x, dtype ) * inv_half_byte ) - one # Should be inverted half byte

# Main test loop for error informaiton gathering.
def test_loop(x):
	# print("---")
	flt = vec3_to_float(x)
	vec = float_to_vec3(flt)
	
	for t in range(3):
		n = int(vec[t].numpy())
		#print(vec.numpy()[t])
		delta = x.numpy()[t] - vec.numpy()[t]
		err_min[t][n] = tf.math.minimum( err_min[t][n], delta)
		err_max[t][n] = tf.math.maximum( err_max[t][n], delta)
		if err_min[t][n] != 0 or err_max[t][n] != 0:
			err_prob[t][n] = err_prob[t][n] + 1

	print(f"{x.numpy()} -> {flt.numpy()} -> {vec.numpy()}", end="\n")

def log_tensor(key, x):
	print(f"{key}:", x.shape, " min: ", tf.reduce_min(x).numpy(), " max: ", tf.reduce_max(x).numpy(), " mean: ",  tf.reduce_mean(x).numpy() )

def restore_spatial_x(x):
	x = float_to_vec2(x)
	x = tf.transpose(x, (0,1,2,3) )
	print("img_x restore: ", x.shape)
	x = tf.reshape(x, (2048,2048,3))
	print("img_x restore: ", x.shape)

def log_loss(x, y, key=">>"):
	x = tf.cast(x, tf.int32 )
	y = tf.cast(y, tf.int32 )
	x = tf.cast(x, dtype)
	y = tf.cast(y, dtype)

	sqr_err = tf.math.abs(tf.math.square( (x-y) ))
	x_ase = tf.reduce_sum(sqr_err)
	x_mse = tf.reduce_mean(sqr_err)
	x_max_mae = tf.reduce_max(sqr_err)

	abs_err = tf.math.abs( (x-y) )
	x_aae = tf.reduce_sum(abs_err)
	x_mae = tf.reduce_mean(abs_err)
	x_max_mae = tf.reduce_max(abs_err)
	print(f"[byte error] [{key}] ase: {x_ase} | aae: {x_aae} | mse {x_mse} | mae {x_mae} | max_err: {x_max_mae}")

# Reshapes to half size by [-1] axis
def axial_compress(x):
	x = tf.transpose(x, (2,0,1))
	x = tf.reshape(x, (3, x.shape[-2], int(x.shape[-1]/2), 2))
	x = vec2_to_float(x)
	# log_tensor("img_x", img_x)

	x = tf.transpose(x, (1,2,0) )
	# tf.keras.utils.save_img('./nebula.png.img_x.png', x)
	return x

def axial_restore(x, cast=False):
	# print("img_x: ", x.shape)
	x = tf.transpose(x, (2,0,1) )
	#log_tensor("restore_float input: ", x)
	x = float_to_vec2(x, cast=cast)
	#log_tensor("restore_float output: ", x)
	x = tf.transpose(x, (0,1,2,3) )
	#print("img_x restore: ", img_x_restore.shape)
	x = tf.reshape(x, ( x.shape[-4], x.shape[-3], x.shape[-2]*x.shape[-1]))
	#print("img_x restore: ", img_x_restore.shape)
	x = tf.transpose(x, (1,2,0) )
	#print("x restore: ", x_restore.shape)
	
	#print(f"img_x ase: {img_mse} | aae: {img_mae} | mean_mse {tf.reduce_mean(tf.math.square( (img-img_x_restore))) }")
	return x

def restore_xy(x, flip=False):
	if flip: x = tf.transpose(x, (1,0,2) )
	#log_tensor("axial_y restore in", x)
	x = axial_restore( x )
	#log_tensor("axial_y restore out", x)
	if flip: x = tf.transpose(x, (1,0,2) )
	# log_loss(x_ref, x, "Uncomress to X")

	# Restore X
	if not flip: x = tf.transpose(x, (1,0,2) )
	x = byte_to_float( x )
	log_tensor("axial_xy in", x)
	x = axial_restore( x )
	log_tensor("axial_xy out", x)
	if not flip: x = tf.transpose(x, (1,0,2) )
	return x

# Works, don't touch
def spatial_compress(x):
	# log_tensor("spatial compress input:", img)
	# X axis compression
	orig = x
	log_tensor("axial_x in", x)
	x = axial_compress(x)
	log_tensor("axial_x out", x)
	# Restore X # Works...
	tf.keras.utils.save_img(f'./nebula.png.img_x.png', x )
	RESTORE_X = True
	if RESTORE_X:
		log_tensor("axial_x x in", x)
		x_restore = axial_restore(x)
		log_tensor("axial_x x out", x_restore)
		log_loss(orig, x_restore)
		tf.keras.utils.save_img(f'./nebula.png.img_x_restore.png', tf.cast(x_restore, tf.int32) )
	x_ref = float_to_byte(x)

	# exit()
	#log_tensor("img_x", img_x)
	#log_tensor("img_x_restore", x_restore)
	#log_loss(img, x_restore)

	# Y Axis compression
	x = float_to_byte(x)
	x = tf.transpose(x, (1,0,2) ) # Spaw XY
	log_tensor("axial_y compresss", x)
	x = axial_compress(x)
	x = tf.transpose(x, (1,0,2) )
	output = x  # return x # Actual compressed output without restoration
	
	tf.keras.utils.save_img(f'./nebula.png.img_y.png', x )
	log_tensor("axial_y", x)

	# Restore Y Axis
	x = tf.transpose(x, (1,0,2) )
	log_tensor("axial_y restore in", x)
	x = axial_restore( x )
	log_tensor("axial_y restore out", x)
	x = tf.transpose(x, (1,0,2) )
	# log_loss(x_ref, x, "Uncomress to X")

	# Restore X Axis
	xy_restore = byte_to_float( x )
	log_tensor("axial_xy in", xy_restore)
	xy_restore = axial_restore( xy_restore )
	log_tensor("axial_xy out", xy_restore)

	tf.keras.utils.save_img(f'./nebula.png.xy_restore.png', xy_restore )
	log_loss(orig, xy_restore, "xy_restore error")
	
	return output

def spatial_restore(x):
	# log_tensor("spatial compress input:", img)
	# X axis compression
	log_tensor("input", x)
	x = tf.transpose(x, (1,0,2) )

	# Restore X - 
	x = axial_restore(x)
	log_tensor("axial_restore", x)
	x = tf.transpose(x, (1,0,2) )
	tf.keras.utils.save_img(f'./nebula.png.restore_y.png', x )
	
	x = byte_to_float(x)
	log_tensor("axial_restore byte", x)
	# exit()
	# TODO: Fix transpose
	x = axial_restore(x, cast=True)
	log_tensor("axial_restore byte", x)
	tf.keras.utils.save_img(f'./nebula.png.restore_x.png', x )
	return x
	
	# Y Axis compression
	x = tf.transpose(x, (1,0,2) ) # Spaw XY
	x = float_to_byte(y)
	x = axial_compress(y)
	x = tf.transpose(y, (1,0,2) )
	return x

if __name__ == "__main__":
	start = time.perf_counter_ns()

	img = tf.keras.utils.load_img('./nebula.png')
	img = tf.keras.preprocessing.image.img_to_array(img)
	orig_shape = img.shape
	log_tensor("img raw: ", img)
	img = tf.image.resize(img, size=(2048,2048), method="lanczos3") # Bicubic gives best upscale whe upscaling with lanczos3
	img = tf.clip_by_value(img, 0.0, 255.0)
	img_resize = tf.image.resize(tf.image.resize(img, size=(1024,1024), method="nearest"), size=(2048,2048), method="nearest")
	log_loss(img, img_resize, key="nearest loss      ")
	img_resize = tf.image.resize(tf.image.resize(img, size=(1024,1024), method="gaussian"), size=(2048,2048), method="gaussian")
	log_loss(img, img_resize, key="gaussian loss     ")
	img_resize = tf.image.resize(tf.image.resize(img, size=(1024,1024), method="bilinear"), size=(2048,2048), method="bilinear")
	log_loss(img, img_resize, key="bilinear loss     ")
	img_resize = tf.image.resize(tf.image.resize(img, size=(1024,1024), method="mitchellcubic"), size=(2048,2048), method="mitchellcubic")
	log_loss(img, img_resize, key="mitchellcubic loss")
	img_resize = tf.image.resize(tf.image.resize(img, size=(1024,1024), method="bicubic"), size=(2048,2048), method="bicubic")
	log_loss(img, img_resize, key="bicubic loss      ")
	img_resize = tf.image.resize(tf.image.resize(img, size=(1024,1024), method="lanczos5"), size=(2048,2048), method="lanczos5")
	log_loss(img, img_resize, key="lanczos5 loss     ")
	img_bicubic = tf.image.resize(img, size=(1024,1024), method="bicubic")
	img_resize = tf.image.resize(img_bicubic, size=(2048,2048), method="lanczos3")
	log_loss(img, img_resize, key="lanczos3 loss     ")
	# exit()

	img = tf.convert_to_tensor(img)
	img = tf.cast(img, tf.int32) # Required because of resize.
	tf.keras.utils.save_img('./nebula.png.resized.png', img )
	img = tf.cast(img, dtype)
	# log_tensor("img", img)
	#exit()

	img_spat = spatial_compress(img)
	
	tf.keras.utils.save_img(f'./nebula.png.img_spat.png', img_spat )

	flt_img = vec3_to_float( float_to_byte(img_spat) ) # Compress resized image

	log_tensor("[MAX COMPRESSION] flt_img", flt_img)
	tf.keras.utils.save_img('./nebula.png.spectre_compress.png', tf.stack( [float_to_byte(flt_img),flt_img*0,flt_img*0], axis=-1) )

	vec_img = float_to_vec3(flt_img) # Restore image spectre information directly
	vec_img = tf.image.resize(vec_img, ( int(vec_img.shape[0]*1.5), int(vec_img.shape[1]*1.5)) , method="lanczos3")
	vec_img = tf.image.resize(vec_img, (flt_img.shape[0]*2, flt_img.shape[1]*2) , method="bilinear")
	log_tensor("vec_img", vec_img)
	tf.keras.utils.save_img('./nebula.png.spectre_restore.png', vec_img)

	vec_img = tf.cast(vec_img, tf.int32 )
	img = tf.cast(img, tf.int32 )
	
	delta = tf.abs(img - vec_img)*tf.constant( 100, dtype=tf.int32)
	tf.keras.utils.save_img('./nebula.png.delta.png', delta)
	
	log_loss(img, vec_img, "p2_spa           ")
	
	#Direct convertion
	flt_img = vec3_to_float( tf.cast(img, dtype=dtype) )
	vec_img = float_to_vec3(flt_img)
	tf.keras.utils.save_img('./nebula.png.direct.png', vec_img)
	log_loss(img, vec_img, "p2_spe           ")

	ERROR_VEC2 = False
	if ERROR_VEC2:
		err_max = [ 0 for x  in range(256) ]
		err_min = [ 0 for x  in range(256) ]
		for ri in range(256):
			for gi in range(256):
					#x = tf.convert_to_tensor( [ai, bi, gi, ri], dtype=dtype)
					x = tf.convert_to_tensor([ri, gi], dtype=dtype)

					# x = bfloat_to_float(x)
					flt = vec2_to_float(x)
					vec = float_to_vec2(flt)
					
					x = tf.cast(x, vec.dtype)

					delta = x.numpy() - vec.numpy()
					err_min[ri] = tf.math.minimum( err_min[ri], delta[0]).numpy()
					err_max[ri] = tf.math.maximum( err_max[ri], delta[0]).numpy()

					err_min[gi] = tf.math.minimum( err_min[gi], delta[1]).numpy()
					err_max[gi] = tf.math.maximum( err_max[gi], delta[1]).numpy()

					flt_32 = tf.cast(flt, tf.float32)
					print(f"{x.numpy()} -> {flt.numpy()} [{flt_32}] -> {vec.numpy()} [{delta}]", end="\n")
		print(err_max)
		print(err_min)
	# Save test image
	#exit()
	
	# exit()
	ERROR_VEC3 = False
	if ERROR_VEC3:
		it = 0
		for ri in range(32):
			for gi in range(32):
				# Move vec 2 here ?
				for bi in range(32):
					#if it > 10:
					# 	break

					x = tf.convert_to_tensor( [bi, gi, ri], dtype=dtype)
					test_loop(x)
					x = tf.convert_to_tensor( [bi+32, gi+32, ri+32], dtype=dtype)
					test_loop(x)
					x = tf.convert_to_tensor( [bi+64, gi+64, ri+64], dtype=dtype)
					test_loop(x)
					x = tf.convert_to_tensor( [bi+64+32, gi+64+32, ri+64+32], dtype=dtype)
					test_loop(x)
					x = tf.convert_to_tensor( [bi+128, gi+128, ri+128], dtype=dtype)
					test_loop(x)
					x = tf.convert_to_tensor( [bi+128+32, gi+128+32, ri+128+32], dtype=dtype)
					test_loop(x)
					x = tf.convert_to_tensor( [bi+128+64, gi+128+64, ri+128+64], dtype=dtype)
					test_loop(x)
					x = tf.convert_to_tensor( [bi+128+64+32, gi+128+64+32, ri+128+64+32], dtype=dtype)
					test_loop(x)

					it = it + 1
					print(f"it: {it} of {32*32*32}")

		end = time.perf_counter_ns()
		d_time = end - start

		for t in range(3):
			err_min[t] = [ x.numpy() for x in err_min[t] ]
			err_max[t] = [ x.numpy() for x in err_max[t] ]
			err_prob[t] = [ x.numpy() for x in err_prob[t] ]

		print(f"Error min: {err_min}")
		print(f"Error max: {err_max}")
		print(f"Error prob: {err_prob}")