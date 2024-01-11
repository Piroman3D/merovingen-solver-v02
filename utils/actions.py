# Merovingen Solver Library
# Author: Dmitry Prozorovsky.
# IOL Office Creative Union.
# MMXXIII

import tensorflow as tf

def convert_image(model, path, binary=True):
	print(f"Converting image: {path}")
	
	img = tf.keras.utils.load_img(path)
	img = tf.convert_to_tensor(img)
	print(f"Image converted: {img.shape}")
	img = tf.cast(img, tf.float32)
	img = ((img*(1.0/255.0))-0.5)*2.0

	if binary:
		img_shape = img.shape
		img = tf.reshape( img, (-1, 3))

		output = []
		batch = 4096*2

		for it in range( int(img.shape[0]/batch)+1 ):
			#print(f"Running model; {model}")
			start = batch*it
			end = start+batch
			end = min(end, img.shape[0] )

			print(f"{start} of {img.shape[0]}")

			x = img[start:end]
			#x = tf.expand_dims(x , axis=0 )
			output.append( model(x) )
			#print(f"Model tested: {output.shape}")

		output = tf.concat(output, axis=0)
		output = ((output + 1.0)*0.5)*255.0
		output = tf.reshape( output, img_shape )
		#output = tf.cast( output, tf.uint8)
		tf.keras.utils.save_img( f"{path}_edited.png", output)
	return
