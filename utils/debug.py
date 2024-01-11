# Merovingen Solver Library
# Author: Dmitry Prozorovsky.
# IOL Office Creative Union.
# MMXXIII

import tensorflow as tf
from PIL import Image
import numpy as np

import base64
from io import BytesIO

import os
import sys
import time

# Add scripts folder to the system path, so they can be imported.
sys.path.insert(0, f'{os.path.normpath( os.path.dirname(os.path.abspath(__file__))) }')

_IOLWSDEBUG_ = False
try:
	import socket
	from websocket import create_connection, WebSocket
	_IOLWSDEBUG_ = True
except:
	class WebSocket:
		pass

	_IOLWSDEBUG_ = False
	print("[ERROR] Failed to import websocket library...")
	print("        install websocket library to run debug in web streaming mode...")

import inspect, re
import datetime
print(f'[IOL Debug library]: Enabled | [WS] {_IOLWSDEBUG_}')

# Linear interpolation
def lerp(x,y,alpha):
	return ( (1.0-alpha) * x) + ((alpha) * y) 

def debug_print(func, time=True):
	def wrapped_func(*args, **kwargs):
		timestr = datetime.datetime.now().strftime('%H:%M:%S')
		return func( f"[{timestr}]", *args, **kwargs)
	return wrapped_func

def debug_print_image(image, text):
	output = base64.b64encode(image)
	result = send_data(text, f"{text}|image|{str(output, encoding='utf8')}")
	return result


sprint = print
print = debug_print(print)
ws = None

def error(*args, **kwargs):
	sprint("\033[91m", end="")
	sprint(*args, **kwargs)
	sprint("\033[0m", end="")

def wait( symbol:str, count=100):
	sprint("\033[91m", end="")
	for n in range(count):
		sprint(symbol, end="")
		time.sleep(0.01)
	sprint("\033[0m", end="\n")

def success(*args, **kwargs):
	sprint("\033[92m", end="")
	sprint(*args, **kwargs)
	sprint("\033[0m", end="")

def warning(*args, **kwargs):
	sprint("\033[93m", end="")
	sprint(*args, **kwargs)
	sprint("\033[0m", end="")

debug_server = "ws://127.0.0.1:7070/"
global_parameters = {}
global_parameters_ui = {}

global_color_space = 'rgb'
def set_global_color_space(color_space=None):
	global global_color_space
	
	if color_space != global_color_space:
		global_color_space = color_space
		print(f'Global color space was set to: {global_color_space}')

global_color_range = (-1,1)
def set_global_color_range(color_range=None):
	global global_color_range
	
	if color_range != global_color_range:
		global_color_range = color_range
		print(f'Global color range was set to: {color_range}')

global_color_linear = True
def set_global_color_linear(linear=None):
	global global_color_linear
	
	if global_color_linear != linear:
		global_color_linear = linear
		print(f'Global color linear was set to: {linear}')

class MerovingenWebSocket(WebSocket):
	def recv_frame(self):
		global global_parameters
		frame = super().recv_frame()
		if frame:
			if str(frame.data, encoding="utf-8").startswith('set_param'):
				#print("~~~"*60)
				#print(f'Debug received data: { str(frame.data, encoding="utf-8")}')
				#print("~~~"*60)
				try:
					data = str(frame.data, encoding="utf-8")
					_ , key, value = data.split('|')
					if key in global_parameters:
						_type = type(global_parameters[key])
						try:
							#print(f'{_type} is trying to parse:{value}')
							#print(f'Parameter_type: {_type}')
							if _type != list:
								global_parameters[key] = _type(value)
							else:
							   # print(f'Parsing list of prameters: {value}')
								values = value.split(',')
								_values_type = type( global_parameters[key][0] )
								#print(f'Parsing list of prameters: {values} with type: {_values_type}')
								for n, item in enumerate(values):
									#print(f'Parsing : {item}')
									global_parameters[key][n] = _values_type(item)
							print(f'DEBUG: [{key}] = {global_parameters[key]}')
						except:
							print(f'{key} [{str(_type)}] failed parsing: {value}')
					else:
						print(f'Received unkown key: {key}')

				except:
					print(f'Failed to parse: {data}')
			else:
				pass
			#	data = str(frame.data, encoding="utf-8")[:100]
			#	print("!!!!!"*60)
			#	print(f'Received data: {data}')
			#	print("!!!!!"*60)
		return frame

max_tries = 3
tries = 0
def send_data(key, data):
	global ws
	global tries
	global max_tries

	if ws is None and tries <= max_tries:
		print('Reconnecting to server...')
		ws = connect_to_server()
		tries = tries + 1
	try: 
		if _IOLWSDEBUG_:
			_result = ws.send(data)
		return True
	except:
		print(f'{key} failed to send data.' )
		ws = None
		return False

	return False

def server_synch_thread():
	global ws
	global global_parameters
	while True:
		if ws != None:
			try:
				ws.recv_frame()
			except:
				pass
				#print(f'Receive data loop broken...')
		else:
			ws = connect_to_server()


error_connection_count = 0
def connect_to_server():
	global ws
	global debug_server
	global global_parameters
	global error_connection_count
	
	if not _IOLWSDEBUG_:
		print("[WARNING] Debug server was skipped...")
		return
	
	try:
		warning(f"DEBUG: Connecting to: {debug_server}")
		#ws = create_connection(debug_server)
		ws = create_connection(
			debug_server,
			sockopt=((socket.IPPROTO_TCP, socket.TCP_NODELAY, 1),),
			class_=MerovingenWebSocket
		)
		success(f"DEBUG SUCCESS: Connection established: {debug_server}")

		import threading
		from threading import Thread

		#Start listenning thread for synching variables edited on the web page
		t = Thread(target=server_synch_thread, args=())
		t.daemon = True
		t.start()
		success(f'DEBUG SUCCESS: Server synch thread started.')
		
		return ws
	except:
		error_connection_count = error_connection_count+1
		error(f"[ERROR]: Make sure that merovingen server is running and websocket is turned on in the settings.")
		print(f"[ERROR]: Failed to connect to cassiopeia debug server: {debug_server}")
		if error_connection_count > 4:
			print(f'Switching debug server to localhost: ws://{socket.gethostname()}:8666/')
			debug_server = f"ws://{socket.gethostname()}:8666/"
	return None

def draw_tensor4(tensor , key="image", auto_color_space=True, color_space=None, print_info=True):

	if tensor == None:
		return None
	#minimum = tf.reduce_min(tensor)
	#maximum = tf.reduce_max(tensor)
	#
	#print(f"Draw 4: {tensor.shape} [ {minimum} ~ {maximum} ]")
	#print(f"Draw 4: {tensor.shape}")
	if tensor.shape[0] > 1:
		split_0 , split_1 = tf.split( tensor , [1, tensor.shape[0]-1], axis=0 )
		tensor = split_0
		#draw_tensor3(split_0, key=f"{key}_012")
		#draw_tensor3(split_1, key=f"{key}_345")
		#return
	
	tensor3 = tf.squeeze(tensor, axis=0)
	return draw_tensor3(tensor3, key=key, auto_color_space=auto_color_space,  color_space=color_space, print_info=print_info)
	
#Actually should be move out from debug...
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

def draw_tensor3(tensor, key="image" , auto_color_space=True , color_space=None, print_info=True):
	global global_color_space
	global global_color_range
	global global_color_linear

	if color_space == None:
		color_space = global_color_space

	if isinstance(tensor, np.ndarray):
		output = Image.fromarray(tensor)

		buffer = BytesIO()
		output.save(buffer, format="PNG")
		output = base64.b64encode(buffer.getvalue() )

		#shape = str(shape)
		#shape = shape.replace(',','x').replace('(','').replace(')','').replace(' ','')
		send_data(key, f"{key}|image|{str(output, encoding='utf8')}")
		return output

	if tensor == None:
		print(f'ERROR: Invalid image tensor for: {key}')
		return None

	shape = tensor.shape
	minimum = tf.reduce_min(tensor)
	maximum = tf.reduce_max(tensor)

	if print_info:
		print(f"[{key}] [{global_color_space}] vec3: {tensor.shape} [ {minimum} ~ {maximum} ]")
	
	if len(tensor.shape) != 3:
		tensor = tf.repeat(tensor, 3, axis=-1)
		print(f'Invalid image tensor shape: {key} : {tensor.shape}')
		return

	if tensor.shape[-1] > 3:
		if global_color_space == 'multispace' and auto_color_space==True:
			rgb, sobel, hsv = tf.split(tensor , [3,3,3] , axis=-1)
			rgb = draw_tensor3(rgb, key=f"{key}_rgb", auto_color_space=True, color_space='rgb',print_info=print_info )
			sobel = draw_tensor3(sobel, key=f"{key}_sobel", auto_color_space=True, color_space='rgb',print_info=print_info )
			hsv = draw_tensor3(hsv, key=f"{key}_hsv", auto_color_space=True, color_space='hsv',print_info=print_info )
			return [rgb, sobel, hsv]
		else:
			split_0 , split_1 = tf.split( tensor , [3, tensor.shape[-1]-3], axis=-1 )
			draw_tensor3(split_0, key=f"{key}_012", auto_color_space=auto_color_space, print_info=print_info)
			draw_tensor3(split_1, key=f"{key}_345", auto_color_space=auto_color_space, print_info=print_info)
		return

	output = tensor - global_color_range[0]
	output = output / ( global_color_range[1]-global_color_range[0] )

	if auto_color_space == True:
		if color_space == 'raw':
			pass
		if color_space == 'hsv':
			output = tf.image.hsv_to_rgb(output)
		if color_space == 'rgb':
			output = output
		if color_space == 'lms':
			output = lms_2_rgb(output)

	if global_color_linear == True and color_space != 'raw':
		output = tf.pow( tf.abs(output), 1.0/2.2 ) #Positive linear space to sRGB conversion.

	return_tensor = output

	if color_space != 'raw':
		output = output*255

	output = tf.clip_by_value( output, 0, 255, name=None)
	output = tf.cast(output , tf.uint8)
	output = tf.image.encode_png(output, compression=-1)
	
	output = base64.b64encode( output.numpy() )
	
	shape = str(shape)
	shape = shape.replace(',','x').replace('(','').replace(')','').replace(' ','')
	
	send_data(key, f"{key}_{shape}|image|{str(output, encoding='utf8')}")
	return return_tensor


def equal(a, b, hint):
	if a != b:
		print(f'Error [{hint}]: {a} != {b}')
		exit()
	return


def draw_vector1(tensor, key="float"):
	if tensor == None:
		return

	isTensor = True
	if type(tensor) == type( int(0) ):
		isTensor = False
		output = tensor
	if type(tensor) == type( float(0.1) ):
		isTensor = False
		output = tensor

	if isTensor:
		output = tensor.numpy()

	return send_data(key , f"{key}|float|{str(output)}") 

#Reduce your tensors to vector1 values to show the values in the chart.
def draw_chart(x, y, key="chart", reduce_methdo='reduce_mean'):
	if x == None or y == None: return

	output_x = x
	if 'ten' in str(type(y)).lower(): # tensor type
		#output_y = tf.reduce_mean(y).numpy()
		output_y = y.numpy()
	else:
		output_y = y

	return send_data( key, f"{key}|chart|{str(output_x)}|{str(output_y)}" )

def draw_3d_point(tensor, key='', color=[255,0,0], t='imposter', split_batch=True):
	#print(f"imposter_shape: {tensor.shape}")
	if split_batch:
		n = 0
		while tensor.shape[0] > 1:
			split_0 , split_1 = tf.split( tensor , [1, tensor.shape[0]-1], axis=0 )
			draw_3d_point( tensor=split_0, key=f"{key}_{n}", color=color, split_batch=False)
			tensor = split_1
			n = n+1

	return send_data( key,  f"{key}|imposter|{color}|{ str(tensor.numpy()) }" )


def draw_parameter_gui(key, value, ui_parameters):
	return send_data(key ,  f"{key}|parameter|{value}|{ui_parameters}")

def set_param(key, value):
	global_parameters[key] = value

def param(key, value=None, ui=None, tensor=True):
	global global_parameters
	global global_parameters_ui

	if key in global_parameters:
		draw_parameter_gui( key=key, value=global_parameters[key], ui_parameters=global_parameters_ui[key])
		output = global_parameters[key]
	else:
		#if all value are provided, we are creating the parameter.
		if key != None and value != None and ui != None:
			global_parameters[key] = value
			global_parameters_ui[key] = ui
			draw_parameter_gui( key=key, value=value, ui_parameters=ui)
			output = value

	if tensor:
		output = tf.convert_to_tensor(output)

	return output

def PrintSummary():
	print("#"*60)
	print(f"Tensorflow version: {tf.__version__}")
	if ws == None:
		status = f'Connection failed...'
	else:
		status = ws.status
	print(f"Debug server: [{status}] {debug_server} ")
	print(f"GPU Name: {tf.test.gpu_device_name()}") 
	
	for x, device in enumerate(tf.config.list_physical_devices()):
		print( f'Devices [{x}]: {device.name}')


if __name__ == '__main__':
	#tf.device('CPU')
	# PrintSummary()
	# ws = connect_to_server()
	# exit()
	pass