import signal
import time
import threading
import numpy
import random
irq = True
def signal_handler(sig, frame):
	global irq
	print('You pressed Ctrl+C!')
	irq = False
	# exit()
	return

x_out = []
y_out = []

import matplotlib.pyplot as plt

def make_data():
	global irq
	global input_data_f
	global output_data_f
	global x_out
	global y_out

	it = 0
	while irq:
		numpy.random.seed(random.randint(0, 6553600) + it) 
		x = numpy.random.uniform( -10.0, 10.0, size=size_x)
		y = numpy.sin(x)
		# print(x, y)
		x_out.append(x[0])
		y_out.append(y[0])
		it = it+1
	print("Exiting...")
	
if __name__ == "__main__":
	signal.signal(signal.SIGINT, signal_handler)

	file_index = 0
	size_x = 1
	size_y = 1

	signal.signal(signal.SIGINT, signal_handler)

	import socket
	print(f'Starting output calculation...')

	for n in range(40):
		thread = threading.Thread(None, make_data, f"make_data_{n:2d}", () )
		thread.start()

	plt.show()
	prev_len = 0
	it = 0
	while irq:
		print(len(x_out))
		
		try:
			if it%10 == 0:
				plt.cla()
				plt.clf()
			plt.scatter(x_out[prev_len:], y_out[prev_len:(len(x_out))], s=0.1)
			plt.pause(0.05)
			print("Plot updated")
			prev_len = len(x_out)
		except:
			print("Plot update failed")
			pass
		time.sleep(0.1)
		plt.pause(0.05)
		it = it + 1

	print("Waiting.", end="")
	for n in range(1000):
		print(".", end="")
	print("\nSaving data...", end="\n")

	input_data_f = open(f'./sin_function_x.bin', 'wb')
	output_data_f = open(f'./sin_function_y.bin', 'wb')

	numpy.array(x_out).astype('float32').tofile(input_data_f, '', '<f')
	numpy.array(y_out).astype('float32').tofile(output_data_f, '', '<f')
	
	input_data_f.close()
	output_data_f.close()

	print("Exiting...")