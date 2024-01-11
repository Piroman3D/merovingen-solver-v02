# Merovingen Solver Library
# Author: Dmitry Prozorovsky.
# IOL Office Creative Union.
# MMXXIII

import time
import tensorflow as tf

# import spectre_error
#err_max_vec3 = [
# 	0, 0, 0, 2, 0, 0, 3, 0, 2, 0, 0, 3, 0, 2, 0, 0, 3, 0, 0, 3, 0, 2, 0, 0, 3, 0, 2, 0, 0, 3, 0, 2, 0, 0, 2, 0, 0, 3, 0, 2, 0, 0, 3, 0, 2, 0, 0, 3, 0, 2, 0, 0, 2, 0, 0, 3, 0, 2, 0, 0, 3, 0, 2, 0, 0, 2, 0, 2, 0, 0, 2, 0, 0, 3, 0, 2, 0, 0, 3, 0, 2, 0, 0, 2, 0, 2, 0, 0, 2, 0, 0, 3, 0, 2, 0, 0, 2, 0, 2, 0, 0, 2, 0, 0, 3, 0, 2, 0, 0, 2, 0, 2, 0, 0, 2, 0, 2, 0, 0, 2, 0, 0, 3, 0, 2, 0, 0, 2, 0, 2, 0, 0, 2, 0, 2, 0, 0, 2, 0, 0, 2, 0, 2, 0, 0, 2, 0, 2, 0, 0, 2, 0, 2, 0, 0, 2, 0, 0, 2, 0, 2, 0, 0, 2, 0, 2, 0, 0, 2, 0, 1, 0, 0, 2, 0, 0, 2, 0, 2, 0, 0, 2, 0, 1, 0, 0, 2, 0, 0, 2, 0, 2, 0, 0, 2, 0, 2, 0, 0, 2, 0, 1, 0, 0, 2, 0, 0, 2, 0, 2, 0, 0, 2, 0, 1, 0, 0, 2, 0, 1, 0, 0, 2, 0, 0, 2, 0, 2, 0, 0, 2, 0, 1, 0, 0, 2, 0, 1, 0, 0, 2, 0, 0, 2, 0, 1, 0, 0, 2, 0, 1, 0, 0, 2, 0, 1
# ]

# Error min: [
# [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
# [0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1, 0, 0, -1, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1, 0, 0, -1, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
# [0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1, 0, 0, -1, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1, 0, 0, -1, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1, 0, 0, -1, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
# Error max: [
# [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
# [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
# [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0]]
# Error prob: [
# [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
# [0, 0, 0, 0, 3072, 0, 3008, 0, 0, 0, 0, 2350, 672, 0, 0, 0, 1410, 1632, 0, 0, 0, 470, 2592, 0, 3008, 0, 0, 0, 0, 3008, 0, 0, 0, 0, 0, 3072, 0, 0, 0, 0, 3072, 0, 3008, 0, 0, 0, 0, 2162, 864, 0, 0, 0, 1222, 1824, 0, 0, 0, 282, 2784, 0, 3008, 0, 0, 0, 0, 564, 2496, 0, 0, 0, 0, 3072, 0, 0, 0, 0, 3072, 0, 3008, 0, 0, 0, 0, 1974, 1056, 0, 0, 0, 1034, 2016, 0, 3008, 0, 63, 0, 0, 1410, 1632, 0, 0, 0, 470, 2592, 0, 0, 0, 0, 3072, 0, 3008, 0, 0, 0, 0, 2820, 192, 0, 0, 0, 1880, 1152, 0, 0, 0, 940, 2112, 0, 2473, 0, 0, 0, 0, 1222, 1824, 0, 0, 0, 282, 2784, 0, 0, 0, 0, 3072, 0, 3008, 0, 0, 0, 0, 2632, 384, 0, 0, 0, 1692, 1344, 0, 3072, 0, 0, 0, 0, 1974, 1056, 0, 0, 0, 1034, 2016, 0, 0, 0, 94, 2976, 0, 3008, 0, 0, 0, 0, 3008, 0, 0, 0, 0, 2444, 576, 0, 0, 0, 0, 2496, 0, 2820, 192, 0, 0, 0, 1880, 1152, 0, 0, 0, 940, 2112, 0, 0, 0, 0, 3072, 0, 3008, 0, 0, 0, 0, 3008, 0, 0, 0, 0, 0, 1440, 0, 0, 0, 0, 3072, 0, 2632, 384, 0, 0, 0, 1692, 1344, 0, 0, 0, 752, 2304, 0, 3008, 0, 0, 0, 0, 3008, 0, 0, 0, 0, 3008, 0, 0],
# [0, 0, 0, 0, 3072, 0, 1024, 0, 0, 0, 0, 768, 768, 0, 0, 0, 448, 1728, 0, 0, 0, 96, 2784, 0, 1024, 0, 0, 0, 0, 1024, 0, 0, 0, 0, 0, 3072, 0, 0, 0, 0, 3072, 0, 1024, 0, 0, 0, 0, 736, 864, 0, 0, 0, 416, 1824, 0, 0, 0, 64, 2880, 0, 1024, 0, 0, 0, 0, 192, 2496, 0, 0, 0, 0, 3072, 0, 0, 0, 0, 3072, 0, 962, 186, 0, 0, 0, 671, 1059, 0, 0, 0, 352, 2016, 0, 1024, 0, 32, 0, 0, 416, 1824, 0, 0, 0, 96, 2784, 0, 0, 0, 0, 3072, 0, 1024, 0, 0, 0, 0, 896, 384, 0, 0, 0, 576, 1344, 0, 0, 0, 256, 2304, 0, 1728, 0, 0, 0, 0, 384, 1920, 0, 0, 0, 64, 2880, 0, 0, 0, 0, 3072, 0, 1024, 0, 0, 0, 0, 864, 480, 0, 0, 0, 544, 1440, 0, 2976, 0, 0, 0, 0, 608, 1248, 0, 0, 0, 289, 2205, 0, 0, 0, 32, 2976, 0, 1024, 0, 0, 0, 0, 1024, 0, 0, 0, 0, 768, 768, 0, 0, 0, 0, 605, 0, 896, 384, 0, 0, 0, 544, 1440, 0, 0, 0, 224, 2400, 0, 0, 0, 0, 3072, 0, 1024, 0, 0, 0, 0, 1024, 0, 0, 0, 0, 0, 576, 0, 0, 0, 0, 3072, 0, 864, 480, 0, 0, 0, 512, 1536, 0, 0, 0, 192, 2496, 0, 1024, 0, 0, 0, 0, 1024, 0, 0, 0, 0, 1024, 0, 0]]


# err_label = tf.constant([x for x in range(256)], dtype=tf.int32)
# err_max_vec3 = tf.constant(err_max_vec3, dtype=tf.int32)
# err_table_vec3 = tf.lookup.StaticHashTable(
# 	tf.lookup.KeyValueTensorInitializer(err_label, err_max_vec3, key_dtype=tf.int32, value_dtype=tf.int32), 0
# )

# dtype = tf.float64
dtype = tf.float32 # Works, but only with 99 values. 999 values works only with float64.
# dtype = tf.float16 # Almost works... 

one = tf.constant( 1.0, dtype=dtype)
half = tf.constant( 0.5, dtype=dtype)
two = tf.constant( 2.0 , dtype=dtype)

vec3_ratio = tf.constant( 0.38828840, dtype=dtype)		# 1/255.0 * 99.0
vec3_nrm_ratio = tf.constant(1.0/9999.99, dtype=dtype) 	# 1/9999.99 * 255.0
inv_vec3_nrm_ratio = tf.constant(9999.99, dtype=dtype) 	# 1/9999.99 * 255.0
# exit()

# bytes_count = tf.constant(256.0 , dtype=dtype)
byte_value = tf.constant(255.0 , dtype=dtype)
# half_byte = tf.math.multiply( byte_value, half )
h100 = tf.constant(100, dtype=dtype)
inv_h100 = tf.constant(0.01, dtype=dtype)
# inv_half_byte = tf.math.truediv( one, half_byte)
# inv_byte = tf.math.truediv( one, byte_value)

# ratio_vec3 = tf.math.multiply(inv_byte, max_value_vec3) # *max_value_vec3
value_vec3 = tf.constant( 9999.99, dtype=dtype) # fractional part can be from 0.0 to 99999999.... 
vec3_out_nrm = tf.constant( [1.0/(99.02344/255.0), 1.0/(99.9999/255.0), 1.0/(99.9999/255.0)], dtype=dtype )
# value_vec3_half = tf.constant( value_vec3 * half, dtype=dtype) # fractional part can be from 0.0 to 99999999.... 
# byte_ratio_vec3 = tf.math.truediv( byte_value, max_value_vec3)
# nrm_ratio_vec3 = tf.math.truediv( two, value_vec3)

# Expects input 0 ~ 255 amd output -1.0 ~ 1.0
def vec3_to_float(x):
	x = tf.multiply( tf.add(x, 0.009) , vec3_ratio)

	# log_tensor("INPUT: ", x)
	r,g,b = tf.split(x, 3, axis=-1)
	
	b = tf.multiply( tf.math.round(b), h100 )
	g = tf.math.round(g)
	r = tf.multiply( r , inv_h100 )

	# log_tensor("R: ", r)
	# log_tensor("G: ", g)
	# log_tensor("B: ", b)

	x = tf.concat( [ r, g, b] , axis=-1)
	x = tf.reduce_sum(x, axis=-1) # + one
	
	# log_tensor("X summ", x)
	x = tf.multiply(x , vec3_nrm_ratio)
	# x = (x - 0.49999983)*2.0000008
	# x = x - 0.0004
	# log_tensor("X nrm", x)
	# print("vec: ", x.numpy(), " -> ", vec.numpy() , " -> ", out_vec.numpy() , end="")
	return x

# Functions expects x in range [-1.0 ~ 1.0 ].
# @tf.function
def float_to_vec3(x):
	# TODO: DO NOT DELETE
	# x = ((x/2.0+0.5))*1235.9995 # # -1.0 1.0 range. input/
	# x = (x/2.0000008 + 0.49999983)/vec3_nrm_ratio

	log_tensor(" -- Unfold input:\n", x)
	x = tf.multiply( x, inv_vec3_nrm_ratio) #vec3_nrm_ratio
	log_tensor(" -- Unfold norm:\n", x)

	x_b = tf.multiply(x, inv_h100)
	x_g = (tf.math.mod(x_b, one) * h100)
	x_r = (tf.math.mod(x, one) * h100)

	log_tensor("xR: ", x_r)
	log_tensor("xG: ", x_g)
	log_tensor("xB: ", x_b)

	x = tf.stack( [ x_r, x_g, x_b,] , axis=-1)
	x = tf.multiply(x, vec3_out_nrm )

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

# Main test loop for error informaiton gathering.
def test_loop(x):
	flt = vec3_to_float(x)
	vec = float_to_vec3(flt)
	
	# print(f"{x.numpy()} -> {flt.numpy()} -> {vec.numpy()}", end="\n")
	for t in range(3):
		n = int(vec[t].numpy())
		# print(t, " -> ",vec.numpy()[t])
		delta = x.numpy()[t] - vec.numpy()[t]
		err_min[t][n] = tf.math.minimum( err_min[t][n], delta)
		err_max[t][n] = tf.math.maximum( err_max[t][n], delta)
		if err_min[t][n] != 0 or err_max[t][n] != 0:
			err_prob[t][n] = err_prob[t][n] + 1

def log_tensor(key, x):
	print(f"{key:12s}:", x.shape, " min: ", tf.reduce_min(x).numpy(), " max: ", tf.reduce_max(x).numpy(), " mean: ",  tf.reduce_mean(x).numpy())
	pass

def log_loss(x, y, key=">>"):

	sqr_err = tf.math.abs(tf.math.square( (x-y) ))
	x_ase = tf.reduce_sum(sqr_err)
	x_mse = tf.reduce_mean(sqr_err)
	x_max_mae = tf.reduce_max(sqr_err)

	abs_err = tf.math.abs( (x-y) )
	x_aae = tf.reduce_sum(abs_err)
	x_mae = tf.reduce_mean(abs_err)
	x_max_mae = tf.reduce_max(abs_err)
	print(f"[byte error] [{key}] \n	acc se: {x_ase}\n	acc ae: {x_aae}\n	mse {x_mse}\n	mae {x_mae}\n	max_err: {x_max_mae}")
	return sqr_err

def convert_image(path):
	img = tf.keras.utils.load_img(path)
	img = tf.keras.preprocessing.image.img_to_array(img)
	orig_shape = img.shape
	log_tensor( "pixel: ", tf.convert_to_tensor(img) )

	img = tf.convert_to_tensor(img)
	img = tf.cast(img, dtype)
	
	# log_tensor("img [-1 ~ 1] : ", img)
	start = time.perf_counter_ns()
	# Fold
	flt_img = vec3_to_float( img ) # Compress spectral
	# log_tensor("[spectre] flt_img", flt_img)
	# Unfold
	vec_img = float_to_vec3(flt_img) # Restore image spectre information directly
	print(f"dt: {time.perf_counter_ns()-start}")
	# log_tensor("vec_img", vec_img)
	tf.keras.utils.save_img(f'{path}.spectre_v05_compress.png', tf.stack( [ flt_img, flt_img, flt_img], axis=-1))
	tf.keras.utils.save_img(f'{path}.spectre_v05_restore.png', vec_img)
	log_loss(img, vec_img, "spectral_loss")
	print("\n")

def tf_batch_to_canvas(X, cols: int = None):
	import math
	if len(X.shape.as_list()) > 4:
		raise ValueError("input tensor has more than 4 dimensions.")
	N, H, W, C = X.shape.as_list()
	rc = math.sqrt(N)
	if cols is None:
		rows = cols = math.ceil(rc)
	else:
		cols = max(1, cols)
		rows = math.ceil(N / cols)
	n_gray_tiles = cols * rows - N
	if n_gray_tiles > 0:
		gray_tiles = tf.zeros((n_gray_tiles, H, W, C), X.dtype)
		X = tf.concat([X, gray_tiles], 0)
	image_shape = (H, W)
	n_channels = C
	return image_grid(X, (rows, cols), image_shape, n_channels)

if __name__ == "__main__":
	# convert_image('./birch.png')
	# exit()
	import numpy as np
	img_A = tf.keras.utils.load_img("./birch.png.spectre_v05_compress.png")
	# img_A = tf.keras.preprocessing.image.img_to_array(img_A)
	img_B = tf.keras.utils.load_img("./birch.png")
	# img_B = tf.keras.preprocessing.image.img_to_array(img_B)
	img = np.concatenate((img_A, img_B), axis=1)
	img = tf.image.resize(img, (1024, 1024), preserve_aspect_ratio=True)
	# img = tf.contrib.gan.eval.image_grid([img_A, img_B])
	tf.keras.utils.save_img(f'birch.spectre.png', img)

	exit()
	# exit()
	ERROR_VEC3 = True
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
					print(f"it: {it} of {32*32*32}", end="\r")

		end = time.perf_counter_ns()
		d_time = end - start

		for t in range(3):
			err_min[t] = [ x.numpy() for x in err_min[t] ]
			err_max[t] = [ x.numpy() for x in err_max[t] ]
			err_prob[t] = [ x.numpy() for x in err_prob[t] ]

		print(f"Error min: {err_min}")
		print(f"Error max: {err_max}")
		print(f"Error prob: {err_prob}")
		print(f"exec time: {d_time}")
	exit()