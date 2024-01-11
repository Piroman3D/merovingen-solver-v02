// # Merovingen Solver Library
// # Author: Dmitry Prozorovsky.
// # IOL Office Creative Union.
// # MMXXIII

#include <stdlib.h>
#include <stdio.h>
#include <inttypes.h>
#include <math.h>

#include "spectre.h"

#define DEBUG

static const float half = 0.5;
static const float one = 1.0;

static const float bytes_count = 256.0;
static const float max_value_vec3 = 99.0;
static const float ratio_vec3 = 0.38671875; // max_value_vec3 / bytes_count
static const float vec3_mult[3] = {10000, 100.0, 1.0 };

static const float minus_mask_vec3 = 10101;
static const float nrm_ratio_vec3 = 2.000002000002e-06; // two / value_vec3 ( float64 division )
// static const float nrm_ratio_vec3 = 2.0000020413135644e-06; // two / value_vec3 ( float32 division )

// float_to_vec3
static const float value_vec3 = 999999.0;
static const float byte_ratio_vec3 = 2.5757575757575757; // byte_value / max_value_vec3

static const uint8_t err_max_vec3[256] = {
	0, 0, 0, 2, 0, 0, 3, 0, 2, 0, 0, 3, 0, 2, 0, 0, 3, 0, 0, 3, 0, 2, 0, 0, 3, 0, 2, 0, 0, 3, 0, 2, 0, 0, 2, 0, 0, 3, 0, 2, 0, 0, 3, 0, 2, 0, 0, 3, 0, 2, 0, 0, 2, 0, 0, 3, 0, 2, 0, 0, 3, 0, 2, 0, 0, 2, 0, 2, 0, 0, 2, 0, 0, 3, 0, 2, 0, 0, 3, 0, 2, 0, 0, 2, 0, 2, 0, 0, 2, 0, 0, 3, 0, 2, 0, 0, 2, 0, 2, 0, 0, 2, 0, 0, 3, 0, 2, 0, 0, 2, 0, 2, 0, 0, 2, 0, 2, 0, 0, 2, 0, 0, 3, 0, 2, 0, 0, 2, 0, 2, 0, 0, 2, 0, 2, 0, 0, 2, 0, 0, 2, 0, 2, 0, 0, 2, 0, 2, 0, 0, 2, 0, 2, 0, 0, 2, 0, 0, 2, 0, 2, 0, 0, 2, 0, 2, 0, 0, 2, 0, 1, 0, 0, 2, 0, 0, 2, 0, 2, 0, 0, 2, 0, 1, 0, 0, 2, 0, 0, 2, 0, 2, 0, 0, 2, 0, 2, 0, 0, 2, 0, 1, 0, 0, 2, 0, 0, 2, 0, 2, 0, 0, 2, 0, 1, 0, 0, 2, 0, 1, 0, 0, 2, 0, 0, 2, 0, 2, 0, 0, 2, 0, 1, 0, 0, 2, 0, 1, 0, 0, 2, 0, 0, 2, 0, 1, 0, 0, 2, 0, 1, 0, 0, 2, 0, 1
};

#define ECC // Enable/Disable error correction for transform.

uint8_t*
float_to_vec3(float* x, size_t size){

	int cnt = size;
	int out_size = cnt*3;

	uint8_t* out = malloc(cnt*sizeof(uint8_t));

	#ifdef DEBUG
		printf("float_to_vec3: malloc: %lu\n", cnt*sizeof(uint8_t) );
	#endif

	while(cnt--){
		int idx = ((out_size-cnt)-1)*3;
		int _i = idx + 0;
		int _j = idx + 1;
		int _k = idx + 2;

		// x = (x + one) * half
		// x = tf.math.round( x * value_vec3 )
		float v = roundf((x[cnt] + one) * half * value_vec3);
		//# r_f = x * tf.constant(0.0001, dtype=dtype)
		//r_f = tf.math.floor(x * tf.constant(0.0001, dtype=dtype) )
		//g_f = tf.math.floor(x * tf.constant(0.01, dtype=dtype) )
		//b_f = x
		
		//float b_f = v;

		//r = r_f
		//g = g_f - (r_f * tf.constant(100.0, dtype=dtype) )
		//b = b_f - (r_f * tf.constant(10000.0, dtype=dtype) ) - (g * tf.constant(100.0, dtype=dtype) )
		//float r = r_f;
		
		float r = floorf(v * (float)0.0001 );
		float g = floorf(v * (float)0.01   ) - (r * (float)100.0 );
		float b = v - 	(r * (float)10000.0) - (g * (float)100.0 );

		out[_i] = (uint8_t)(r * byte_ratio_vec3);
		out[_j] = (uint8_t)(g * byte_ratio_vec3);
		out[_k] = (uint8_t)(b * byte_ratio_vec3);
		#ifdef ECC
			// out[_i] -= rand( err_max_vec3[out[_i]] );
			// out[_j] -= rand( err_max_vec3[out[_i]] );
			// out[_k] -= rand( err_max_vec3[out[_i]] );
		#endif
		printf("[%d] -> [%d, %d, %d] \n", cnt, out[_i], out[_j], out[_k]);
	}
	return out;

	//if cast:
	//	x = tf.math.ceil(x) ## Works faster and total error is lower. 
	//	x = tf.cast(x, tf.int32)
}

// Convert array of floats in byte range [0~255] to compressed array of floats in range [-1.0, 1.0]
void
vec3_to_float(uint8_t* x, size_t size, float** out, int* out_size){
	int cnt = size/3;
	*out_size = cnt;
	*out = malloc(cnt*sizeof(float));
	//float test[10] = {0,1,2,3,4,5,6,7,8,9};
	float* test = malloc(10*sizeof(float));

	#ifdef DEBUG
		printf("vec3_to_float memory allocated [%dx%lu]: %lu\n", cnt, sizeof(float), sizeof(*test));
	#endif

	while(cnt--){
		// printf("vec3_to_uint8: %d | size: %d\n", x[size], size);
		// int index = size - _size;
		int idx = ((*out_size-cnt)-1)*3;
		int _i = idx + 0;
		int _j = idx + 1;
		int _k = idx + 2;

		#ifdef DEBUG
			printf("[%d] uint8_t: [%d, %d, %d]\n " , idx , x[_i], x[_j], x[_k]);
			printf("	[%d] float: [%.f, %.f, %.f] -> " , idx , (float)x[_i], (float)x[_j], (float)x[_k]);
		#endif

		// x = tf.multiply(x, ratio_vec3)
		// x = tf.math.ceil(x + one) * vec3_mult
		// x = tf.math.reduce_sum(x, axis=-1) - minus_mask_vec3
		(*out)[cnt] = 
			(ceilf( ((float)x[_i] * ratio_vec3) + one ) * vec3_mult[0]) +
			(ceilf( ((float)x[_j] * ratio_vec3) + one ) * vec3_mult[1]) +
			(ceilf( ((float)x[_k] * ratio_vec3) + one ) * vec3_mult[2])
		;
		(*out)[cnt] = (*out)[cnt] - minus_mask_vec3;
		
		// # compound = int(vec)
		// # print(f"compound: {compound}")
		
		// x = ( x * nrm_ratio_vec3 ) - one
		(*out)[cnt] = ( (*out)[cnt] * nrm_ratio_vec3 ) - one;
		
		#ifdef DEBUG
			printf(" %f\n", (*out)[cnt]);
		#endif
	}
	// return out;
}

#ifdef DEBUG
int
main(){
	printf("ceil test: %f\n\n", ceil(10.834849));

	// uint8_t x[21] = {13, 42, 93, 4, 18, 127, 123, 94, 87, 45, 255, 232, 9, 87, 97, 9, 87, 97, 9, 87, 97};
	uint8_t x[15] = {13, 42, 93, 4, 18, 127, 123, 94, 87, 45, 255, 232, 9, 87, 97};

	int out_size;
	float* out = NULL;
	vec3_to_float(x, sizeof(x), &out, &out_size);
	printf("\ncompressed values [%d]: \n", out_size );
	
	int cnt = out_size;
	while(cnt--){
		uint8_t bfloat = (uint8_t)(((out[cnt]+1.0)*half)*255.0);
		printf("[%d] -> %f [ %3d ] \n", cnt, out[cnt], bfloat );
	}
	
	printf("\nrestoring values [%d]...\n", out_size );
	uint8_t* restore = float_to_vec3( out, out_size );
	
	return 1;
}

#endif