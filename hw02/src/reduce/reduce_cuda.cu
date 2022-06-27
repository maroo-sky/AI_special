#include "reduce.h"

extern "C" {

__global__ void kernel_reduce_sum(const float* a, float* result, int num_elements) {
    // TODO implement
	
	extern __shared__ int data[];
	unsigned int tot = threadIdx.x;
	unsigned int i = blockIdx.x * (blockDim.x*2) + threadIdx.x;
	data[tot] = a[i] + a[i + blockDim.x];
	__syncthreads();

	for (unsigned int s = blockDim.x/2; s>0; s>>=1){

		if (tot < s) {
			data[idx] += data[idx + s];
		}
		__syncthreads();
	}

	if (tot == 0) result[blockIdx.x] = data[0];

}

}
