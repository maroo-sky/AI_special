#include "softmax.h"

extern "C" {

__global__ void kernel_softmax(const float* a, float* result, int num_elements) {
    // TODO implement

	unsigned int length = blockIdx.x * blockDim.x + threadIdx.x;
       	
	if (length < num_elements){
		float* a_ptr = (float*) a + num_elements;
	
		float sum = 0.f;
		for (int i = 0; i < length; i++) {
			sum += exp(*a_ptr++);
		}	
		result[length] = exp(a[length]) / sum;
	}	
}

}
