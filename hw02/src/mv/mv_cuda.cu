#include "mv.h"

extern "C" {

__global__ void kernel_mat_vec(const float* a, const float* b, float* result,
                               int64_t h, int64_t w) {
    // TODO implement
	// a: (h,w)
	// b: (w)

	unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
	
	if (col < w) {
		float* a_ptr = (float*) a + col * w;
		float* b_ptr = (float*) b + col;

		float sum = 0.f;
		for (int i = 0; i < w; i++) {
			sum += (*a_ptr++) * (b*ptr++);
		}
		result[col] = sum;
	}

}

}
