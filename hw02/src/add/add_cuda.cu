#include "add.h"

extern "C" {

__global__ void kernel_vec_add(const float* a, const float* b, float* result, int num_elements) {

    unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < num_elements) {
        result[i] = a[i] + b[i];
    }
}

}