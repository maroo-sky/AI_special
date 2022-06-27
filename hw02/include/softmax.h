#ifndef CUDAPRACTICE_SOFTMAX_H
#define CUDAPRACTICE_SOFTMAX_H

#include "tensor.h"

void softmax_reference(Vec& vec, Vec& result, float& time);

void softmax_cuda(Vec& vec, Vec& result, int threads_per_block, float& time);

extern "C" {

__global__ void kernel_softmax(const float* a, float* result, int num_elements);

}


#endif //CUDAPRACTICE_SOFTMAX_H
