#ifndef CUDAPRACTICE_REDUCE_H
#define CUDAPRACTICE_REDUCE_H

#include "tensor.h"

void reduce_sum_reference(Vec& vec, float& result, float& time);

void reduce_sum_cuda(Vec& vec, float& result, int threads_per_block, float& time);

extern "C" {

__global__ void kernel_reduce_sum(const float* a, float* result, int num_elements);

}

#endif //CUDAPRACTICE_REDUCE_H
