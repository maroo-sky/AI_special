#ifndef CUDAPRACTICE_ADD_H
#define CUDAPRACTICE_ADD_H

#include "tensor.h"

void vec_add_reference(Vec& a, Vec& b, Vec& result, float& time);

void vec_add_cuda(Vec& a, Vec& b, Vec& result, int threads_per_block, float& time);

extern "C" {

__global__ void kernel_vec_add(const float* a, const float* b, float* result, int num_elements);

}

#endif //CUDAPRACTICE_ADD_H
