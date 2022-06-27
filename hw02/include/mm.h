#ifndef CUDAPRACTICE_MM_H
#define CUDAPRACTICE_MM_H

#include "tensor.h"

void mat_mul_reference(Mat& lhs, Mat& rhs, Mat& result, float& time);

void mat_mul_cuda(Mat& lhs, Mat& rhs, Mat& result, int tile_length, float& time);

extern "C" {

__global__ void kernel_mat_mul(const float* a, const float* b, float* result,
                               int64_t h, int64_t k, int64_t w);

__global__ void kernel_mat_mul_tile_b16(const float* a, const float* b, float* result,
                                        int64_t h, int64_t k, int64_t w);
__global__ void kernel_mat_mul_tile_b32(const float* a, const float* b, float* result,
                                        int64_t h, int64_t k, int64_t w);

}

#endif //CUDAPRACTICE_MM_H
