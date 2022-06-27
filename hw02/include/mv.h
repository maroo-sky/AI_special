#ifndef CUDAPRACTICE_MV_H
#define CUDAPRACTICE_MV_H

#include "tensor.h"

void mat_vec_reference(Mat& mat, Vec& vec, Vec& result, float& time);

void mat_vec_cuda(Mat& mat, Vec& vec, Vec& result, int tile_length, float& time);

extern "C" {

__global__ void kernel_mat_vec(const float* a, const float* b, float* result,
                               int64_t h, int64_t w);

}

#endif //CUDAPRACTICE_MV_H
