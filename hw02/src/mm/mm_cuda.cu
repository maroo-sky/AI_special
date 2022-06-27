#include "mm.h"

extern "C" {
__global__ void kernel_mat_mul(const float* a, const float* b, float* result,
                               int64_t h, int64_t k, int64_t w) {
    // a:       (h, k) row-wise
    // b:       (k, w) col-wise
    // result:  (h, w) row-wise

    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;

    if ((col < w) && (row < h)) {

        float* a_ptr = (float*) a + row * k;
        float* b_ptr = (float*) b + col * k;

        float sum = 0.f;
        for (int i = 0; i < k; i++) {
            sum += (*a_ptr++) * (*b_ptr++);
        }

        result[row * w + col] = sum;
    }
}
}

template<int block_size>
__device__ void kernel_mat_mul_tile(const float* a, const float* b, float* result,
                                    int64_t h, int64_t k, int64_t w) {
    // assume square sub-matrix [block_size, block_size]

    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;

    unsigned int a_begin_row = block_size * blockIdx.y;
    unsigned int b_begin_col = block_size * blockIdx.x;

    unsigned int ty = threadIdx.y;
    unsigned int tx = threadIdx.x;

    float sum = 0.f;
    for (int depth = 0; depth < k; depth += block_size) {

        // memory is shared
        __shared__ float aSub[block_size][block_size];
        __shared__ float bSub[block_size][block_size];

        // load a single element of matrix
        if ((a_begin_row + ty < h) && (depth + tx < k)) {
            aSub[ty][tx] = a[(a_begin_row + ty) * k + (depth + tx)];
        }
        else {
            aSub[ty][tx] = 0.f;
        }

        if ((b_begin_col + tx < w) && (depth + ty < k)) {
            bSub[ty][tx] = b[(b_begin_col + tx) * k + (depth + ty)];
        }
        else {
            bSub[ty][tx] = 0.f;
        }

        // ensure all elements are loaded in sub-matrix
        __syncthreads();

        // compute
        for (int i = 0; i < block_size; i++) {
            sum += aSub[ty][i] * bSub[i][tx];
        }

        // ensure all computation is done
        __syncthreads();

    }

    if ((col < w) && (row < h)) {
        result[row * w + col] = sum;
    }
}

extern "C" {

__global__ void kernel_mat_mul_tile_b16(const float* a, const float* b, float* result,
                                        int64_t h, int64_t k, int64_t w) {
    kernel_mat_mul_tile<16>(a, b, result, h, k, w);
}

__global__ void kernel_mat_mul_tile_b32(const float* a, const float* b, float* result,
                                        int64_t h, int64_t k, int64_t w) {
    kernel_mat_mul_tile<32>(a, b, result, h, k, w);
}

}