#include "mm.h"


void mat_mul_cuda(Mat& lhs, Mat& rhs, Mat& result, int tile_length, float& time) {

    // ---------------------------------------------------------------- //

    // check shape
    if ((lhs.cols != rhs.rows) || (lhs.rows != result.rows) || (rhs.cols != result.cols)) {
        printf("[ERROR] CUDA MM shape mismatch (LHS: (%ld, %ld), RHS: (%ld, %ld), RES: (%ld, %ld)\n",
               lhs.rows, lhs.cols, rhs.rows, rhs.cols, result.rows, result.cols);
        return;
    }

    // check align
    if ((lhs.align != MatAlign::ROW_WISE) || (rhs.align != MatAlign::COL_WISE) ||
        (result.align != MatAlign::ROW_WISE)) {
        printf("[ERROR] CUDA MM align mismatch\n");
        return;
    }

    // clear to 0
    result.clear();

    // ---------------------------------------------------------------- //
    cudaError_t err;

    // cuda memory alloc
    int64_t num_elements_lhs = lhs.rows * lhs.cols;
    size_t num_bytes_lhs = num_elements_lhs * sizeof(float);

    float* cu_lhs = nullptr;
    err = cudaMalloc(&cu_lhs, num_bytes_lhs);
    if (err != cudaSuccess) {
        printf("[ERROR] CUDA memory alloc fail\n");
        return;
    }

    int64_t num_elements_rhs = rhs.rows * rhs.cols;
    size_t num_bytes_rhs = num_elements_rhs * sizeof(float);

    float* cu_rhs = nullptr;
    err = cudaMalloc(&cu_rhs, num_bytes_rhs);
    if (err != cudaSuccess) {
        printf("[ERROR] CUDA memory alloc fail\n");
        return;
    }

    int64_t num_elements_result = result.rows * result.cols;
    size_t num_bytes_result = num_elements_result * sizeof(float);

    float* cu_result = nullptr;
    err = cudaMalloc(&cu_result, num_bytes_result);
    if (err != cudaSuccess) {
        printf("[ERROR] CUDA memory alloc fail\n");
        return;
    }

    // ---------------------------------------------------------------- //

    // cuda memory move
    err = cudaMemcpy(cu_lhs, lhs.data, num_bytes_lhs, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        printf("[ERROR] CUDA memory move host to device fail\n");
        return;
    }

    err = cudaMemcpy(cu_rhs, rhs.data, num_bytes_rhs, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        printf("[ERROR] CUDA memory move host to device fail\n");
        return;
    }

    // ---------------------------------------------------------------- //

    dim3 dimBlock(tile_length, tile_length);
    dim3 dimGrid((rhs.cols + tile_length - 1) / tile_length,
                 (lhs.rows + tile_length - 1) / tile_length);

    void* args[] = {&cu_lhs, &cu_rhs, &cu_result, &lhs.rows, &lhs.cols, &rhs.cols};

    auto start_time = get_time();

    // launch
    if (tile_length == 16) {
        err = cudaLaunchKernel((const void*) kernel_mat_mul_tile_b16,
                               dimGrid, dimBlock,
                               args, 0, nullptr);
//        err = cudaLaunchKernel((const void*) kernel_mat_mul,
//                               dimGrid, dimBlock,
//                               args, 0, nullptr);
    }
    else if (tile_length == 32) {
        err = cudaLaunchKernel((const void*) kernel_mat_mul_tile_b32,
                               dimGrid, dimBlock,
                               args, 0, nullptr);
//        err = cudaLaunchKernel((const void*) kernel_mat_mul,
//                               dimGrid, dimBlock,
//                               args, 0, nullptr);
    }
    else {
        err = cudaLaunchKernel((const void*) kernel_mat_mul,
                               dimGrid, dimBlock,
                               args, 0, nullptr);
    }
    if (err != cudaSuccess) {
        printf("[ERROR] CUDA internal error %d\n", (int) err);
        return;
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("[ERROR] CUDA device synchronize fail\n");
        return;
    }

    time += get_duration_us(start_time);

    // ---------------------------------------------------------------- //

    // cuda memory move
    err = cudaMemcpy(result.data, cu_result, num_bytes_result, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        printf("[ERROR] CUDA memory move device to host fail\n");
        return;
    }

    // ---------------------------------------------------------------- //

    // release
    err = cudaFree(cu_lhs);
    if (err != cudaSuccess) {
        printf("[ERROR] CUDA memory free fail\n");
        return;
    }

    err = cudaFree(cu_rhs);
    if (err != cudaSuccess) {
        printf("[ERROR] CUDA memory free fail\n");
        return;
    }

    err = cudaFree(cu_result);
    if (err != cudaSuccess) {
        printf("[ERROR] CUDA memory free fail\n");
        return;
    }

    // ---------------------------------------------------------------- //
}