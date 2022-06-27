#include "add.h"


void vec_add_cuda(Vec& a, Vec& b, Vec& result, int threads_per_block, float& time) {

    // ---------------------------------------------------------------- //

    // check shape
    if ((a.length != b.length) || (a.length != result.length)) {
        printf("[ERROR] CUDA Add shape mismatch (A: (%ld,), B: (%ld,), RES: (%ld,)\n",
               a.length, b.length, result.length);
        return;
    }

    // clear to 0
    result.clear();

    // ---------------------------------------------------------------- //
    cudaError_t err;

    // cuda memory alloc
    int64_t num_elements = a.length;
    size_t num_bytes = num_elements * sizeof(float);

    float* cu_a = nullptr;
    err = cudaMalloc(&cu_a, num_bytes);
    if (err != cudaSuccess) {
        printf("[ERROR] CUDA memory alloc fail\n");
        return;
    }

    float* cu_b = nullptr;
    err = cudaMalloc(&cu_b, num_bytes);
    if (err != cudaSuccess) {
        printf("[ERROR] CUDA memory alloc fail\n");
        return;
    }

    float* cu_result = nullptr;
    err = cudaMalloc(&cu_result, num_bytes);
    if (err != cudaSuccess) {
        printf("[ERROR] CUDA memory alloc fail\n");
        return;
    }

    // ---------------------------------------------------------------- //

    // cuda memory move
    err = cudaMemcpy(cu_a, a.data, num_bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        printf("[ERROR] CUDA memory move host to device fail\n");
        return;
    }

    err = cudaMemcpy(cu_b, b.data, num_bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        printf("[ERROR] CUDA memory move host to device fail\n");
        return;
    }

    // ---------------------------------------------------------------- //

    dim3 dimBlock(threads_per_block);
    dim3 dimGrid((num_elements + threads_per_block - 1) / threads_per_block);
    void* args[] = {&cu_a, &cu_b, &cu_result, &num_elements};

    auto start_time = get_time();

    // launch
    err = cudaLaunchKernel((const void*) kernel_vec_add,
                           dimGrid, dimBlock,
                           args, 0, nullptr);
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
    err = cudaMemcpy(result.data, cu_result, num_bytes, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        printf("[ERROR] CUDA memory move device to host fail\n");
        return;
    }

    // ---------------------------------------------------------------- //

    // release
    err = cudaFree(cu_a);
    if (err != cudaSuccess) {
        printf("[ERROR] CUDA memory free fail\n");
        return;
    }

    err = cudaFree(cu_b);
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
