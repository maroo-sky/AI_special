#include "softmax.h"

void softmax_cuda(Vec& vec, Vec& result, int threads_per_block, float& time) {

    // ---------------------------------------------------------------- //

    // check shape
    if (vec.length != result.length) {
        printf("[ERROR] CUDA Softmax shape mismatch (Vec: (%ld,), RES: (%ld,)\n",
               vec.length, result.length);
        return;
    }

    // clear to 0
    result.clear();

    // ---------------------------------------------------------------- //
    cudaError_t err;

    // TODO (1) CUDA memory alloc (2) memory move
    int64_t num_elements_vec = vec.length;
    size_t num_bytes_vec = num_elements_vec * sizeof(float);

    float* cu_vec = nullptr;
    err = cudaMalloc(&cu_vec, num_bytes_vec);
    if (err != cudaSuccess) {
	printf("[ERROR] CUDA memory alloc fail\n");
	return;
    }

    int64_t num_elements_result = result.length;
    size_t num_bytes_result = num_elements_result * sizeof(float);

    float* cu_result = nullptr;
    err = cudaMalloc(&cu_result, num_bytes_result);
    if (err != cudaSuccess){
	printf("[ERROR] CUDA memory alloc fail\n");
	return;
    }

    // (2) cuda memory move
    err = cudaMemcpy(cu_vec, vec.data, num_bytes_vec, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
	printf("[ERROR] CUDA memory move host to device fail\n");
	return;
    }

    // ---------------------------------------------------------------- //

    // TODO set block and grid
    dim3 dimBlock(threads_per_block);
    dim3 dimGrid((vec.length + threads_per_block -1) / threads_per_block);

    void* args[] = {&cu_vec, &cu_result, &vec.length}
    // ---------------------------------------------------------------- //
    auto start_time = get_time();

    // TODO launch kernel
    err = cudaLaunchKernel((const void*) kernel_softmax,
		dimGrid, dimBlock,
		args, 0, nullptr);

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("[ERROR] CUDA device synchronize fail\n");
        return;
    }

    time += get_duration_us(start_time);

    // ---------------------------------------------------------------- //

    // TODO CUDA memory move
    err = cudaMemcpy(result.data, cu_result, num_bytes_result, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess){
	printf("[ERROR] CUDA memory move device to host fail\n");
	return;
    }

    // ---------------------------------------------------------------- //

    // TODO CUDA memory free
    err = cudaFree(cu_vec);
    if (err != cudaSuccess){
	printf("[ERROR] CUDA memory free fail\n");
	return;
    }

    err = cudaFree(cu_result);
    if (err != cudaSuccess){
	printf("[ERROR] CUDA memory free fail\n");
	return;
    }

    // ---------------------------------------------------------------- //

}
