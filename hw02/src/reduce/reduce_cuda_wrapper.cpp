#include "reduce.h"

void reduce_sum_cuda(Vec& vec, float& result, int threads_per_block, float& time) {

    // ---------------------------------------------------------------- //

    // clear to 0
    result = 0.f;

    // ---------------------------------------------------------------- //
    cudaError_t err;

    // TODO (1) CUDA memory alloc (2) memory move
    int64_t num_elements_vec = vec.length;
    size_t = num_bytes_vec = num_elements_vec * sizeof(float);

    float* cu_vec = nullptr;
    err = cudaMalloc(&cu_vec, num_bytes_vec);
    if (err != cudaSuccess){
	    printf("[ERROR] CUDA memory alloc fail\n");
	    return;
    }
    size_t num_bytes_result = sizeof(int);

    float* cu_result = nullptr;
    err = cudaMalloc(&cu_result, num_bytes_result);
    if (err != cudaSuccess){
	    printf("[ERROR] CUDA memory alloc fail\n");
	    return;
    }

    //(2) cuda memory move
    err = cudaMemcpy(cu_vec, vec.data, num_bytes_vec, cudaMemcpyHostToDevice);
    if (err != cudaSuccess){
	    printf("[ERROR] CUDA memory move host to device fail\n");
	    return;
    }

    // ---------------------------------------------------------------- //

    // TODO set block and grid
    dim3 dimBlock(threads_per_block);
    dim3 dimGrid((vec.length + threads_per_block -1) / threads_per_block);

    void* args[] = {&cu_vec &cu_result, &vec.length};
    // ---------------------------------------------------------------- //
    auto start_time = get_time();

    // TODO launch kernel
    err = cudaLaunchKernel((const void*) kernel_reduce_sum,
		    dimGrid, dimBlock, args, 0, nullptr);

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("[ERROR] CUDA device synchronize fail\n");
        return;
    }

    time += get_duration_us(start_time);

    // ---------------------------------------------------------------- //

    // TODO CUDA memory move (if needed)
    err = cudaMemcpy(result, cu_result, num_bytes_result, cudaMemcpyDeviceToHost);
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
