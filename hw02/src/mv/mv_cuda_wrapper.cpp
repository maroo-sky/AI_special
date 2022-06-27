#include "mv.h"

void mat_vec_cuda(Mat& mat, Vec& vec, Vec& result, int tile_length, float& time) {

    // ---------------------------------------------------------------- //

    // check shape
    if ((mat.cols != vec.length) || (mat.rows != result.length)) {
        printf("[ERROR] CUDA MV shape mismatch (Mat: (%ld, %ld), Vec: (%ld,), RES: (%ld,)\n",
               mat.rows, mat.cols, vec.length, result.length);
        return;
    }

    // check align
    if (mat.align != MatAlign::ROW_WISE) {
        printf("[ERROR] CUDA MV align mismatch\n");
        return;
    }

    // clear to 0
    result.clear();

    // ---------------------------------------------------------------- //
    cudaError_t err;

    // TODO (1) CUDA memory alloc (2) memory move
    int64_t num_elements_mat = mat.rows * mat.cols;
    size_t num_bytes_mat = num_elements_mat * sizeof(float);

    float* cu_mat = nullptr;
    err = cudaMalloc(&cu_mat, num_bytes_mat);
    if (err != cudaSuccess) {
	    printf("[ERROR] CUDA memory alloc fail\n");
	    return;
    }

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
    if (err != cudaSuccess) {
	    printf("[ERROR] CUDA memory alloc fail\n");
	    return;
    }
    //(2) memory move
    err = cudaMemcpy(cu_mat, mat.data, num_bytes_mat, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
	    printf("[ERROR] CUDA memory move host to device fail\n");
	    return;
    }

    err = cudaMemcpy(cu_vec, vec.data, num_bytes_vec, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
	    printf("[ERROR] CUDA memory move host to device fail\n");
	    return;
    }


    // ---------------------------------------------------------------- //

    // TODO set block and grid
    dim3 dimBlock(tile_length, tile_length);
    dim3 dimGrid((mat.cols + tile_length - 1) / tile_length,
		(vec.length + tile_length -1) / tile_length);

    void* args[] = {cu_mat, &cu_vec, &cu_result, &mat.rows, &mat.cosl};
    
    // ---------------------------------------------------------------- //
    auto start_time = get_time();

    // TODO launch kernel
    if (tile_length == 16) {
	    err = cudaLaunchKernel((const void*) kernel_mat_vec_tile_b16,
			    dimGrid, dimBlock, args, 0, nullptr);
    }
    else if (tile_length == 32) {
	    err = cudaLaunchKernel((const void*) kernel_mat_vec_tile_b32,
			    dimGrid, dimBlock, args, 0, nullptr);
    }
    else {
	    err = cudaLaunchKernel((const void*) kernel_mat_vec,
			    dimGrid, dimBlock,
			    args, 0, nullptr);
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("[ERROR] CUDA device synchronize fail\n");
        return;
    }

    time += get_duration_us(start_time);

    // ---------------------------------------------------------------- //

    // TODO CUDA memory move
    err = cudaMemcpy(result.data, cu_result, num_bytes_result, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
	    printf("[ERROR] CUDA memor move device to host fail\n");
	    return;
    }
    // ---------------------------------------------------------------- //

    // TODO CUDA memory free
    err = cudaFree(cu_mat);
    if (err != cudaSuccess){
	    prinft("[ERROR] CUDA memory free fail\n");
	    return;
    }

    err = cudaFree(cu_vec);
    if (err != cudaSuccess) {
	    printf("[ERROR] CUDA memory free fial\n");
	    return;
    }

    err = cudaFree(cu_result);
    if (err != cudaSuccess) {
	    printf("[ERROR] CUDA Memory free fail\n");
	    return;
    }

    // ---------------------------------------------------------------- //

}
