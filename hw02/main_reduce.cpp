#include "reduce.h"


int main(int argc, char** argv) {

    const int num_iters_ref = 2;
    const int num_iters_cuda = 20;

    // ------------------------------------------------------------------ //
    // Change setting
    // ------------------------------------------------------------------ //
    const int64_t length = 1024 * 16;  // 16 K = 64 KB
    const int threads_per_block = 128;
    // ------------------------------------------------------------------ //

    printf("[LOG] Start ReduceSum\n");
    printf("[LOG]\t\t(%ld,)\n", length);
    printf("[LOG]\t\tThreads per block: %d\n", threads_per_block);

    Vec vec(length);
    vec.fill();

    float result_reference = 0.f;
    float result_cuda = 0.f;

    // ------------------------------------------------------------------ //

    float reference_time_sum = 0.f;
    for (int iter = 0; iter < num_iters_ref; iter++) {
        reduce_sum_reference(vec, result_reference, reference_time_sum);
    }
    printf("[TIME] Reference: %f (us)\n", reference_time_sum / num_iters_ref);

    // ------------------------------------------------------------------ //

    float cuda_time_sum = 0.f;
    for (int iter = 0; iter < num_iters_cuda; iter++) {
        reduce_sum_cuda(vec, result_cuda, threads_per_block, cuda_time_sum);
    }
    printf("[TIME] CUDA: %f (us)\n", cuda_time_sum / num_iters_cuda);

    // ------------------------------------------------------------------ //

    difference_check(&result_reference, &result_cuda, 1, 1e-5);

    printf("[LOG] Done ReduceSum\n");
    return 0;
}