#include "softmax.h"


int main(int argc, char** argv) {

    const int num_iters_ref = 2;
    const int num_iters_cuda = 20;

    // ------------------------------------------------------------------ //
    // Change setting
    // ------------------------------------------------------------------ //
    const int64_t length = 1024 * 2;  // 2 K = 8 KB
    const int threads_per_block = 128;
    // ------------------------------------------------------------------ //

    printf("[LOG] Start Softmax\n");
    printf("[LOG]\t\t(%ld,) -> (%ld,)\n", length, length);
    printf("[LOG]\t\tThreads per block: %d\n", threads_per_block);

    Vec vec(length);
    vec.fill();

    Vec result_reference(length);
    Vec result_cuda(length);

    // ------------------------------------------------------------------ //

    float reference_time_sum = 0.f;
    for (int iter = 0; iter < num_iters_ref; iter++) {
        softmax_reference(vec, result_reference, reference_time_sum);
    }
    printf("[TIME] Reference: %f (us)\n", reference_time_sum / num_iters_ref);

    // ------------------------------------------------------------------ //

    float cuda_time_sum = 0.f;
    for (int iter = 0; iter < num_iters_cuda; iter++) {
        softmax_cuda(vec, result_cuda, threads_per_block, cuda_time_sum);
    }
    printf("[TIME] CUDA: %f (us)\n", cuda_time_sum / num_iters_cuda);

    // ------------------------------------------------------------------ //

    difference_check(result_reference.data, result_cuda.data, length, 1e-5);

    printf("[LOG] Done Softmax\n");
    return 0;
}