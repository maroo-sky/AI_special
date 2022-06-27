#include "add.h"


int main(int argc, char** argv) {

    const int num_iters_ref = 2;
    const int num_iters_cuda = 20;

    // ------------------------------------------------------------------ //
    // Change setting
    // ------------------------------------------------------------------ //
    const int64_t length = 1024 * 1024 * 8;  // 8 M = 32 MB
    const int threads_per_block = 128;
    // ------------------------------------------------------------------ //

    printf("[LOG] Start Add\n");
    printf("[LOG]\t\t(%ld,) + (%ld,) -> (%ld,)\n", length, length, length);
    printf("[LOG]\t\tThreads per block: %d\n", threads_per_block);

    Vec a(length);
    Vec b(length);
    a.fill();
    b.fill();

    Vec result_reference(length);
    Vec result_cuda(length);

    // ------------------------------------------------------------------ //

    float reference_time_sum = 0.f;
    for (int iter = 0; iter < num_iters_ref; iter++) {
        vec_add_reference(a, b, result_reference, reference_time_sum);
    }
    printf("[TIME] Reference: %f (us)\n", reference_time_sum / num_iters_ref);

    // ------------------------------------------------------------------ //

    float cuda_time_sum = 0.f;
    for (int iter = 0; iter < num_iters_cuda; iter++) {
        vec_add_cuda(a, b, result_cuda, threads_per_block, cuda_time_sum);
    }
    printf("[TIME] CUDA: %f (us)\n", cuda_time_sum / num_iters_cuda);

    // ------------------------------------------------------------------ //

    difference_check(result_reference.data, result_cuda.data, length, 1e-5);

    printf("[LOG] Done Add\n");
    return 0;
}