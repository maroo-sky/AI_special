#include "mm.h"

int main(int argc, char** argv) {

    const int num_iters_ref = 2;
    const int num_iters_cuda = 20;

    // ------------------------------------------------------------------ //
    // Change setting
    // ------------------------------------------------------------------ //
    const int64_t rows = 1024;
    const int64_t depth = 1024;
    const int64_t cols = 1024;
    const int tile_length = 32; // 16, 32 -> use tiling
    // ------------------------------------------------------------------ //

    printf("[LOG] START MatMul\n");
    printf("[LOG]\t\t(%ld, %ld) x (%ld, %ld) -> (%ld, %ld)\n",
           rows, depth, depth, cols, rows, cols);
    printf("[LOG]\t\tTile length: %d\n", tile_length);

    Mat lhs(rows, depth, MatAlign::ROW_WISE);
    Mat rhs(depth, cols, MatAlign::COL_WISE);
    lhs.fill();
    rhs.fill();

    Mat result_reference(rows, cols, MatAlign::ROW_WISE);
    Mat result_cuda(rows, cols, MatAlign::ROW_WISE);

    // ------------------------------------------------------------------ //

    float reference_time_sum = 0.f;
    for (int iter = 0; iter < num_iters_ref; iter++) {
        mat_mul_reference(lhs, rhs, result_reference, reference_time_sum);
    }
    printf("[TIME] Reference: %f (us)\n", reference_time_sum / num_iters_ref);

    // ------------------------------------------------------------------ //

    float cuda_time_sum = 0.f;
    for (int iter = 0; iter < num_iters_cuda; iter++) {
        mat_mul_cuda(lhs, rhs, result_cuda, tile_length, cuda_time_sum);
    }
    printf("[TIME] CUDA: %f (us)\n", cuda_time_sum / num_iters_cuda);

    // ------------------------------------------------------------------ //

    difference_check(result_reference.data, result_cuda.data, rows * cols, 1e-5);

    printf("[LOG] Done MatMul\n");
    return 0;
}
