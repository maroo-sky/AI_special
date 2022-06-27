#include "mv.h"

int main(int argc, char** argv) {

    const int num_iters_ref = 2;
    const int num_iters_cuda = 20;

    // ------------------------------------------------------------------ //
    // Change setting
    // ------------------------------------------------------------------ //
    const int64_t rows = 1024;
    const int64_t cols = 2048;
    const int tile_length = 32; // 16, 32 -> use tiling
    // ------------------------------------------------------------------ //

    printf("[LOG] START MatVec\n");
    printf("[LOG]\t\t(%ld, %ld) x (%ld,) -> (%ld,)\n",
           rows, cols, cols, rows);
    printf("[LOG]\t\tTile length: %d\n", tile_length);

    Mat mat(rows, cols, MatAlign::ROW_WISE);
    Vec vec(cols);
    mat.fill();
    vec.fill();

    Vec result_reference(rows);
    Vec result_cuda(rows);

    // ------------------------------------------------------------------ //

    float reference_time_sum = 0.f;
    for (int iter = 0; iter < num_iters_ref; iter++) {
        mat_vec_reference(mat, vec, result_reference, reference_time_sum);
    }
    printf("[TIME] Reference: %f (us)\n", reference_time_sum / num_iters_ref);

    // ------------------------------------------------------------------ //

    float cuda_time_sum = 0.f;
    for (int iter = 0; iter < num_iters_cuda; iter++) {
        mat_vec_cuda(mat, vec, result_cuda, tile_length, cuda_time_sum);
    }
    printf("[TIME] CUDA: %f (us)\n", cuda_time_sum / num_iters_cuda);

    // ------------------------------------------------------------------ //

    difference_check(result_reference.data, result_cuda.data, rows, 1e-5);

    printf("[LOG] Done MatVec\n");
    return 0;
}
