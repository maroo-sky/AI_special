#include "mv.h"

void mat_vec_reference(Mat& mat, Vec& vec, Vec& result, float& time) {

    // ---------------------------------------------------------------- //

    // check shape
    if ((mat.cols != vec.length) || (mat.rows != result.length)) {
        printf("[ERROR] Reference MV shape mismatch (Mat: (%ld, %ld), Vec: (%ld,), RES: (%ld,)\n",
               mat.rows, mat.cols, vec.length, result.length);
        return;
    }

    // check align
    if (mat.align != MatAlign::ROW_WISE) {
        printf("[ERROR] Reference MV align mismatch\n");
        return;
    }

    // clear to 0
    result.clear();

    // ---------------------------------------------------------------- //
    auto start_time = get_time();

    // core
    for (int row = 0; row < mat.rows; row++) {
        float acc = 0.f;
        auto mat_ptr = mat.data + row * mat.cols;
        auto vec_ptr = vec.data;

        for (int col = 0; col < mat.cols; col++) {
            acc += (*mat_ptr++) * (*vec_ptr++);
        }
        result.set(row, acc);
    }

    time += get_duration_us(start_time);

    // ---------------------------------------------------------------- //
}