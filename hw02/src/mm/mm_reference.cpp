#include "mm.h"


void mat_mul_reference(Mat& lhs, Mat& rhs, Mat& result, float& time) {

    // ---------------------------------------------------------------- //

    // check shape
    if ((lhs.cols != rhs.rows) || (lhs.rows != result.rows) || (rhs.cols != result.cols)) {
        printf("[ERROR] Reference MM shape mismatch (LHS: (%ld, %ld), RHS: (%ld, %ld), RES: (%ld, %ld)\n",
               lhs.rows, lhs.cols, rhs.rows, rhs.cols, result.rows, result.cols);
        return;
    }

    // check align
    if ((lhs.align != MatAlign::ROW_WISE) || (rhs.align != MatAlign::COL_WISE) ||
        (result.align != MatAlign::ROW_WISE)) {
        printf("[ERROR] Reference MM align mismatch\n");
        return;
    }

    // clear to 0
    result.clear();

    // ---------------------------------------------------------------- //
    auto start_time = get_time();

    // core
    for (int row = 0; row < lhs.rows; row++) {
        for (int col = 0; col < rhs.cols; col++) {

            auto row_ptr = lhs.data + row * lhs.cols;
            auto col_ptr = rhs.data + col * rhs.rows;

            float acc = 0.f;
            for (int i = 0; i < lhs.cols; i++) {
                acc += (*row_ptr++) * (*col_ptr++);
            }
            result.set(row, col, acc);
        }
    }

    time += get_duration_us(start_time);

    // ---------------------------------------------------------------- //
}