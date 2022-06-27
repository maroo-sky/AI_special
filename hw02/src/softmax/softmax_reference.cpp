#include "softmax.h"

void softmax_reference(Vec& vec, Vec& result, float& time) {

    // ---------------------------------------------------------------- //

    // check shape
    if (vec.length != result.length) {
        printf("[ERROR] Reference Softmax shape mismatch (Vec: (%ld,), RES: (%ld,)\n",
               vec.length, result.length);
        return;
    }

    // clear to 0
    result.clear();

    // ---------------------------------------------------------------- //
    auto start_time = get_time();

    // core
    auto vec_ptr = vec.data;

    float exp_sum = 0.f;
    for (int64_t i = 0; i < result.length; i++) {
        exp_sum += std::exp(*vec_ptr++);
    }

    vec_ptr = vec.data;  // reset
    for (int64_t i = 0; i < result.length; i++) {
        float val = *vec_ptr++;
        result.set(i, std::exp(val) / exp_sum);
    }

    time += get_duration_us(start_time);

    // ---------------------------------------------------------------- //
}