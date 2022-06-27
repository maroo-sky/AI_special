#include "reduce.h"

void reduce_sum_reference(Vec& vec, float& result, float& time) {

    // ---------------------------------------------------------------- //

    // clear to 0
    result = 0.f;

    // ---------------------------------------------------------------- //
    auto start_time = get_time();

    // core
    auto vec_ptr = vec.data;

    float sum = 0.f;
    for (int64_t i = 0; i < vec.length; i++) {
        sum += (*vec_ptr++);
    }
    result = sum;

    time += get_duration_us(start_time);

    // ---------------------------------------------------------------- //
}
