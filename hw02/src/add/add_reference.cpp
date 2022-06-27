#include "add.h"

void vec_add_reference(Vec& a, Vec& b, Vec& result, float& time) {

    // ---------------------------------------------------------------- //

    // check shape
    if ((a.length != b.length) || (a.length != result.length)) {
        printf("[ERROR] Reference Add shape mismatch (A: (%ld,), B: (%ld,), RES: (%ld,)\n",
               a.length, b.length, result.length);
        return;
    }

    // clear to 0
    result.clear();

    // ---------------------------------------------------------------- //
    auto start_time = get_time();

    // core
    auto a_ptr = a.data;
    auto b_ptr = b.data;

    for (int64_t i = 0; i < result.length; i++) {
        float l_val = (*a_ptr++);
        float r_val = (*b_ptr++);
        result.set(i, l_val + r_val);
    }

    time += get_duration_us(start_time);

    // ---------------------------------------------------------------- //
}
