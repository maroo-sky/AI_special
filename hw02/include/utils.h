#ifndef CUDAPRACTICE_UTILS_H
#define CUDAPRACTICE_UTILS_H

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <chrono>

#include <cuda_runtime.h>

inline std::chrono::system_clock::time_point get_time() {
    return std::chrono::system_clock::now();
}

inline float get_duration_us(std::chrono::system_clock::time_point t) {
    auto n = get_time();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(n - t);
    return static_cast<float>(duration.count());
}


inline void
difference_check(float* reference, float* target, int64_t length, float tolerance = 1e-5) {

    int64_t count = 0;
    float diff_max = 0.f;
    float diff_mean = 0.f;
    for (int64_t i = 0; i < length; i++) {
        float diff = std::abs(reference[i] - target[i]);
        diff_mean += diff;
        if (diff > diff_max) {
            diff_max = diff;
        }
        if (diff > tolerance) {
            count += 1;
        }
    }
    diff_mean /= length;

    printf("[DIFF] Difference check\n"
           "[DIFF]\t\tTotal: %ld, Error: %ld (tolerance %f)\n"
           "[DIFF]\t\tDifference mean: %f, max: %f\n",
           length, count, tolerance, diff_mean, diff_max);

}

#endif //CUDAPRACTICE_UTILS_H
