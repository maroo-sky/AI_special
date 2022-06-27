#ifndef CUDAPRACTICE_TENSOR_H
#define CUDAPRACTICE_TENSOR_H

#include "utils.h"

template<typename T>
struct VecType {
    // ---------------------------------------------------------------- //
    // Members
    // ---------------------------------------------------------------- //
    T* data = nullptr;
    int64_t length = 0;

    // --------------------------------------------------------------- //
    // Construct, Destruct
    // --------------------------------------------------------------- //
    explicit VecType(int64_t l) {
        length = l;
        alloc();
    }

    ~VecType() {
        release();
    }

    bool alloc() {
        if ((!data) && (length > 0)) {
            data = new T[length];
        }
        if (!data) {
            printf("[ERROR] Vector memory alloc fail\n");
            return false;
        }
        return true;
    }

    void release() {
        length = 0;
        delete[] data;
    }

    // --------------------------------------------------------------- //
    // Member functions
    // --------------------------------------------------------------- //

    bool check(int64_t i) {
        if (!data) {
            printf("[ERROR] Vector data null\n");
            return false;
        }
        if ((i < 0) || (i >= length)) {
            printf("[ERROR] Vector invalid access (%ld,) but shape is (%ld,)\n", i, length);
            return false;
        }
        return true;
    }

    void clear() {
        if (!data) return;
        memset(data, 0, length * sizeof(T));
    }

    void fill() {
        if (!data) return;
        for (int64_t i = 0; i < length; i++) {
            data[i] = static_cast<T>(rand() / (float) RAND_MAX - 0.5f);
        }
    }

    T get(int64_t i) {
        if (!check(i)) return 0;
        return data[i];
    }

    void set(int64_t i, T val) {
        if (!check(i)) return;
        data[i] = val;
    }

};

typedef VecType<float> Vec;


enum class MatAlign {
    ROW_WISE = 0,
    COL_WISE = 1,
};

template<typename T>
struct MatType {

    // ---------------------------------------------------------------- //
    // Members
    // ---------------------------------------------------------------- //
    T* data = nullptr;
    int64_t rows = 0;
    int64_t cols = 0;
    MatAlign align = MatAlign::ROW_WISE;

    // --------------------------------------------------------------- //
    // Construct, Destruct
    // --------------------------------------------------------------- //
    MatType(int64_t r, int64_t c, MatAlign a = MatAlign::ROW_WISE) {
        rows = r;
        cols = c;
        align = a;
        alloc();
    }

    ~MatType() {
        release();
    }

    bool alloc() {
        if ((!data) && (rows > 0) && (cols > 0)) {
            data = new T[rows * cols];
        }
        if (!data) {
            printf("[ERROR] Matrix memory alloc fail\n");
            return false;
        }
        return true;
    }

    void release() {
        rows = 0;
        cols = 0;
        delete[] data;
    }

    // --------------------------------------------------------------- //
    // Member functions
    // --------------------------------------------------------------- //

    bool check(int64_t ri, int64_t ci) {
        if (!data) {
            printf("[ERROR] Matrix data null\n");
            return false;
        }
        if ((ri < 0) || (ri >= rows) || (ci < 0) || (ci >= cols)) {
            printf("[ERROR] Matrix invalid access (%ld, %ld) but shape is (%ld, %ld)\n", ri, ci, rows, cols);
            return false;
        }
        return true;
    }

    bool check(int64_t i) {
        if (!data) {
            printf("[ERROR] Matrix data null\n");
            return false;
        }
        if ((i < 0) || (i >= rows * cols)) {
            printf("[ERROR] Matrix invalid access (%ld,) but shape is (%ld, %ld)\n", i, rows, cols);
            return false;
        }
        return true;
    }

    void clear() {
        if (!data) return;
        memset(data, 0, rows * cols * sizeof(T));
    }

    void fill() {
        if (!data) return;
        for (int64_t i = 0; i < rows * cols; i++) {
            data[i] = static_cast<T>(rand() / (float) RAND_MAX - 0.5f);
        }
    }

    T get(int64_t ri, int64_t ci) {
        if (!check(ri, ci)) return 0;
        if (align == MatAlign::ROW_WISE) {
            return data[ri * cols + ci];
        }
        else {  // COL_WISE
            return data[ci * rows + ri];
        }
    }

    T get(int64_t i) {
        if (!check(i)) return 0;
        return data[i];
    }

    void set(int64_t ri, int64_t ci, T val) {
        if (!check(ri, ci)) return;

        if (align == MatAlign::ROW_WISE) {
            data[ri * cols + ci] = val;
        }
        else {  // COL_WISE
            data[ci * rows + ri] = val;
        }
    }

    void set(int64_t i, T val) {
        if (!check(i)) return;
        data[i] = val;
    }

};

typedef MatType<float> Mat;

#endif //CUDAPRACTICE_TENSOR_H
