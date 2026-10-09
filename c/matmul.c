// Naive square matrix multiplication C = A @ B, for n x n matrices stored row-major in flat arrays.
// Both functions perform exactly 2 * n^3 floating-point operations; they differ only in loop order.

#include <stddef.h>

// Textbook order: the innermost loop walks B down a column, jumping n doubles in memory at every step.
void matmul_ijk(const double *a, const double *b, double *c, size_t n) {
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < n; j++) {
            double sum = 0.0;
            for (size_t k = 0; k < n; k++) {
                sum += a[i * n + k] * b[k * n + j];
            }
            c[i * n + j] = sum;
        }
    }
}

// Swapped order: the innermost loop walks rows of B and C contiguously, so every cache line is fully used
// and the compiler can vectorise it with SIMD instructions.
void matmul_ikj(const double *a, const double *b, double *c, size_t n) {
    for (size_t i = 0; i < n * n; i++) {
        c[i] = 0.0;
    }
    for (size_t i = 0; i < n; i++) {
        for (size_t k = 0; k < n; k++) {
            double a_ik = a[i * n + k];
            for (size_t j = 0; j < n; j++) {
                c[i * n + j] += a_ik * b[k * n + j];
            }
        }
    }
}
