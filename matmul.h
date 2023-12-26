#ifndef MATMUL_H
#define MATMUL_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

void matmul_d(uint64_t m, uint64_t n, uint64_t p, double* a, double* b, double* c);

void matmul_f(uint64_t m, uint64_t n, uint64_t p, float* a, float* b, float* c);

//void matmul_f(uint64_t m, uint64_t n, uint64_t p, float* a, float* b, float* c);


#ifdef __cplusplus
}
#endif

#endif //MATMUL_H