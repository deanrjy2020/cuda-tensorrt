#ifndef __UTILS_HPP__
#define __UTILS_HPP__

#include <cuda_runtime.h>

#include <system_error>

#define CUDA_CHECK(call)                                                                    \
    {                                                                                       \
        cudaError_t err = call;                                                             \
        if (err != cudaSuccess) {                                                           \
            printf("ERROR: %s:%d, ", __FILE__, __LINE__);                                   \
            printf("CODE:%s, DETAIL:%s\n", cudaGetErrorName(err), cudaGetErrorString(err)); \
            exit(1);                                                                        \
        }                                                                                   \
    }

void initMatrix(float* data, int size, int low, int high, int seed);
void printMat(float* data, int size);
void compareMat(float* h_data, float* d_data, int size);

#endif  //__UTILS_HPP__
