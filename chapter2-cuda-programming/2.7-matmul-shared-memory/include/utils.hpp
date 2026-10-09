#ifndef __UTILS_HPP__
#define __UTILS_HPP__

#include <cuda_runtime.h>
#include <stdarg.h>

#include <system_error>

// set to 2 for dry run mode: width x width = 8x8, block num = 4x4, block size = 2x2
#define BLOCKSIZE 16
//#define BLOCKSIZE 2

#define CUDA_CHECK(call) __cudaCheck(call, __FILE__, __LINE__)
#define LAST_KERNEL_CHECK() __kernelCheck(__FILE__, __LINE__)
#define LOG(...) __log_info(__VA_ARGS__)

inline static void __cudaCheck(cudaError_t err, const char* file, const int line) {
    if (err != cudaSuccess) {
        printf("ERROR: %s:%d, ", file, line);
        printf("CODE:%s, DETAIL:%s\n", cudaGetErrorName(err), cudaGetErrorString(err));
        exit(1);
    }
}

inline static void __kernelCheck(const char* file, const int line) {
    cudaError_t err = cudaPeekAtLastError();
    if (err != cudaSuccess) {
        printf("ERROR: %s:%d, ", file, line);
        printf("CODE:%s, DETAIL:%s\n", cudaGetErrorName(err), cudaGetErrorString(err));
        exit(1);
    }
}

inline static void __log_info(const char* format, ...) {
    char msg[1000];
    va_list args;
    va_start(args, format);

    vsnprintf(msg, sizeof(msg), format, args);

    fprintf(stdout, "%s\n", msg);
    va_end(args);
}

void initMatrix(float* data, int size, int low, int high, int seed);
void printMat(float* data, int size);
void compareMat(float* h_data, float* d_data, int size);

#endif  //__UTILS_HPP__//
