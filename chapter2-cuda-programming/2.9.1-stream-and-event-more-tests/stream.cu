#include <cuda_runtime.h>

#include <utils.hpp>

// #define MAX_ITER 10000     // memcpy == kernel / 10    (kernel执行的太快看不出来overlapping)
#define MAX_ITER 100000  // memcpy == kernel / 100   (开始能够看出来kernel的overlapping)
// #define MAX_ITER 10000000   // memcpy == kernel / 10000  (可以非常清楚的看到kernel的Overlapping)
#define SIZE 32

// 为了能够体现延迟，这里特意使用clock64()来进行模拟sleep
// 否则如果kernel计算太快，而无法观测到kernel在multi stream中的并发
// 大家根据自己的情况需修改sleep的时间
__global__ void SleepKernel(
    int64_t num_cycles) {
    int64_t cycles = 0;
    int64_t start = clock64();
    while (cycles < num_cycles) {
        cycles = clock64() - start;
    }
}

/*
multiStream = false:
    用一个stream做5次, 每次: H2D, kernel, D2H
multiStream = true:
    用5个stream做, 每个stream: H2D, kernel, D2H
*/
void SleepMultiStream(float* src_host, float* tar_host, int width,
                      bool multiStream, cudaStream_t* streams) {
    int size = width * width * sizeof(float);

    float* src_device;
    float* tar_device;

    CUDA_CHECK(cudaMalloc((void**)&src_device, size));
    CUDA_CHECK(cudaMalloc((void**)&tar_device, size));

    for (int i = 0; i < COUNT; i++) {
        if (multiStream) {
            CUDA_CHECK(cudaMemcpyAsync(src_device, src_host, size, cudaMemcpyHostToDevice, streams[i]));
        } else {
            CUDA_CHECK(cudaMemcpy(src_device, src_host, size, cudaMemcpyHostToDevice));
        }
        dim3 dimBlock(BLOCKSIZE, BLOCKSIZE);
        dim3 dimGrid(width / BLOCKSIZE, width / BLOCKSIZE);

        /* 这里面我们把参数写全了 <<<dimGrid, dimBlock, sMemSize, stream>>> */
        SleepKernel<<<dimGrid, dimBlock, 0, multiStream ? streams[i] : 0>>>(MAX_ITER);
        if (multiStream) {
            CUDA_CHECK(cudaMemcpyAsync(src_host, src_device, size, cudaMemcpyDeviceToHost, streams[i]));
        } else {
            CUDA_CHECK(cudaMemcpy(src_host, src_device, size, cudaMemcpyDeviceToHost));
        }
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    cudaFree(tar_device);
    cudaFree(src_device);
}
