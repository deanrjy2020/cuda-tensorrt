#include <cuda_runtime.h>
#include <stdio.h>

#include <string>

#include "utils.hpp"

//============================================================================
// 纯C++程序, makefile里面的nvcc相关的都没有用.
// todo, 理解每个值的意思, 加注释

int main() {
    int count;
    int index = 0;
    cudaGetDeviceCount(&count);
    while (index < count) {
        cudaSetDevice(index);
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, index);
        int clockRate = 0;
        int memoryClockRate = 0;
        cudaError_t err = cudaDeviceGetAttribute(&clockRate, cudaDevAttrClockRate, index);
        if (err != cudaSuccess) {
            LOG("cudaDevAttrClockRate: %s", cudaGetErrorString(err));
            return 1;
        }
        err = cudaDeviceGetAttribute(&memoryClockRate, cudaDevAttrMemoryClockRate, index);
        if (err != cudaSuccess) {
            LOG("cudaDevAttrMemoryClockRate: %s", cudaGetErrorString(err));
            return 1;
        }
        LOG("%-40s", "*********************Architecture related**********************");
        LOG("%-40s%d%s", "Device id: ", index, "");
        LOG("%-40s%s%s", "Device name: ", prop.name, "");
        LOG("%-40s%.1f%s", "Device compute capability: ", prop.major + (float)prop.minor / 10, "");
        LOG("%-40s%.2f%s", "GPU global meory size: ", (float)prop.totalGlobalMem / (1 << 30), "GB");
        LOG("%-40s%.2f%s", "L2 cache size: ", (float)prop.l2CacheSize / (1 << 20), "MB");
        LOG("%-40s%.2f%s", "Shared memory per block: ", (float)prop.sharedMemPerBlock / (1 << 10), "KB");
        LOG("%-40s%.2f%s", "Shared memory per SM: ", (float)prop.sharedMemPerMultiprocessor / (1 << 10), "KB");
        LOG("%-40s%.2f%s", "Device clock rate: ", clockRate * 1E-6, "GHz");
        LOG("%-40s%.2f%s", "Device memory clock rate: ", memoryClockRate * 1E-6, "Ghz");
        LOG("%-40s%d%s", "Number of SM: ", prop.multiProcessorCount, "");
        LOG("%-40s%d%s", "Warp size: ", prop.warpSize, "");

        LOG("%-40s", "*********************Parameter related************************");
        LOG("%-40s%d%s", "Max block numbers: ", prop.maxBlocksPerMultiProcessor, "");
        LOG("%-40s%d%s", "Max threads per block: ", prop.maxThreadsPerBlock, "");
        LOG("%-40s%d:%d:%d%s", "Max block dimension size:", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2], "");
        LOG("%-40s%d:%d:%d%s", "Max grid dimension size: ", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2], "");
        index++;
        printf("\n");
    }
    return 0;
}
/**

*********************Architecture related**********************
Device id:                              0
Device name:                            Xavier
Device compute capability:              7.2
GPU global meory size:                  30.26GB
L2 cache size:                          0.50MB
Shared memory per block:                48.00KB
Shared memory per SM:                   96.00KB
Device clock rate:                      1.38GHz
Device memory clock rate:               0.67Ghz
Number of SM:                           8
Warp size:                              32
*********************Parameter related************************
Max block numbers:                      32
Max threads per block:                  1024
Max block dimension size:               1024:1024:64
Max grid dimension size:                2147483647:65535:65535

在/usr/local/cuda/samples/1_Utilities/deviceQuery里面有更详细的.
*/
