#include <cuda_runtime.h>
#include <stdio.h>

#include "matmul.hpp"
#include "timer.hpp"
#include "utils.hpp"

int seed;
int main() {
    Timer timer;

    int width = 1 << 12;  // 4,096
    if (BLOCKSIZE == 2) { // dry run mode.
        width = 8;
    }
    int low = 0;
    int high = 1;
    int size = width * width;
    int blockSize = BLOCKSIZE; // 16 by default, 2 for dry run.
    bool statMem = true;
    char str[100];

    float* h_matM = (float*)malloc(size * sizeof(float));
    float* h_matN = (float*)malloc(size * sizeof(float));
    float* h_matP = (float*)malloc(size * sizeof(float));
    float* d_matP = (float*)malloc(size * sizeof(float));

    seed = 1;
    initMatrix(h_matM, size, low, high, seed);
    seed += 1;
    initMatrix(h_matN, size, low, high, seed);

    LOG("Input size is %d x %d", width, width);
    /* GPU warmup */
    timer.start_gpu();
    MatmulOnDevice(h_matM, h_matN, h_matP, width, blockSize);
    timer.stop_gpu();
    timer.duration_gpu("matmul in gpu(warmup)");

    /* GPU general implementation <<<256, 16>>>*/
    timer.start_gpu();
    MatmulOnDevice(h_matM, h_matN, d_matP, width, blockSize);
    timer.stop_gpu();
    std::sprintf(str, "matmul in gpu(without shared memory)<<<%d, %d>>>", width / blockSize, blockSize);
    timer.duration_gpu(str);
    compareMat(h_matP, d_matP, size);

    /* GPU general implementation <<<256, 16>>>*/
    timer.start_gpu();
    MatmulSharedOnDevice(h_matM, h_matN, d_matP, width, blockSize, statMem);
    timer.stop_gpu();
    std::sprintf(str, "matmul in gpu(with shared memory(static))<<<%d, %d>>>", width / blockSize, blockSize);
    timer.duration_gpu(str);
    compareMat(h_matP, d_matP, size);

    /* GPU general implementation <<<256, 16>>>*/
    statMem = false;
    timer.start_gpu();
    MatmulSharedOnDevice(h_matM, h_matN, d_matP, width, blockSize, statMem);
    timer.stop_gpu();
    std::sprintf(str, "matmul in gpu(with shared memory(dynamic))<<<%d, %d>>>", width / blockSize, blockSize);
    timer.duration_gpu(str);
    compareMat(h_matP, d_matP, size);

    return 0;
}

/*

jetson agx xavier上的结果:
    debug:
    Input size is 4096 x 4096
    matmul in gpu(warmup)                                        uses 10812.857422 ms
    matmul in gpu(without shared memory)<<<256, 16>>>            uses 10787.309570 ms
    matmul in gpu(with shared memory(static))<<<256, 16>>>       uses 19958.664062 ms
    matmul in gpu(with shared memory(dynamic))<<<256, 16>>>      uses 13722.878906 ms
    release:
    Input size is 4096 x 4096
    matmul in gpu(warmup)                                        uses 1469.154663 ms
    matmul in gpu(without shared memory)<<<256, 16>>>            uses 1420.153809 ms
    matmul in gpu(with shared memory(static))<<<256, 16>>>       uses 873.997803 ms
    matmul in gpu(with shared memory(dynamic))<<<256, 16>>>      uses 1110.216919 ms

*/