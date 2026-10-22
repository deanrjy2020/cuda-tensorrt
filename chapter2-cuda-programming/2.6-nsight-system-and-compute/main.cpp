#include <cuda_runtime.h>
#include <stdio.h>

#include "matmul.hpp"
#include "timer.hpp"
#include "utils.hpp"

int seed;
int main() {
    Timer timer;
    int width = 1 << 10;  // 1,024
    int low = 0;
    int high = 1;
    int size = width * width;
    int blockSize = 16;
    char str[50];

    float* h_matM = (float*)malloc(size * sizeof(float));
    float* h_matN = (float*)malloc(size * sizeof(float));
    float* d_matP = (float*)malloc(size * sizeof(float));

    seed = 1;
    initMatrix(h_matM, size, low, high, seed);
    seed += 1;
    initMatrix(h_matN, size, low, high, seed);

    /* GPU warmup */
    timer.start();
    MatmulOnDevice(h_matM, h_matN, d_matP, width, blockSize);
    timer.stop();
    std::sprintf(str, "matmul in gpu(warmup)<<<%d, %d>>>", width / blockSize, blockSize);
    timer.duration<Timer::ms>(str);

    /* GPU general implementation <<<512, 2>>>*/
    blockSize = 2;
    timer.start();
    MatmulOnDevice(h_matM, h_matN, d_matP, width, blockSize);
    timer.stop();
    std::sprintf(str, "matmul in gpu(general)<<<%d, %d>>>", width / blockSize, blockSize);
    timer.duration<Timer::ms>(str);

    /* GPU general implementation <<<256, 4>>>*/
    blockSize = 4;
    timer.start();
    MatmulOnDevice(h_matM, h_matN, d_matP, width, blockSize);
    timer.stop();
    std::sprintf(str, "matmul in gpu(general)<<<%d, %d>>>", width / blockSize, blockSize);
    timer.duration<Timer::ms>(str);

    /* GPU general implementation <<<128, 8>>>*/
    blockSize = 8;
    timer.start();
    MatmulOnDevice(h_matM, h_matN, d_matP, width, blockSize);
    timer.stop();
    std::sprintf(str, "matmul in gpu(general)<<<%d, %d>>>", width / blockSize, blockSize);
    timer.duration<Timer::ms>(str);

    /* GPU general implementation <<<64, 16>>>*/
    blockSize = 16;
    timer.start();
    MatmulOnDevice(h_matM, h_matN, d_matP, width, blockSize);
    timer.stop();
    std::sprintf(str, "matmul in gpu(general)<<<%d, %d>>>", width / blockSize, blockSize);
    timer.duration<Timer::ms>(str);

    /* GPU general implementation <<<32, 32>>>*/
    blockSize = 32;
    timer.start();
    MatmulOnDevice(h_matM, h_matN, d_matP, width, blockSize);
    timer.stop();
    std::sprintf(str, "matmul in gpu(general)<<<%d, %d>>>", width / blockSize, blockSize);
    timer.duration<Timer::ms>(str);

    return 0;
}
