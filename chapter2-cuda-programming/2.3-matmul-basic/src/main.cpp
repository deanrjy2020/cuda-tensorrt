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
    float* h_matP = (float*)malloc(size * sizeof(float));
    float* d_matP = (float*)malloc(size * sizeof(float));

    seed = 1;
    initMatrix(h_matM, size, low, high, seed);
    seed += 1;
    initMatrix(h_matN, size, low, high, seed);

    /* CPU */
    timer.start();
    MatmulOnHost(h_matM, h_matN, h_matP, width);
    timer.stop();
    timer.duration<Timer::ms>("matmul in cpu");

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
    compareMat(h_matP, d_matP, size);

    /* GPU general implementation <<<256, 4>>>*/
    blockSize = 4;
    timer.start();
    MatmulOnDevice(h_matM, h_matN, d_matP, width, blockSize);
    timer.stop();
    std::sprintf(str, "matmul in gpu(general)<<<%d, %d>>>", width / blockSize, blockSize);
    timer.duration<Timer::ms>(str);
    compareMat(h_matP, d_matP, size);

    /* GPU general implementation <<<128, 8>>>*/
    blockSize = 8;
    timer.start();
    MatmulOnDevice(h_matM, h_matN, d_matP, width, blockSize);
    timer.stop();
    std::sprintf(str, "matmul in gpu(general)<<<%d, %d>>>", width / blockSize, blockSize);
    timer.duration<Timer::ms>(str);
    compareMat(h_matP, d_matP, size);

    /* GPU general implementation <<<64, 16>>>*/
    blockSize = 16;
    timer.start();
    MatmulOnDevice(h_matM, h_matN, d_matP, width, blockSize);
    timer.stop();
    std::sprintf(str, "matmul in gpu(general)<<<%d, %d>>>", width / blockSize, blockSize);
    timer.duration<Timer::ms>(str);
    compareMat(h_matP, d_matP, size);

    /* GPU general implementation <<<32, 32>>>*/
    blockSize = 32;
    timer.start();
    MatmulOnDevice(h_matM, h_matN, d_matP, width, blockSize);
    timer.stop();
    std::sprintf(str, "matmul in gpu(general)<<<%d, %d>>>", width / blockSize, blockSize);
    timer.duration<Timer::ms>(str);
    compareMat(h_matP, d_matP, size);

    return 0;
}
/*
1, 比较不合理的地方,
    MatmulOnHost只是做计算, 没有内存分配和copy
    MatmulOnDevice里面分配3次, copy 3次, free 3次.

2, jetson agx xavier上的结果:
    debug:
    matmul in cpu                            uses 12568.5 ms
    matmul in gpu(warmup)<<<64, 16>>>        uses 331.029 ms
    matmul in gpu(general)<<<512, 2>>>       uses 2257.53 ms
    Matmul result is same, precision is 1.0E-4
    matmul in gpu(general)<<<256, 4>>>       uses 570.09 ms
    Matmul result is same, precision is 1.0E-4
    matmul in gpu(general)<<<128, 8>>>       uses 172.761 ms
    Matmul result is same, precision is 1.0E-4
    matmul in gpu(general)<<<64, 16>>>       uses 173.601 ms
    Matmul result is same, precision is 1.0E-4
    matmul in gpu(general)<<<32, 32>>>       uses 194.848 ms
    Matmul result is same, precision is 1.0E-4
    release:
    matmul in cpu                            uses 16283 ms
    matmul in gpu(warmup)<<<64, 16>>>        uses 149.837 ms
    matmul in gpu(general)<<<512, 2>>>       uses 195.427 ms
    Matmul result is same, precision is 1.0E-4
    matmul in gpu(general)<<<256, 4>>>       uses 69.3432 ms
    Matmul result is same, precision is 1.0E-4
    matmul in gpu(general)<<<128, 8>>>       uses 38.8649 ms
    Matmul result is same, precision is 1.0E-4
    matmul in gpu(general)<<<64, 16>>>       uses 28.5248 ms
    Matmul result is same, precision is 1.0E-4
    matmul in gpu(general)<<<32, 32>>>       uses 27.8098 ms
    Matmul result is same, precision is 1.0E-4

*/
