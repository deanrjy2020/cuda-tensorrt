#include <cuda_runtime.h>
#include <stdio.h>

#include "gelu.hpp"
#include "stream.hpp"
#include "timer.hpp"
#include "utils.hpp"

int seed;

void sleep_test() {
    Timer timer;

    int width = 1 << 10;  // 1024
    int size = width * width;

    float low = -1.0;
    float high = 1.0;

    int blockSize = 16;
    int taskCnt = 5;
    char str[100];

    /* 初始化 */
    float* src_host;
    float* tar_host;
    cudaMallocHost(&src_host, size * sizeof(float));
    cudaMallocHost(&tar_host, size * sizeof(float));

    seed += 1;
    initMatrixSigned(src_host, size, low, high, seed);
    LOG("Input size is %d x %d floats", width, width);

    /* GPU warmup */
    timer.start_gpu();
    SleepSingleStream(src_host, tar_host, width, blockSize, taskCnt);
    timer.stop_gpu();

    /* 1 stream */
    blockSize = 16;
    timer.start_gpu();
    SleepSingleStream(src_host, tar_host, width, blockSize, taskCnt);
    timer.stop_gpu();
    std::sprintf(str, "kernel <<<(%2d,%2d), (%2d,%2d)>>>, 1 stream: (H2D, kernel, D2H) x %2d,",
                 width / blockSize, width / blockSize, blockSize, blockSize, taskCnt);
    timer.duration_gpu(str);

    /* n streams */
    timer.start_gpu();
    SleepMultiStream(src_host, tar_host, width, blockSize, taskCnt);
    timer.stop_gpu();
    std::sprintf(str, "kernel <<<(%2d,%2d), (%2d,%2d)>>>, %d stream: (H2D, kernel, D2H) for each,",
                 width / blockSize, width / blockSize, blockSize, blockSize, taskCnt);
    timer.duration_gpu(str);
}

void matmul_test() {
    /*
     * 大家试着在这里对matmul计算做一个多流的计算看看整体延迟的改变
     * 可以观测到相比于kernel的计算, memcpy的延迟会很小
     */
}

void gelu_test() {
    /*
     * 大家试着在这里对gelu计算做一个多流的计算看看整体延迟的改变
     * 可以观测到相比于memcpy的计算, kernel的延迟会很小
     */
}

int main() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    // 需要先确认自己的GPU是否支持overlap计算
    if (prop.asyncEngineCount == 0) {
        LOG("device does not support overlap");
    } else {
        LOG("device supports overlap");
    }

    sleep_test();
    // matmul_test();
    // gelu_test();

    // 这里供大家自由发挥。建议花一些在这里做调度的练习。根据ppt里面的方案实际编写几个测试函数。举几个例子在这里
    // e.g. 一个stream处理: H2D, 多个kernel，D2H。之后多个stream进行overlap
    // e.g. 一个stream处理: H2D, 大kernel，小kernel, D2H。之后多个stream进行overlap
    // e.g. 一个stream处理: H2D, 大kernel, H2D, 小kernel, D2H。之后多个stream进行overlap
    // e.g. 一个stream处理: H2D, 小kernel, H2D, 大kernel, D2H。之后多个stream进行overlap
    // e.g. 一个stream处理: H2D, 另外几个流分别只处理kernel, 和D2H。之后所有stream进行overlap
    // e.g. 一个stream处理: H2D(局部), kernel(局部), D2H(局部)。之后所有stream进行overlap

    return 0;
}

/*

jetson agx xavier上的结果:
    release:
    device supports overlap
    Input size is 1048576
    sleep <<<(64,64), (16,16)>>>,  1 stream,  1 memcpy,  5 kernel uses 60.707680 ms
    sleep <<<(64,64), (16,16)>>>,  5 stream,  1 memcpy,  5 kernel uses 52.333855 ms

*/