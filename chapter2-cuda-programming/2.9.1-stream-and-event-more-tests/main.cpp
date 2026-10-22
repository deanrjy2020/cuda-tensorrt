#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>
#include <stdio.h>

#include "gelu.hpp"
#include "stream.hpp"
#include "timer.hpp"
#include "utils.hpp"

int seed;

void sleep_test(int width, cudaStream_t* streams) {
    Timer timer;
    int size = width * width;

    float low = -1.0;
    float high = 1.0;

    char str[100];

    std::sprintf(str, "sleep_test, width=%d", width);
    nvtxRangePush(str);

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
    SleepMultiStream(src_host, tar_host, width, false, nullptr);
    timer.stop_gpu();

    /* 1 stream */
    timer.start_gpu();
    SleepMultiStream(src_host, tar_host, width, false, nullptr);
    timer.stop_gpu();
    std::sprintf(str, "kernel <<<(%2d,%2d), (%2d,%2d)>>>, 1 stream: (H2D, kernel, D2H) x %2d,",
                 width / BLOCKSIZE, width / BLOCKSIZE, BLOCKSIZE, BLOCKSIZE, COUNT);
    timer.duration_gpu(str);

    /* n streams */
    timer.start_gpu();
    SleepMultiStream(src_host, tar_host, width, true, streams);
    timer.stop_gpu();
    std::sprintf(str, "kernel <<<(%2d,%2d), (%2d,%2d)>>>, %d stream: (H2D, kernel, D2H) for each,",
                 width / BLOCKSIZE, width / BLOCKSIZE, BLOCKSIZE, BLOCKSIZE, COUNT);
    timer.duration_gpu(str);

    nvtxRangePop();
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

    /* 先把所需要的stream创建出来 */
    cudaStream_t streams[COUNT];
    for (int i = 0; i < COUNT; i++) {
        CUDA_CHECK(cudaStreamCreate(&streams[i]));
    }
    for (int sft = 4; sft <= 10; ++sft) {
        int width = 1 << sft;  // 16 ~ 1024
        sleep_test(width, streams);
    }
    for (int i = 0; i < COUNT; i++) {
        // 使用完了以后不要忘记释放
        cudaStreamDestroy(streams[i]);
    }
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
    Input size is 16 x 16 floats
    kernel <<<( 1, 1), (16,16)>>>, 1 stream: (H2D, kernel, D2H) x  5,      uses 2.778176 ms
    kernel <<<( 1, 1), (16,16)>>>, 5 stream: (H2D, kernel, D2H) for each,  uses 1.194656 ms
    Input size is 32 x 32 floats
    kernel <<<( 2, 2), (16,16)>>>, 1 stream: (H2D, kernel, D2H) x  5,      uses 2.857760 ms
    kernel <<<( 2, 2), (16,16)>>>, 5 stream: (H2D, kernel, D2H) for each,  uses 1.160448 ms
    Input size is 64 x 64 floats
    kernel <<<( 4, 4), (16,16)>>>, 1 stream: (H2D, kernel, D2H) x  5,      uses 3.135904 ms
    kernel <<<( 4, 4), (16,16)>>>, 5 stream: (H2D, kernel, D2H) for each,  uses 1.322080 ms
    Input size is 128 x 128 floats
    kernel <<<( 8, 8), (16,16)>>>, 1 stream: (H2D, kernel, D2H) x  5,      uses 2.798976 ms
    kernel <<<( 8, 8), (16,16)>>>, 5 stream: (H2D, kernel, D2H) for each,  uses 2.221696 ms
    Input size is 256 x 256 floats
    kernel <<<(16,16), (16,16)>>>, 1 stream: (H2D, kernel, D2H) x  5,      uses 7.825120 ms
    kernel <<<(16,16), (16,16)>>>, 5 stream: (H2D, kernel, D2H) for each,  uses 7.174688 ms
    Input size is 512 x 512 floats
    kernel <<<(32,32), (16,16)>>>, 1 stream: (H2D, kernel, D2H) x  5,      uses 27.998465 ms
    kernel <<<(32,32), (16,16)>>>, 5 stream: (H2D, kernel, D2H) for each,  uses 26.238176 ms
    Input size is 1024 x 1024 floats
    kernel <<<(64,64), (16,16)>>>, 1 stream: (H2D, kernel, D2H) x  5,      uses 66.266273 ms
    kernel <<<(64,64), (16,16)>>>, 5 stream: (H2D, kernel, D2H) for each,  uses 50.511806 ms

*/