#include "cuda_runtime_api.h"
#include "utils.hpp"

/*
^<>V
                                                                                nBegin在4
                                                                                   |
                                                                N matrix           V
                                                                ---------------------------------
                                                                |  0  1 |  2  3 |  4  5 |  6  7 |
                                                                |  8  9 | 10 11 | 12 13 | 14 15 |
                                                                ---------------------------------
                                                                | 16 17 | 18 19 | 20 21 | 22 23 |
                                                                | 24 25 | 26 27 | 28 29 | 30 31 |
                                                                ---------------------------------
                                                                | 32 33 | 34 35 | 36 37 | 38 39 |
                                                                | 40 41 | 42 43 | 44 45 | 46 47 |
                                                                ---------------------------------
                                                                | 48 49 | 50 51 | 52 53 | 54 55 |
                                                                | 56 57 | 58 59 | 60 61 | 62 63 |
                                                                ---------------------------------

    mBegin在16   M matrix                           mEnd在23    P matrix
        |        ---------------------------------      |       ---------------------------------
        |        |  0  1 |  2  3 |  4  5 |  6  7 |      |       |  0  1 |  2  3 |  4  5 |  6  7 |
        |        |  8  9 | 10 11 | 12 13 | 14 15 |      |       |  8  9 | 10 11 | 12 13 | 14 15 |
        |        ---------------------------------      |       ---------------------------------
        -------> | 16 17 | 18 19 | 20 21 | 22 23 | <----        | 16 17 | 18 19 |*20*21*| 22 23 |  以*block为例, bRow=2, bCol=1
                 | 24 25 | 26 27 | 28 29 | 30 31 |              | 24 25 | 26 27 |*28*29*| 30 31 |  内部的(tRow, tCol)为: | 00, 10 |
                 ---------------------------------              ---------------------------------                       | 01, 11 |
                 | 32 33 | 34 35 | 36 37 | 38 39 |              | 32 33 | 34 35 | 36 37 | 38 39 |
                 | 40 41 | 42 43 | 44 45 | 46 47 |              | 40 41 | 42 43 | 44 45 | 46 47 |
                 ---------------------------------              ---------------------------------
                 | 48 49 | 50 51 | 52 53 | 54 55 |              | 48 49 | 50 51 | 52 53 | 54 55 |
                 | 56 57 | 58 59 | 60 61 | 62 63 |              | 56 57 | 58 59 | 60 61 | 62 63 |
                 ---------------------------------              ---------------------------------

    data:
        P_8x8 = M_8x8 * N_8x8, 一共4x4个block, 每个BLOCKSIZE=2x2,
    思路:
        如果不用smem的话, 以thread为单位, 每个P对应的thread要做8次loop, 即把M的一整行和N的一整列对应相乘相加
        smem:
            用上smem, smem是每个block内共享, 以block为单位, 每个P矩阵对应的block做4次loop, 即把M的一整行block和N的一整列block相乘相加
            每次都是把两个block放到两个smem里面做计算, 然后下一对block, 这里的block就是tile, 2x2
*/
__global__ void MatmulSharedStaticKernel(float *M, float *N, float *P, int width) {
    __shared__ float Ms[BLOCKSIZE][BLOCKSIZE];
    __shared__ float Ns[BLOCKSIZE][BLOCKSIZE];

    // Block index, 即当前在P的哪个block里面
    int bCol = blockIdx.x;
    int bRow = blockIdx.y;
    // Thread index in current block
    int tCol = threadIdx.x;
    int tRow = threadIdx.y;

    // mBegin和mEnd是thread的global id, 用来loop M里面的block用的, loop4次.
    // 对应的M里面的一行block的第一个block, 左上角thread
    int mBegin = width * BLOCKSIZE * bRow;
    // 对应的M里面的一行block的最后一个block, 右上角thread
    int mEnd = mBegin + width - 1;
    // 因为row-major的, 每次+2就可以到下一个block了.
    int mStep = BLOCKSIZE;

    // 同理用来loop N里面的block, 不用bEnd, 和a一起结束就好了.
    // 对应N里面的一列block的第一个block, 左上角thread
    int nBegin = BLOCKSIZE * bCol;
    // 要跳过2行(2*width)才能到下一个block.
    int nStep = BLOCKSIZE * width;

    // 每个thread一个, 对应P矩阵位置上的值, 累加到里面, 4对block做完了就是最终的值.
    float pVal = 0;

    // 以block为单位loop, 4次, 每次:
    //      把M和N里面的block load到smem里面.
    //      以block为单位计算当前thread对应的P里面的值, 累加起来.
    for (int m = mBegin, n = nBegin; m <= mEnd; m += mStep, n += nStep) {
        // 把M和N里面的整个block load到smem里面来, 4个thread每个都load自己对应位置上的
        // 线程自己位置上 = 以m为定位, 同一行的就是M[m+tCol], 下一行的再加上width * tRow
        Ms[tRow][tCol] = M[m + width * tRow + tCol];
        // N同理.
        Ns[tRow][tCol] = N[n + width * tRow + tCol];

        // 同步确定当前block的thread都把data load到smem了
        __syncthreads();

        // Multiply the two matrices together;
        // each thread computes one element
        // of the block sub-matrix
#pragma unroll
        // 当前P位置上的元素要用M block的一行和N block的一列对应相乘相加.
        for (int k = 0; k < BLOCKSIZE; ++k) {
            pVal += Ms[tRow][k] * Ns[k][tCol];
        }

        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        __syncthreads();
    }

    // P矩阵中, 由当前的block的bRow/bCol算得当前block的第一个元素(左上角)的一维位置pIdx.
    int rows = BLOCKSIZE * bRow;
    int pIdx = width * rows + BLOCKSIZE * bCol;
    // 然后以pIdx为基准, 由当前thread在当前block坐标得到对应的位置.
    // 同一行就是pIdx+tCol, 下一行再加width*tRow
    P[pIdx + width * tRow + tCol] = pVal;
}

/*
    和static smem大体一样, 不一样的地方:
        1, 动态内存是一维的, 上面的静态是二维的(也可以做成一维)
            一维用: Ms[tRow * blockSize + tCol], 二维用: Ms[tRow][tCol]
        2, blockSize用参数传进来, 不用BLOCKSIZE.
*/
__global__ void MatmulSharedDynamicKernel(float *M, float *N, float *P, int width, int blockSize) {
    /*
        声明动态共享变量的时候需要加extern，同时需要是一维的
        注意这里有个坑, 不能够像这样定义：
            extern __shared__ float Ms[];
            extern __shared__ float Ns[];
        因为在cuda中定义动态共享变量的话，无论定义多少个他们的地址都是一样的。
        所以如果想要像上面这样使用的话，需要用两个指针分别指向shared memory的不同位置才行
    */
    extern __shared__ float Ms[];
    float *Ns = &Ms[blockSize * blockSize];

    int bCol = blockIdx.x;
    int bRow = blockIdx.y;

    int tCol = threadIdx.x;
    int tRow = threadIdx.y;

    int mBegin = width * blockSize * bRow;
    int mEnd = mBegin + width - 1;
    int mStep = blockSize;

    int nBegin = blockSize * bCol;
    int nStep = blockSize * width;

    float pVal = 0;

    for (int m = mBegin, n = nBegin; m <= mEnd; m += mStep, n += nStep) {
        Ms[tRow * blockSize + tCol] = M[m + width * tRow + tCol];
        Ns[tRow * blockSize + tCol] = N[n + width * tRow + tCol];

        __syncthreads();

#pragma unroll
        for (int k = 0; k < blockSize; ++k) {
            pVal += Ms[tRow * blockSize + k] * Ns[k * blockSize + tCol];
        }
        __syncthreads();
    }

    int rows = blockSize * bRow;
    int pIdx = width * rows + blockSize * bCol;
    P[pIdx + width * tRow + tCol] = pVal;
}

/*
    使用Tiling技术
    一个tile处理的就是block, 将一个矩阵分为多个小的tile，这些tile之间的执行独立，并且可以并行
*/
void MatmulSharedOnDevice(float *M_host, float *N_host, float *P_host, int width, int blockSize, bool staticMem) {
    /* 设置矩阵大小 */
    int size = width * width * sizeof(float);
    long int sMemSize = blockSize * blockSize * sizeof(float) * 2;

    /* 分配M, N在GPU上的空间*/
    float *M_device;
    float *N_device;
    CUDA_CHECK(cudaMalloc((void **)&M_device, size));
    CUDA_CHECK(cudaMalloc((void **)&N_device, size));

    /* 分配M, N拷贝到GPU上*/
    CUDA_CHECK(cudaMemcpy(M_device, M_host, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(N_device, N_host, size, cudaMemcpyHostToDevice));

    /* 分配P在GPU上的空间*/
    float *P_device;
    CUDA_CHECK(cudaMalloc((void **)&P_device, size));
    ;

    /* 调用kernel来进行matmul计算, 在这个例子中我们用的方案是：使用一个grid，一个grid里有width*width个线程 */
    dim3 dimBlock(blockSize, blockSize);
    dim3 dimGrid(width / blockSize, width / blockSize);
    if (staticMem) {
        MatmulSharedStaticKernel<<<dimGrid, dimBlock>>>(M_device, N_device, P_device, width);
    } else {
        MatmulSharedDynamicKernel<<<dimGrid, dimBlock, sMemSize, nullptr>>>(M_device, N_device, P_device, width, blockSize);
    }

    /* 将结果从device拷贝回host*/
    CUDA_CHECK(cudaMemcpy(P_host, P_device, size, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaDeviceSynchronize());

    /* 注意要在synchronization结束之后排查kernel的错误 */
    LAST_KERNEL_CHECK();

    /* Free */
    cudaFree(P_device);
    cudaFree(N_device);
    cudaFree(M_device);
}
