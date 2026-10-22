#include <cuda_runtime.h>
#include <stdio.h>

//============================================================================
// 坐标统一按 (x, y, z) 顺序打印；线性编号按 x 最快、其次 y、最后 z 展开。
// 这是编号规则，不是执行顺序；二维数据中约定 x 表示列、y 表示行。

// block 编号不决定调度先后：block 1 的输出可能先于 block 0。
// 不要依赖线程的打印顺序，也不要假定它们同时开始；输出先后不能证明执行起止顺序。
__global__ void print_idx_kernel() {
    printf("block idx xyz: (%3d, %3d, %3d), thread idx xyz: (%3d, %3d, %3d)\n",
           blockIdx.x, blockIdx.y, blockIdx.z,
           threadIdx.x, threadIdx.y, threadIdx.z);
}

__global__ void print_dim_kernel() {
    printf("grid dimension xyz: (%3d, %3d, %3d), block dimension xyz: (%3d, %3d, %3d)\n",
           gridDim.x, gridDim.y, gridDim.z,
           blockDim.x, blockDim.y, blockDim.z);
}

__global__ void print_thread_idx_per_block_kernel() {
    // 块内线性编号：x 的步长为 1，y 的步长为 blockDim.x，z 的步长为一层的线程数。
    // 加法项写成 z、y、x 不影响结果；改变分量对应的权重才会改变编号规则。
    int index = threadIdx.z * blockDim.x * blockDim.y +
                threadIdx.y * blockDim.x +
                threadIdx.x;

    printf("block idx xyz: (%3d, %3d, %3d), thread idx: %3d\n",
           blockIdx.x, blockIdx.y, blockIdx.z,
           index);
}

__global__ void print_thread_idx_per_grid_kernel() {
    int bSize = blockDim.z * blockDim.y * blockDim.x;

    int bIndex = blockIdx.z * gridDim.x * gridDim.y +
                 blockIdx.y * gridDim.x +
                 blockIdx.x;

    int tIndex = threadIdx.z * blockDim.x * blockDim.y +
                 threadIdx.y * blockDim.x +
                 threadIdx.x;

    // 先展开 block 编号，再展开块内线程编号，得到整个 grid 中唯一的线程编号。
    // 二维情况下，这个编号一般不等于整个数据矩阵的行优先下标 y * width + x。
    int index = bIndex * bSize + tIndex;

    printf("block idx: %3d, thread idx in block: %3d, thread idx: %3d\n",
           bIndex, tIndex, index);
}

__global__ void print_cord_kernel() {
    // 线程在当前 block 内的线性编号。
    int index = threadIdx.z * blockDim.x * blockDim.y +
                threadIdx.y * blockDim.x +
                threadIdx.x;

    // 线程在整个二维 grid 中的坐标：block 的起始坐标加上块内偏移，x 为列、y 为行。
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    printf("block idx xyz: (%3d, %3d, %3d), thread idx: %3d, cord: (%3d, %3d)\n",
           blockIdx.x, blockIdx.y, blockIdx.z,
           index, x, y);
}

void debug_print(const char* fn, dim3 gridDim, dim3 blockDim) {
    printf("\n%s: gridDim.xyz=(%d, %d, %d), blockDim.xyz=(%d, %d, %d)\n",
           fn,
           gridDim.x, gridDim.y, gridDim.z,
           blockDim.x, blockDim.y, blockDim.z);
}

void print_one_dim() {
    int inputSize = 8;                   // 一维的数据
    int blockDim = 4;                    // block里面thread个数
    int gridDim = inputSize / blockDim;  // grid里面block个数

    dim3 block(blockDim);  // 只定义x分量, y, z为1
    dim3 grid(gridDim);

    // 每次启动共 2 个 block，每个 block 有 4 个线程，即 8 个逻辑线程。
    // 同一 block 内按每 32 个线程组成一个 warp；不同 block 的线程不会合并为一个 warp。
    // 因此本例有 2 个 warp，每个只有 4 个有效线程，而不是 8 个线程组成一个 warp。
    debug_print(__FUNCTION__, grid, block);

    // 依次运行各个示例，观察同一组线程的不同编号表示。
    printf("print_idx_kernel\n");
    print_idx_kernel<<<grid, block>>>();
    // 启动 kernel 对 CPU 是异步的；同步成功后再继续下一个示例，不会给线程排序。
    cudaDeviceSynchronize();

    printf("print_dim_kernel\n");
    print_dim_kernel<<<grid, block>>>();
    cudaDeviceSynchronize();

    printf("print_thread_idx_per_block_kernel\n");
    print_thread_idx_per_block_kernel<<<grid, block>>>();
    cudaDeviceSynchronize();

    printf("print_thread_idx_per_grid_kernel\n");
    print_thread_idx_per_grid_kernel<<<grid, block>>>();
    cudaDeviceSynchronize();
}

void print_two_dim() {
    int inputWidth = 4;

    int blockDim = 2;
    int gridDim = inputWidth / blockDim;

    dim3 block(blockDim, blockDim);
    dim3 grid(gridDim, gridDim);

    // grid 为 (2, 2, 1)，block 为 (2, 2, 1)，共 4 个 block、16 个线程。
    debug_print(__FUNCTION__, grid, block);

    // 依次运行各个示例，观察二维坐标与线性编号的对应关系。
    printf("print_idx_kernel\n");
    print_idx_kernel<<<grid, block>>>();
    cudaDeviceSynchronize();

    printf("print_dim_kernel\n");
    print_dim_kernel<<<grid, block>>>();
    cudaDeviceSynchronize();

    printf("print_thread_idx_per_block_kernel\n");
    print_thread_idx_per_block_kernel<<<grid, block>>>();
    cudaDeviceSynchronize();

    printf("print_thread_idx_per_grid_kernel\n");
    print_thread_idx_per_grid_kernel<<<grid, block>>>();
    cudaDeviceSynchronize();
}

// 打印坐标, 用二维方便显示.
void print_cord() {
    int inputWidth = 4;

    int blockDim = 2;
    int gridDim = inputWidth / blockDim;

    dim3 block(blockDim, blockDim);
    dim3 grid(gridDim, gridDim);

    debug_print(__FUNCTION__, grid, block);

    printf("print_cord_kernel\n");
    print_cord_kernel<<<grid, block>>>();
    cudaDeviceSynchronize();
}

int main() {
    /*
    常见的同步方式：

    cudaDeviceSynchronize：阻塞调用它的 CPU 线程，等待当前设备此前提交的任务完成。
    cudaStreamSynchronize：阻塞调用它的 CPU 线程，等待指定流此前提交的任务完成。
    cudaThreadSynchronize：已弃用，使用 cudaDeviceSynchronize。
    __syncthreads：在 kernel 内同步同一 block 中的线程，不能同步不同 block。
    同步用于等待任务或线程到达同步点，不保证线程按编号顺序执行或打印。
    */
    print_one_dim();
    print_two_dim();
    print_cord();
    return 0;
}
