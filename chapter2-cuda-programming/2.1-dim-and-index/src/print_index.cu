#include <cuda_runtime.h>
#include <stdio.h>

//============================================================================
// grid里的block和block里的thread都是row major(先增加x, 然后y, 然后z), 左上角开始

// 源程序不知道为什么用zyx的形式, 全部改回xyz顺序
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

    // 这里的index就是thread的global id
    int index = bIndex * bSize + tIndex;

    printf("block idx: %3d, thread idx in block: %3d, thread idx: %3d\n",
           bIndex, tIndex, index);
}

__global__ void print_cord_kernel() {
    // thread idx in block
    int index = threadIdx.z * blockDim.x * blockDim.y +
                threadIdx.y * blockDim.x +
                threadIdx.x;

    // thread在整个grid里面的坐标, 注意x=col, y=row
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

    debug_print(__FUNCTION__, grid, block);

    /* 这里建议大家吧每一函数都试一遍*/
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

void print_two_dim() {
    int inputWidth = 4;

    int blockDim = 2;
    int gridDim = inputWidth / blockDim;

    dim3 block(blockDim, blockDim);
    dim3 grid(gridDim, gridDim);

    debug_print(__FUNCTION__, grid, block);

    /* 这里建议大家吧每一函数都试一遍*/
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
    synchronize是同步的意思，有几种synchronize

    cudaDeviceSynchronize: CPU与GPU端完成同步，CPU不执行之后的语句，直到这个语句以前的所有cuda操作结束
    cudaStreamSynchronize: 跟cudaDeviceSynchronize很像，但是这个是针对某一个stream的。只同步指定的stream中的cpu/gpu操作，其他的不管
    cudaThreadSynchronize: 现在已经不被推荐使用的方法
    __syncthreads:         线程块内同步
    */
    print_one_dim();
    print_two_dim();
    print_cord();
    return 0;
}
