2.3 基础矩阵乘法

本例分别编译 CPU 矩阵乘法、CUDA kernel 和主程序，仅依赖 CUDA Toolkit 和 C++ 编译器。

在仓库根目录配置、构建、运行（Windows，已加载 MSVC 开发环境的 Git Bash）：

cmake --preset windows-debug
cmake --build --preset windows-debug --target 2.3-matmul-basic
./build/windows-debug/bin/2.3-matmul-basic.exe

Release 构建时，将 windows-debug 替换为 windows-release。
VS Code 中打开仓库根目录，选择 2.3-matmul-basic 作为 build/debug target。
程序计算 1024 × 1024 矩阵乘法，比较 CPU 与不同 block 大小的 GPU 结果，并输出耗时。
GPU 耗时包含内存分配、数据传输和释放；并非单独的 kernel 执行时间。
本例没有外部数据文件，可直接在仓库根目录运行。
