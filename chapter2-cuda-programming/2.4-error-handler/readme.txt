2.4 CUDA 错误处理

本例在基础矩阵乘法中演示 CUDA API 和 kernel 错误检查，仅依赖 CUDA Toolkit 和 C++ 编译器。

在仓库根目录配置、构建、运行（Windows，已加载 MSVC 开发环境的 Git Bash）：

cmake --preset windows-debug
cmake --build --preset windows-debug --target 2.4-error-handler
./build/windows-debug/bin/2.4-error-handler.exe

Release 构建时，将 windows-debug 替换为 windows-release。
VS Code 中打开仓库根目录，选择 2.4-error-handler 作为 build/debug target。

程序先比较 blockSize 为 2、4、8、16、32 时的 CPU/GPU 矩阵乘法结果。
最后故意设置 blockSize=64，即每个 block 有 64 × 64 = 4096 个线程，触发非法启动配置。
预期触发 CUDA 错误检查并以退出码 1 结束，这是错误处理示例的预期行为。
当前 CUDA 13.4 环境中，错误首先在后续 cudaMemcpy 处被 CUDA_CHECK 捕获，输出 cudaErrorInvalidValue（invalid argument），尚未执行到 LAST_KERNEL_CHECK。
本例没有外部数据文件，可直接在仓库根目录运行。
