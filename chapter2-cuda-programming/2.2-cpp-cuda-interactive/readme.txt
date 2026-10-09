2.2 C++ / CUDA 交互

本例分别编译 src/main.cpp 和 src/print_index.cu，通过普通 C++ 函数调用 CUDA kernel 的封装函数。
仅依赖 CUDA Toolkit 和 C++ 编译器，不需要 TensorRT、cuDNN 或 OpenCV。

在仓库根目录配置、构建、运行（Windows，已加载 MSVC 开发环境的 Git Bash）：

cmake --preset windows-debug
cmake --build --preset windows-debug --target 2.2-cpp-cuda-interactive
./build/windows-debug/bin/2.2-cpp-cuda-interactive.exe

Release 构建时，将以上 windows-debug 替换为 windows-release。
VS Code 中打开仓库根目录，选择相同的 build/debug target。
程序打印 block 索引、block 内线程索引和全局线性线程索引；GPU 打印顺序不保证一致。
本例没有外部数据文件，可直接在仓库根目录运行。
