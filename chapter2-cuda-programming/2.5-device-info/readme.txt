2.5 CUDA 设备信息

本例使用 C++ 调用 CUDA Runtime API，输出 GPU 名称、计算能力、显存、缓存及线程/block/grid 限制。
仅依赖 CUDA Toolkit 和 C++ 编译器，不需要 TensorRT、cuDNN 或 OpenCV。

在仓库根目录配置、构建、运行（Windows，已加载 MSVC 开发环境的 Git Bash）：

cmake --preset windows-debug
cmake --build --preset windows-debug --target 2.5-device-info
./build/windows-debug/bin/2.5-device-info.exe

Release 构建时，将 windows-debug 替换为 windows-release。
VS Code 中打开仓库根目录，选择 2.5-device-info 作为 build/debug target。
本例只有 C++ 源文件，通过 CUDA::cudart 链接 CUDA Runtime，无需编译 CUDA kernel。
本例没有外部数据文件，可直接在仓库根目录运行。
GPU/显存频率通过 cudaDeviceGetAttribute 查询，兼容 CUDA 13.4 中已移除的 cudaDeviceProp 频率字段。
