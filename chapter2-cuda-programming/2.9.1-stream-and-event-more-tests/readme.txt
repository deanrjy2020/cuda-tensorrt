2.9.1-stream-and-event-more-tests

使用根目录 CMake 构建，仅依赖 CUDA Toolkit 和 C++ 编译器。
Windows 下请先加载 MSVC x64 开发环境，再在仓库根目录执行：

cmake --preset windows-debug
cmake --build --preset windows-debug --target 2.9.1-stream-and-event-more-tests
./build/windows-debug/bin/2.9.1-stream-and-event-more-tests.exe

Release 构建时，将 windows-debug 替换为 windows-release。
VS Code 中选择 2.9.1-stream-and-event-more-tests 作为 build/debug target。
本例没有外部数据文件。
使用 asyncEngineCount 查询复制与计算重叠支持，兼容 CUDA 13.4。
NVTX 标记使用 CUDA Toolkit 附带的 nvtx3 头文件。
