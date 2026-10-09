2.6-nsight-system-and-compute

使用根目录 CMake 构建，仅依赖 CUDA Toolkit 和 C++ 编译器。
Windows 下请先加载 MSVC x64 开发环境，再在仓库根目录执行：

cmake --preset windows-debug
cmake --build --preset windows-debug --target 2.6-nsight-system-and-compute
./build/windows-debug/bin/2.6-nsight-system-and-compute.exe

Release 构建时，将 windows-debug 替换为 windows-release。
VS Code 中选择 2.6-nsight-system-and-compute 作为 build/debug target。
本例没有外部数据文件。
