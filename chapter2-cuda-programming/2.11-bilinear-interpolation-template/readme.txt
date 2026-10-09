2.11-bilinear-interpolation-template

依赖 CUDA Toolkit、C++ 编译器和 OpenCV C++ 开发包（core、imgproc、imgcodecs）。
根 CMake 自动查找 ~/bin/opencv/build，也可通过 -DOpenCV_DIR=... 指定。

在仓库根目录执行（Windows，已加载 MSVC x64 开发环境）：
cmake --preset windows-debug
cmake --build --preset windows-debug --target 2.11-bilinear-interpolation-template
./build/windows-debug/bin/2.11-bilinear-interpolation-template.exe

Release 构建时，将 windows-debug 替换为 windows-release。
CMake 自动复制对应配置的 OpenCV DLL 到可执行文件旁，无需修改系统 PATH。
CMake 构建使用源码目录中的示例图片，运行工作目录不限。
输出目录：build/windows-debug/chapter2-cuda-programming/2.11-bilinear-interpolation-template/results/
VS Code 可从仓库根目录选择同名 build/debug target。
