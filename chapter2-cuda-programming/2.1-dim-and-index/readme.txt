2.1 Grid、Block 与线程索引

本例仅依赖 CUDA Toolkit 和 C++ 编译器，不需要 TensorRT、cuDNN 或 OpenCV。
保留当前仓库的 xyz 输出顺序和教学注释。

在仓库根目录配置、构建、运行（Windows，已加载 MSVC 开发环境的 Git Bash）：

cmake --preset windows-debug
cmake --build --preset windows-debug --target 2.1-dim-and-index
./build/windows-debug/bin/2.1-dim-and-index.exe

程序依次打印一维、二维的 block/thread 索引、维度、线性索引及二维坐标。
GPU 线程的打印顺序不保证与索引顺序相同；线性索引计算顺序不代表执行顺序。
本例没有外部数据文件，可直接在仓库根目录运行。
