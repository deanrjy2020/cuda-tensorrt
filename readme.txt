CUDA / TensorRT

todo, gitshowcheck 修正

CUDA / TensorRT 学习工程

在仓库根目录打开 VS Code，使用 CMake + Ninja 按 target 构建各个 demo。
当前已接入 2.1 至 2.11（含 2.9.1）；2.10 和 2.11 额外依赖 OpenCV C++ 开发包。
原始资料目录不参与构建，也不会被修改。

Windows 工具链

- VS Code：安装推荐的 CMake Tools、C/C++ 扩展。
- CMake 3.24+、Ninja。
- CUDA Toolkit：需要 nvcc，仅有显卡驱动不能编译 .cu 文件。
- 与所选 CUDA Toolkit 兼容的 MSVC x64 Build Tools；无需 Visual Studio IDE。
- 默认 CUDA 架构为 89，适用于 RTX 4070（含 Laptop GPU）。

2.1 至 2.9（含 2.9.1）无需安装 TensorRT、cuDNN、OpenCV。

2.10 / 2.11：OpenCV 默认查找 ~/bin/opencv/build；其他路径可在配置命令加 -DOpenCV_DIR="C:/实际路径/opencv/build"。OpenCV DLL 自动复制到 bin，示例图片输出到各自构建目录的 results 文件夹。

从根目录使用 VS Code

1. 打开本仓库根目录。
2. 执行 CMake: Select Configure Preset，选择 windows-debug 或 windows-release。
3. 执行 CMake: Configure。Windows preset 的 external architecture/toolset 用于让 CMake Tools 准备 MSVC 环境；若未自动找到工具链，从已加载 MSVC 的终端启动 code .。
4. 执行 CMake: Set Build Target，选择与 demo 文件夹同名的 target（已接入 2.1 至 2.11，含 2.9.1），然后执行 CMake: Build。
5. 执行 CMake: Set Debug Target，选择同一个 target；使用 CMake: Run Without Debugging 运行或 CMake: Debug 调试。

这里配置的是 Windows 主机端 C++ 调试，不包含 CUDA kernel 单步调试。
根目录共用一份 VS Code 配置，后续 demo 不需要单独打开文件夹。

Git Bash 命令行

Ninja 不提供 C++ 编译器。先打开已安装 Build Tools 对应的 x64 Native Tools Command Prompt，在其中启动 Git Bash（按实际 Git 安装路径调整）：

"C:\Program Files\Git\bin\bash.exe" --login -i

然后在 Git Bash 中：

Debug:

cmake --preset windows-debug
cmake --build --preset windows-debug --target 2.1-dim-and-index
./build/windows-debug/bin/2.1-dim-and-index.exe

Release：

cmake --preset windows-release
cmake --build --preset windows-release --target 2.1-dim-and-index
./build/windows-release/bin/2.1-dim-and-index.exe

两个 preset 均使用 -G Ninja，构建目录互相独立。build/<preset>/compile_commands.json 自动生成，不提交到 Git。
若 Toolkit 不在默认位置，可在配置命令增加 -DCMAKE_CUDA_COMPILER="C:/实际路径/bin/nvcc.exe"；更换编译器后用 cmake --fresh --preset windows-debug 重新配置。

逐个迁移

每个 demo 有自己的 CMakeLists.txt，并由根目录 add_subdirectory() 显式注册。target 名称必须与 demo 文件夹名完全一致（保留数字、点号和连字符），例如 2.1-dim-and-index；后续章节新增或迁移时也遵循此规则。
当前删除了根 Makefile 和 2.1 的 Makefile/本地 Makefile.config；2.2 至 2.11 原有 Makefile 暂时保留，推荐使用根 CMake 构建；其他章节的 Makefile 及共用配置保留到对应章节迁移时再处理。
仓库已有的 .vscode_2.1_bk 备份保持不变。

原有仓库维护笔记

LearnAI

todo:
    当前哪些不是644
    git ls-files -z | xargs -0 stat -c '%a %n' | awk '$1 != 644'
    把不是的全部改成644 (没有的话不用run这个了)
    git ls-files -z | xargs -0 stat -c '%a %n' | awk '$1 != 644 {print $2}' | xargs chmod 644

    Git 默认只跟踪是否设置了「可执行位（+x）」。
    Git 记录的文件权限信息非常有限，只区分：
    普通文件：100644
    可执行文件：100755
    Git 不会记录 rw-r--r--（644）和 rw-rw-r--（664）之间的区别，即使你用 chmod 改变了读写权限
    只要执行位不变，Git 不会认为文件有变
