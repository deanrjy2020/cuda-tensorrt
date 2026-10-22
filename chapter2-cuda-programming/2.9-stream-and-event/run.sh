#!/usr/bin/env bash
set -euo pipefail

sample_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
root_dir="$(cd -- "$sample_dir/../.." && pwd)"
cd -- "$sample_dir"

case "$OSTYPE" in
    linux*)
        cmake -S . -B build -G Ninja \
            -DCMAKE_BUILD_TYPE=Debug \
            -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
            -DCMAKE_CUDA_ARCHITECTURES=89 \
            -DCMAKE_CUDA_FLAGS_DEBUG="-g -G"
        executable=./build/2.9-stream-and-event
        ;;
    msys*|cygwin*)
        # Run from Git Bash with the x64 MSVC environment loaded.
        cmake -S . -B build -G Ninja \
            -DCMAKE_BUILD_TYPE=Debug \
            -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
            -DCMAKE_CXX_COMPILER=cl \
            -DCMAKE_CUDA_HOST_COMPILER=cl \
            -DCMAKE_CUDA_ARCHITECTURES=89
        executable=./build/2.9-stream-and-event.exe
        ;;
    *)
        printf "Unsupported platform: %s\n" "$OSTYPE" >&2
        exit 1
        ;;
esac

cmake --build build -j 8
cp build/compile_commands.json "$root_dir/compile_commands.json"

echo "running..."
"$executable"
