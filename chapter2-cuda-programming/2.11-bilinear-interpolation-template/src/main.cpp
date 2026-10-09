#include <cuda_runtime.h>
#include <stdio.h>

#include <iostream>

#include "preprocess.hpp"
#include "timer.hpp"
#include "utils.hpp"
using namespace std;

#ifndef DEMO_DATA_DIR
#define DEMO_DATA_DIR "data/"
#endif
#ifndef DEMO_RESULTS_DIR
#define DEMO_RESULTS_DIR "results/"
#endif
int main() {
    Timer timer;

    string file_path = DEMO_DATA_DIR "fox.png";
    string output_prefix = DEMO_RESULTS_DIR;
    string output_path = "";

    cv::Mat input = cv::imread(file_path);
    if (input.empty()) {
        cerr << "Failed to read image: " << file_path << endl;
        return 1;
    }
    int tar_h = 500;
    int tar_w = 500;

    cv::Mat resizedInput_cpu;
    cv::Mat resizedInput_gpu;

    /*
     * bilinear interpolation resize的CPU/GPU速度比较
     * 由于CPU端做完预处理之后，进入如果有DNN也需要将数据传送到device上，
     * 所以这里为了让测速公平，仅对下面的部分进行测速:
     *
     * - host端
     *     cv::resize的bilinear interpolation
     *     normalization进行归一化处理
     *     BGR2RGB来实现通道调换
     *
     * - device端
     *     bilinear interpolation + normalization + BGR2RGB的自定义核函数
     *
     * 由于这个章节仅是初步CUDA学习，大家在自己构建推理模型的时候可以将这些地方进行封装来写的好看点，
     * 在这里代码我们更关注实现的逻辑部分
     *
     * tatics 列表
     * 0: 最近邻差值缩放 + 全图填充
     * 1: 双线性差值缩放 + 全图填充
     * 2: 双线性差值缩放 + 填充(letter box)
     * 3: 双线性差值缩放 + 填充(letter box) + 平移居中
     * */

    resizedInput_cpu = preprocess_cpu(input, tar_h, tar_w, timer);
    output_path = output_prefix + getFileName(file_path) + "_resized_bilinear_cpu.png";
    cv::cvtColor(resizedInput_cpu, resizedInput_cpu, cv::COLOR_RGB2BGR);
    if (!cv::imwrite(output_path, resizedInput_cpu)) {
        cerr << "Failed to write image: " << output_path << endl;
        return 1;
    }

    /*
     * 一般来说，CUDA核函数设计成模版函数，方便多种不同类型的数据进行代码复用
     * 比如说，后面的课程会讲到的某些TensorRT plugin的核函数，对于输入tensor是FP32, FP16, INT8都会做类似的处理
     * 不熟悉模版函数与函数模版的人借用这个时间练习一下
     */
    resizedInput_gpu = preprocess_gpu<uint8_t>(input, tar_h, tar_w, timer);
    output_path = output_prefix + getFileName(file_path) + "_resized_bilinear_letterbox_center_gpu_uint8.png";
    cv::cvtColor(resizedInput_cpu, resizedInput_cpu, cv::COLOR_RGB2BGR);
    if (!cv::imwrite(output_path, resizedInput_gpu)) {
        cerr << "Failed to write image: " << output_path << endl;
        return 1;
    }

    resizedInput_gpu = preprocess_gpu<float>(input, tar_h, tar_w, timer);
    output_path = output_prefix + getFileName(file_path) + "_resized_bilinear_letterbox_center_gpu_float32.png";
    resizedInput_gpu.convertTo(resizedInput_gpu, CV_8UC3);
    if (!cv::imwrite(output_path, resizedInput_gpu)) {
        cerr << "Failed to write image: " << output_path << endl;
        return 1;
    }

    return 0;
}

/*

Resize(bilinear) in cpu takes:                               uses 14.011460 ms
Resize(bilinear-letterbox-center, uint8) in gpu takes:       uses 0.969664 ms
Resize(bilinear-letterbox-center, float32) in gpu takes:     uses 0.975840 ms

*/