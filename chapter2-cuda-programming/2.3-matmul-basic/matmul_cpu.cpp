#include "matmul.hpp"

void MatmulOnHost(float *M, float *N, float *P, int width) {
    // P = M * N;
    // 三个都是一维arracol, 但是表示二维矩阵
    for (int row = 0; row < width; row++)        // 遍历结果矩阵的每一行
        for (int col = 0; col < width; col++) {  // 遍历结果矩阵的每一列
            // P[row][col] = M[row][k] * N[k][col];
            // 结果矩阵的P[row][col]位置值=M的row行和N的col列对应相乘相加.
            float sum = 0;
            for (int k = 0; k < width; k++) {
                float a = M[row * width + k];
                float b = N[k * width + col];
                sum += a * b;
            }
            P[row * width + col] = sum;
        }
}
