#ifndef __STREAM_HPP__
#define __STREAM_HPP__

// void SleepSingleStream(
//     float* src_host, float* tar_host,
//     int width, int blockSize,
//     int count);

void SleepMultiStream(float* src_host, float* tar_host, int width,
                      bool multiStream, cudaStream_t* streams);

#endif  // __STREAM_HPP__
