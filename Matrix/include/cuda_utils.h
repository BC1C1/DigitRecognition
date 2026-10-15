#pragma once

#include <cuda_runtime.h>
#include <stdexcept>
#include <iostream>
#include <curand.h>
#include <thread>
#include <mutex>


inline cudaStream_t& getStream() {
    static cudaStream_t stream = []() {
        cudaStream_t g_stream;
        cudaStreamCreate(&g_stream);
        return g_stream;
        }();
    return stream;
}

inline void cuda_initAll() {
    cudaMemPool_t defaultPool;
    cudaDeviceGetDefaultMemPool(&defaultPool, 0);
    uint64_t threshold = UINT64_MAX;
    cudaMemPoolSetAttribute(defaultPool, cudaMemPoolAttrReleaseThreshold, &threshold);
}

template <typename T>
void cudaMallocAsync_throw(T** gpu_ptr, size_t size) {
    //if (size * sizeof(float) > 16 * 1024 * 1024) {   // 只打印 > 16MB 的
    //    std::cout << "[cudaMalloc] " << (size * sizeof(float) / 1024.0 / 1024.0) << " MB" << std::endl;
    //}
	auto err = cudaMallocAsync(gpu_ptr, size, getStream());
	if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
}

template <typename T>
void cudaFreeAsync_throw(T* gpu_ptr) {
    auto err = cudaFreeAsync(gpu_ptr, getStream());
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
}

inline curandGenerator_t getRandnGenerator() {
    static curandGenerator_t gen = [] {
        curandGenerator_t g;
        curandCreateGenerator(&g, CURAND_RNG_PSEUDO_DEFAULT);
        curandSetPseudoRandomGeneratorSeed(g, 42);
        return g;
        }();
    return gen;
}

inline void warnOddOnce(size_t n) {
    static std::once_flag flag;
    std::call_once(flag, [n] {
        fprintf(stderr, "[launchFillNormal] Warning: odd n=%zu, slow path. "
            "Prefer even n to avoid extra memcpy and random consumption.\n", n);
        });
}
