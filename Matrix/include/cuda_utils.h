#pragma once

#include <cuda_runtime.h>
#include <stdexcept>
#include <iostream>
#include <curand.h>
#include <thread>
#include <mutex>

template <typename T>
void cudaMalloc_throw(T** gpu_ptr, size_t size) {
    //if (size * sizeof(float) > 16 * 1024 * 1024) {   // 只打印 > 16MB 的
    //    std::cout << "[cudaMalloc] " << (size * sizeof(float) / 1024.0 / 1024.0) << " MB" << std::endl;
    //}
	auto err = cudaMalloc(gpu_ptr, size);
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
