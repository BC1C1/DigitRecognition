#pragma once
// ============================================================================
//  GpuTestCommon.h —— GPU 算子测试的公共设施
//
//  为什么和 TestCommon.h 分开：那份是给 CPU 版 Matrix 用的，它的 maxDiff 走
//  Matrix::operator()，而现在 operator() 解引用的是**显存指针** —— 在 GPU 矩阵
//  上调用它会直接段错误。这一份所有数据交换都显式走 cudaMemcpy。
//
//  【约定】
//    - Matrix 默认在显存：Matrix(r,c) 分配的就是显存，operator() 不可用
//    - 造数据一律在 host 端生成再上传（不用 Matrix::randnMatrix，那是 cuRAND，
//      而且它每次调用都重置种子，不适合用来造"两个不同的矩阵"）
//    - 每条 kernel 调用后查两次错误：launch 之后一次（抓启动错误），
//      cudaDeviceSynchronize 之后再一次（抓执行期的越界访问）
// ============================================================================

#include "TypeDefine.h"
#include <cuda_runtime.h>
#include "cuda_utils.h"
#include "cuda_bridge.h"

#include <cstdio>
#include <cmath>
#include <vector>
#include <algorithm>

namespace gputest {

// ---------------------------------------------------------------------------
// 失败计数（进程级）
// ---------------------------------------------------------------------------
inline int& fails() { static int f = 0; return f; }
inline void resetFails() { fails() = 0; }

// ---------------------------------------------------------------------------
// 两把尺子，沿用之前定的：参与乘加的用 kFloatTol，纯搬运的用 kExactTol
// ---------------------------------------------------------------------------
constexpr double kFloatTol = 1e-3;
constexpr double kExactTol = 1e-12;

inline bool exact(double a, double b) { return std::fabs(a - b) < kExactTol; }
inline bool close(double a, double b, double eps = kFloatTol) { return std::fabs(a - b) < eps; }

// ---------------------------------------------------------------------------
// CUDA 错误检查
// ---------------------------------------------------------------------------
inline bool cudaOk(const char* what)
{
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        printf("  [CUDA 错误] %s: %s\n", what, cudaGetErrorString(e));
        return false;
    }
    return true;
}

// launch 后查一次、同步后再查一次。返回 false 表示这一步不能继续了。
inline bool runAndCheck(const char* what)
{
    const bool okLaunch = cudaOk(what);
    cudaDeviceSynchronize();
    const bool okRun = cudaOk(what);
    return okLaunch && okRun;
}

// ---------------------------------------------------------------------------
// host <-> device
// ---------------------------------------------------------------------------
inline void upload(Matrix& m, const std::vector<float>& h)
{
    cudaMemcpy(m.data_gpu(), h.data(), m.size() * sizeof(float), cudaMemcpyHostToDevice);
}

inline std::vector<float> download(const Matrix& m)
{
    std::vector<float> h(m.size());
    cudaMemcpy(h.data(), m.data_gpu(), m.size() * sizeof(float), cudaMemcpyDeviceToHost);
    return h;
}

// 把已有的显存内容清成 0（不新分配）
inline void zeroDevice(Matrix& m)
{
    cudaMemset(m.data_gpu(), 0, m.size() * sizeof(float));
}

// ---------------------------------------------------------------------------
// 确定性造数。用 LCG 而不是 randnMatrix：
//   - 每个 seed 给出一批不同的数，方便造"两个不相干的矩阵"
//   - 不依赖 cuRAND 的生成器状态
// ---------------------------------------------------------------------------
inline std::vector<float> makeData(size_t n, unsigned seed, float lo = -1.0f, float hi = 1.0f)
{
    std::vector<float> v(n);
    unsigned s = seed * 2654435761u + 1u;
    for (size_t i = 0; i < n; ++i) {
        s = s * 1664525u + 1013904223u;
        const float u = (float)((s >> 8) & 0xFFFFFFu) / (float)0x1000000u;   // [0,1)
        v[i] = lo + u * (hi - lo);
    }
    return v;
}

// 造一个 device 上的矩阵，内容来自 makeData
inline Matrix makeMatrix(size_t r, size_t c, unsigned seed, float lo = -1.0f, float hi = 1.0f)
{
    Matrix m(r, c);
    upload(m, makeData(r * c, seed, lo, hi));
    return m;
}

// 造一个全 value 的矩阵
inline Matrix makeConst(size_t r, size_t c, float value)
{
    Matrix m(r, c);
    std::vector<float> h(r * c, value);
    upload(m, h);
    return m;
}

// ---------------------------------------------------------------------------
// 比较
// ---------------------------------------------------------------------------
inline double maxDiff(const std::vector<float>& a, const std::vector<float>& b)
{
    if (a.size() != b.size()) return 1e308;
    double m = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        const double d = std::fabs((double)a[i] - (double)b[i]);
        if (d > m) m = d;
    }
    return m;
}

// 相对容差：按量级缩放。求和、乘法这类结果会随规模增长的比较用这个。
//
// ⚠ 参数是 double，而且里面所有东西都走 double。写成 std::max(1.0, std::fabs(x))
//   的话，x 一旦是 float，std::fabs(float) 返回 float，模板参数 _Ty 就推导不出来
//   （MSVC C2672）—— 这个坑在本工程里已经踩过三次：TestCommon.h 的 maxDiff、
//   Im2ColTest 的 maxErr，还有这里。所以统一走这一个入口。
inline double relTol(double v, double base = kFloatTol)
{
    return base * std::max(1.0, std::fabs(v));
}

// 找出第一个明显不同的位置，用来打印现场
inline size_t firstDiff(const std::vector<float>& a, const std::vector<float>& b, double tol = kFloatTol)
{
    const size_t n = std::min(a.size(), b.size());
    for (size_t i = 0; i < n; ++i)
        if (std::fabs((double)a[i] - (double)b[i]) > tol) return i;
    return n;
}

} // namespace gputest

// ---------------------------------------------------------------------------
// 断言：失败时打印文件行号所在位置的信息
// ---------------------------------------------------------------------------
#define G_CHECK(cond, ...)                                  \
    do {                                                    \
        if (!(cond)) {                                      \
            ++::gputest::fails();                           \
            printf("  [FAIL] ");                            \
            printf(__VA_ARGS__);                            \
            printf("\n");                                   \
        }                                                   \
    } while (0)
