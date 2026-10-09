#pragma once
// ============================================================================
//  TestCommon.h —— 各个测试文件共用的断言与统计
//
//  为什么单独一份：Im2ColTest.h 和 MatmulTest.h 都要用 T_CHECK，宏重定义会
//  报警告、而且两份体可能不一致。公共的东西只写一次。
//
//  ⚠ 失败计数是【进程级】的：测试文件各自调 run*Tests()，readFail() 读到的
//    是累计值。想在中间看增量就自己记一次差值。
// ============================================================================

#include "TypeDefine.h"
#include <cstdio>
#include <cmath>
#include <algorithm>

namespace dstest {

inline int& fails()
{
    static int f = 0;
    return f;
}

inline void resetFails() { fails() = 0; }

// ---------------------------------------------------------------------------
//  两把尺子：float 化之后必须重定
//
//  double 时代这里是 1e-12 / 1e-9 —— 那是给 53 位尾数定的尺子。
//  float 只有 24 位尾数（eps 约 1.19e-7），K 维累加后的绝对误差按 eps·sqrt(K)·|值|
//  量级增长。实测（K 最大 120、randn 数据）：最大误差 7.6e-6。
//
//  为什么定 1e-3 而不是贴着实测的 8e-6 定：尺子的职责是【能区分要比较的两个假设】。
//  这里要区分的是
//      「累加顺序不同造成的舍入」 约 1e-6
//      「索引 / 搬运 / 转置 写错」 约 1e0
//  两者差六个数量级，1e-3 落在中间；将来 K 放大到几千（im2col 卷积的 12544），
//  误差也只涨到 1e-4 量级，仍然不会误报。
//
//  ⚠ 只有【纯搬运】才配用 exact：im2col 只重排位置、不做算术，float 化之后
//    依然逐位精确（实测全绿）。凡是参与乘加的，一律用 close。
// ---------------------------------------------------------------------------
// 用 constexpr 而不是 inline constexpr：后者是 C++17 特性（C7525），
// 而这个工程没设 CMAKE_CXX_STANDARD，MSVC 默认按 C++14 编。
// 命名空间作用域的 constexpr 变量本身是内部链接，放进头文件不会有 ODR 问题。
constexpr double kFloatTol = 1e-3;    // 参与乘加计算的路径
constexpr double kExactTol = 1e-12;   // 纯搬运（只重排，不计算）

// 严格相等（只给纯搬运、索引类测试用这个）
inline bool exact(double a, double b) { return std::fabs(a - b) < kExactTol; }

// 带容差（GEMM / 卷积 / 内积类测试用这个）
inline bool close(double a, double b, double eps = kFloatTol) { return std::fabs(a - b) < eps; }

// 两个矩阵的最大绝对差；形状不同返回一个巨大的数
//
// ⚠ 必须显式升到 double：X(i,j) 现在是 float，而 <cmath> 里 std::fabs 有 float 重载、
//   返回 float。std::max(double, float) 推导不出模板参数 _Ty，直接编译失败
//   （这是 double 时代永远不会出现、float 化之后才暴露的问题）。
//   顺带把两个 float 相减也放进 double 里做，避免大数相减的抵消误差。
inline double maxDiff(const Matrix& X, const Matrix& Y)
{
    if (X.rows() != Y.rows() || X.cols() != Y.cols()) return 1e308;
    double m = 0.0;
    for (size_t i = 0; i < X.rows(); ++i)
        for (size_t j = 0; j < X.cols(); ++j) {
            const double d = std::fabs((double)X(i, j) - (double)Y(i, j));
            m = std::max(m, d);
        }
    return m;
}

} // namespace dstest

#define T_CHECK(cond, ...)                          \
    do {                                            \
        if (!(cond)) {                              \
            ++::dstest::fails();                    \
            printf("  [FAIL] ");                    \
            printf(__VA_ARGS__);                    \
            printf("\n");                           \
        }                                           \
    } while (0)
