#pragma once
// ============================================================================
//  MaxPoolTest.h —— 最大池化的正确性测试（自包含，不依赖 TestCommon.h）
//
//  【测试认定的接口】如果你的类名或签名不同，改这一处即可（或告诉我，我改测试）
//      class MaxPool2D : public Layer {
//          MaxPool2D(size_t H, size_t W, size_t channel, size_t ksize, size_t stride);
//          Matrix forward(const Matrix& x);      // (N, C*H*W)   -> (N, C*OH*OW)
//          Matrix backward(const Matrix& grad);  // (N, C*OH*OW) -> (N, C*H*W)
//          void   update(double lr);             // 无参数，空实现
//      };
//      OH = (H - ksize) / stride + 1，不补 padding
//      行主序与 Conv2D 一致，逐通道独立（通道之间不混合）
//
//  【两条硬性要求】
//      1) forward 的输出就是窗口最大值（P1/P2/P3）
//      2) backward 是【散射-累加】：梯度只回给 argmax 那一个位置，但 stride < ksize
//         时同一个像素可能是多个窗口的最大值 —— 那些梯度必须相加（P4/P5）
//
//  【测试清单】
//      P1  4x4 K=2 S=2：手算期望（不重叠，最简单）
//      P2  六种形状：K=2/3、S=1/2、C>1、N>1、28x28、stride>ksize
//      P3  独立参照：逐元素必须等于窗口最大值（不抄实现的循环）
//      P4  argmax 路由：只在一个输出位置给梯度，验证它只落到对应的最大值像素
//      P5  重叠累加：K=3 S=1 时中心像素被 4 个窗口同时选中，梯度必须是 4 倍
//      P6  数值梯度：中心差分对拍（max 的稀疏路由会让位置错误立刻暴露）
// ============================================================================

#include "MaxPool2D.h"
#include "TypeDefine.h"

#include <cstdio>
#include <cmath>
#include <algorithm>
#include <limits>

namespace pooltest {

inline int& fails() { static int f = 0; return f; }
inline bool close(double a, double b, double eps = 1e-9) { return std::fabs(a - b) < eps; }

#define P_CHECK(cond, ...)                          \
    do {                                            \
        if (!(cond)) {                              \
            ++::pooltest::fails();                  \
            printf("  [FAIL] ");                    \
            printf(__VA_ARGS__);                    \
            printf("\n");                           \
        }                                           \
    } while (0)

inline size_t outDim(size_t v, size_t K, size_t S) { return (v - K) / S + 1; }

// 窗口最大值（独立参照，直接用原图索引算，不碰被测量的实现）
inline double windowMax(const Matrix& X, size_t n, size_t c,
                        size_t H, size_t W, size_t oh, size_t ow,
                        size_t K, size_t S)
{
    double mx = -std::numeric_limits<double>::infinity();
    for (size_t kh = 0; kh < K; ++kh)
        for (size_t kw = 0; kw < K; ++kw)
            mx = std::max(mx, X(n, c * H * W + (oh * S + kh) * W + ow * S + kw));
    return mx;
}

// ---------------------------------------------------------------------------
// P1：4x4 K=2 S=2 —— 手算。图是 1..16 行主序，四个窗口的最大值是 6 8 14 16
// ---------------------------------------------------------------------------
inline void p1_hand_computed()
{
    printf("[P1] 4x4 K=2 S=2：手算期望 6 8 14 16\n");

    MaxPool2D pool(4, 4, 1, 2, 2);
    Matrix X(1, 16);
    for (size_t i = 0; i < 16; ++i) X(0, i) = (double)(i + 1);

    Matrix y = pool.forward(X);
    P_CHECK(y.rows() == 1 && y.cols() == 4, "形状：期望 1x4，实得 %zux%zu", y.rows(), y.cols());
    if (y.rows() != 1 || y.cols() != 4) return;

    const double want[4] = { 6, 8, 14, 16 };
    for (size_t i = 0; i < 4; ++i)
        P_CHECK(close(y(0, i), want[i]), "输出[%zu]：期望 %.0f，实得 %.1f", i, want[i], y(0, i));
}

// ---------------------------------------------------------------------------
// P2 + P3：多形状扫描 + 独立参照
// ---------------------------------------------------------------------------
inline void p2_shapes_with_reference()
{
    printf("[P2/P3] 六种形状，逐元素与窗口最大值对拍\n");

    struct Case { size_t N, C, H, W, K, S; const char* tag; };
    const Case cases[] = {
        { 1, 1,  4,  4, 2, 2, "不重叠" },
        { 1, 1,  4,  4, 3, 1, "重叠最重" },
        { 2, 3,  5,  5, 2, 2, "多 batch 多通道" },
        { 1, 2,  7,  7, 3, 2, "K=3 S=2" },
        { 2, 1, 28, 28, 2, 2, "MNIST 规模" },
        { 1, 1,  6,  6, 2, 3, "stride>ksize" },
    };

    for (const Case& cs : cases) {
        const size_t OH = outDim(cs.H, cs.K, cs.S), OW = outDim(cs.W, cs.K, cs.S);
        // 注意：每行的列数是 C*OH*OW，不要乘 batch（我第一版就错在这里，
        // N=1 的用例恰好掩盖了它）
        const size_t expCols = cs.C * OH * OW;

        MaxPool2D pool(cs.H, cs.W, cs.C, cs.K, cs.S);
        Matrix X = Matrix::randnMatrix(cs.N, cs.C * cs.H * cs.W);
        Matrix y = pool.forward(X);

        P_CHECK(y.rows() == cs.N && y.cols() == expCols,
                "[%s] 形状：期望 %zux%zu，实得 %zux%zu",
                cs.tag, cs.N, expCols, y.rows(), y.cols());
        if (y.rows() != cs.N || y.cols() != expCols) continue;

        size_t bad = 0;
        for (size_t n = 0; n < cs.N; ++n)
            for (size_t c = 0; c < cs.C; ++c)
                for (size_t oh = 0; oh < OH; ++oh)
                    for (size_t ow = 0; ow < OW; ++ow) {
                        const double want = windowMax(X, n, c, cs.H, cs.W, oh, ow, cs.K, cs.S);
                        const double got = y(n, c * OH * OW + oh * OW + ow);
                        if (!close(got, want)) {
                            ++bad;
                            if (bad <= 3)
                                printf("  [FAIL] [%s] n=%zu c=%zu oh=%zu ow=%zu：期望 %.6f，实得 %.6f\n",
                                       cs.tag, n, c, oh, ow, want, got);
                        }
                    }
        if (bad) { ++fails(); printf("  [FAIL] [%s]：%zu 个格子不对\n", cs.tag, bad); }
    }
}

// ---------------------------------------------------------------------------
// P4：argmax 路由。只在一个输出位置给 5 的梯度，它必须只落到该窗口的最大值像素
//   4x4 图 1..16，窗口(0,0)={1,2,5,6} 的最大值 6 在 (1,1) 即线性下标 5
// ---------------------------------------------------------------------------
inline void p4_argmax_routing()
{
    printf("[P4] argmax 路由：梯度只落到最大值像素\n");

    MaxPool2D pool(4, 4, 1, 2, 2);
    Matrix X(1, 16);
    for (size_t i = 0; i < 16; ++i) X(0, i) = (double)(i + 1);

    Matrix y = pool.forward(X);
    (void)y;

    Matrix g = Matrix::zeroMatrix(1, 4);
    g(0, 0) = 5.0;                       // 只给第一个输出位置
    Matrix dX = pool.backward(g);

    P_CHECK(dX.rows() == 1 && dX.cols() == 16, "形状：期望 1x16，实得 %zux%zu", dX.rows(), dX.cols());
    if (dX.rows() != 1 || dX.cols() != 16) return;

    P_CHECK(close(dX(0, 5), 5.0), "最大值位置 (1,1)=下标5 应得 5.0，实得 %.6f", dX(0, 5));
    size_t nonzero = 0;
    for (size_t i = 0; i < 16; ++i)
        if (!close(dX(0, i), 0.0, 1e-12)) {
            ++nonzero;
            if (i != 5)
                printf("  [FAIL] 下标 %zu 不该有梯度，实得 %.6f\n", i, dX(0, i));
        }
    P_CHECK(nonzero == 1, "只应有 1 个非零位置，实得 %zu 个", nonzero);
}

// ---------------------------------------------------------------------------
// P5：重叠累加。4x4 K=3 S=1，把全局最大值放在中心 (1,1)，则 4 个窗口的 argmax
//     都是它 —— 梯度必须累加成 4 倍。这一条专门抓「写成赋值而不是 +=」
// ---------------------------------------------------------------------------
inline void p5_overlap_accumulate()
{
    printf("[P5] 重叠累加：中心像素被 4 个窗口同时选中，梯度应累加成 4 倍\n");

    MaxPool2D pool(4, 4, 1, 3, 1);
    Matrix X = Matrix::zeroMatrix(1, 16);
    X(0, 5) = 9.0;                        // (1,1) 是全局最大，其余为 0

    Matrix y = pool.forward(X);
    P_CHECK(y.rows() == 1 && y.cols() == 4, "形状：期望 1x4，实得 %zux%zu", y.rows(), y.cols());
    if (y.rows() != 1 || y.cols() != 4) return;
    for (size_t i = 0; i < 4; ++i)
        P_CHECK(close(y(0, i), 9.0), "输出[%zu]：期望 9.0，实得 %.1f", i, y(0, i));

    Matrix g(1, 4);
    for (size_t i = 0; i < 4; ++i) g(0, i) = 1.0;
    Matrix dX = pool.backward(g);

    if (dX.rows() != 1 || dX.cols() != 16) {
        P_CHECK(false, "形状：期望 1x16，实得 %zux%zu", dX.rows(), dX.cols());
        return;
    }
    P_CHECK(close(dX(0, 5), 4.0),
            "中心像素应累加到 4.0（4 个窗口各贡献 1），实得 %.6f"
            "——若为 1.0 说明 backward 写成了赋值而非 +=", dX(0, 5));
    for (size_t i = 0; i < 16; ++i)
        if (i != 5 && !close(dX(0, i), 0.0, 1e-12))
            printf("  [FAIL] 下标 %zu 不该有梯度，实得 %.6f\n", i, dX(0, i));
}

// ---------------------------------------------------------------------------
// P6：数值梯度。L = <pool(X), G>，所以 dL/dX 就是 backward(G)。
//     注意顺序：先取解析梯度，再做差分（因为 forward 会覆盖 argmax 缓存）。
// ---------------------------------------------------------------------------
inline void p6_numeric_gradient()
{
    printf("[P6] 数值梯度：中心差分对拍\n");

    const size_t H = 4, W = 4, C = 2, K = 2, S = 2, N = 2;
    const size_t OH = outDim(H, K, S), OW = outDim(W, K, S);

    MaxPool2D pool(H, W, C, K, S);
    Matrix X = Matrix::randnMatrix(N, C * H * W);
    Matrix G = Matrix::randnMatrix(N, C * OH * OW);

    Matrix y0 = pool.forward(X);              // 必须 forward 才有 argmax 缓存
    (void)y0;
    Matrix dX = pool.backward(G);             // 先拿解析梯度

    auto objective = [&](const Matrix& x) {
        MaxPool2D p(H, W, C, K, S);
        Matrix y = p.forward(x);
        double s = 0.0;
        for (size_t i = 0; i < y.rows(); ++i)
            for (size_t j = 0; j < y.cols(); ++j) s += y(i, j) * G(i, j);
        return s;
    };

    const double eps = 1e-5;
    double maxRel = 0.0;
    size_t bad = 0;
    for (size_t i = 0; i < X.rows(); ++i)
        for (size_t j = 0; j < X.cols(); ++j) {
            Matrix xp = X; xp(i, j) += eps;
            Matrix xm = X; xm(i, j) -= eps;
            const double num = (objective(xp) - objective(xm)) / (2 * eps);
            const double ana = dX(i, j);
            const double rel = std::fabs(num - ana);
            maxRel = std::max(maxRel, rel);
            if (rel > 1e-4) {
                ++bad;
                if (bad <= 4)
                    printf("  [FAIL] X(%zu,%zu)：数值 %.6f，解析 %.6f\n", i, j, num, ana);
            }
        }
    P_CHECK(bad == 0, "数值梯度有 %zu 处不符（最大绝对差 %.2e）", bad, maxRel);
}

inline int runMaxPoolTests()
{
    fails() = 0;
    printf("\n===== 最大池化测试开始 =====\n");
    p1_hand_computed();
    p2_shapes_with_reference();
    p4_argmax_routing();
    p5_overlap_accumulate();
    p6_numeric_gradient();
    printf("===== 最大池化测试结束：失败 %d 条 =====\n\n", fails());
    return fails();
}

} // namespace pooltest
