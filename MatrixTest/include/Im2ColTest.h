#pragma once
// ============================================================================
//  Im2ColTest.h —— im2col 的正确性测试（只管对错，不管快慢）
//
//  【测试认定的规格】im2col 的输出形状与索引：
//      A 的形状 = ( N * OH * OW , C * K * K )         K = ksize
//          OH = (H - K) / stride + 1
//          OW = (W - K) / stride + 1
//      行索引  row = (n * OH + oh) * OW + ow          ——「哪个输出位置」
//      列索引  col = c * K * K + kh * K + kw          ——「哪个求和指标」
//      即行方向是 (n, oh, ow)，列方向是 (c, kh, kw)：
//      输入通道 c 在最外层，核内 (kh, kw) 在最内层。
//
//      「输出位置」和「求和指标」这两组的内部顺序是规格，不是对现有实现的
//      解释。改规格必须同时改这个文件；不一致的症状是「看着像对的、数值
//      微妙地错」，所以 T1/T2 的期望值是手算硬编码的，不看实现脸色。
//
//  A 的行主序 = [n][oh][ow][c][kh][kw]，所以 im2col 的循环按
//      n -> oh -> ow -> c -> kh -> kw
//  排列时，写 A 是纯顺序写（步长 1）；这也是选这个顺序的主要理由。
//
//  【测试清单】
//      T1  4x4 单通道 3x3 stride1：期望值手算硬编码，钉死行列布局
//      T2  双通道：检查 c 分块偏移恰好是 K*K
//      T3  五种形状逐元素对拍：每个格子是否等于原图对应像素
//      T4  C=1 的完整通路：A(4x9) x W(1x9)^T == 手算卷积
//      T5  C>1 的完整通路：A(M, C*K*K) x W(OC, C*K*K)^T vs 朴素卷积
//
//  用法：在 main.cpp 里 #include "Im2ColTest.h"，然后调用 runIm2ColTests()。
//        返回值 = 失败条数，全绿是 0。
//  ============================================================================

#include "TestCommon.h"

namespace im2col_test {

// ---------------------------------------------------------------------------
// 小工具（im2col 是纯搬运，所以比较用严格相等 dstest::exact）
// ---------------------------------------------------------------------------
inline size_t outH(size_t H, size_t K, size_t stride) { return (H - K) / stride + 1; }
inline size_t outW(size_t W, size_t K, size_t stride) { return (W - K) / stride + 1; }

// 按【规格】算行索引 / 列索引。实现必须满足它。
inline size_t rowIdx(size_t OH, size_t OW, size_t n, size_t oh, size_t ow)
{
    return (n * OH + oh) * OW + ow;
}
inline size_t colIdx(size_t K, size_t c, size_t kh, size_t kw)
{
    return c * K * K + kh * K + kw;
}

// ---------------------------------------------------------------------------
// T1 / T2 用的期望值：4x4 单通道，核 3x3，stride 1
//   图（行主序）:  1  2  3  4
//                  5  6  7  8
//                  9  0  1  2
//                  3  4  5  6
//   A 的 4 行依次是 4 个 patch（按 oh、ow 展开），每行 9 个数：
//       row0 = patch(0,0) = a       row1 = patch(0,1) = b
//       row2 = patch(1,0) = c       row3 = patch(1,1) = d
//   也就是 main.cpp 注释里的 a, b, c, d 四行。
// ---------------------------------------------------------------------------
inline const double* expectPatches()   // 4 x 9 = 36 个数，行主序
{
    static const double v[36] = {
        1, 2, 3, 5, 6, 7, 9, 0, 1,      // a = patch(0,0)
        2, 3, 4, 6, 7, 8, 0, 1, 2,      // b = patch(0,1)
        5, 6, 7, 9, 0, 1, 3, 4, 5,      // c = patch(1,0)
        6, 7, 8, 0, 1, 2, 4, 5, 6       // d = patch(1,1)
    };
    return v;
}

// 造出 T1/T2 用的那张 4x4 图（值 = (i+1) % 10）
inline Matrix makeImage4x4()
{
    Matrix im(1, 16);
    for (size_t i = 0; i < 16; ++i)
        im(0, i) = (double)((i + 1) % 10);
    return im;
}

// ---------------------------------------------------------------------------
// T1：4x4 单通道 3x3 stride1 —— 形状与手算期望
// ---------------------------------------------------------------------------
inline void t1_layout_single_channel()
{
    printf("[T1] 4x4 单通道 3x3 stride1：形状 (4x9) 与手算期望\n");

    Matrix im = makeImage4x4();
    Matrix A = im.im2col(1, 1, 4, 4, 3, 1);

    T_CHECK(A.rows() == 4 && A.cols() == 9,
            "形状：期望 4x9（行 = 输出位置，列 = 求和指标），实得 %zux%zu",
            A.rows(), A.cols());
    if (A.rows() != 4 || A.cols() != 9) return;

    const double* e = expectPatches();
    size_t bad = 0;
    for (size_t r = 0; r < 4; ++r)
        for (size_t k = 0; k < 9; ++k) {
            const double want = e[r * 9 + k];
            if (!dstest::exact(A(r, k), want)) {
                ++bad;
                if (bad <= 6)
                    printf("  [FAIL] A(%zu,%zu)：期望 %.1f，实得 %.1f\n",
                           r, k, want, A(r, k));
            }
        }
    if (bad) {
        ++dstest::fails();
        printf("  [FAIL] T1 行列布局：36 个格子里有 %zu 个不等于期望值\n", bad);
        for (size_t r = 0; r < 4; ++r) {
            printf("         第%zu行实得：", r);
            for (size_t k = 0; k < 9; ++k) printf("%.0f ", A(r, k));
            printf("\n");
        }
    }
}

// ---------------------------------------------------------------------------
// T2：双通道，检查 c 方向的分块偏移是不是 K*K（不是 OH*OW*K*K）
//     第二通道 = 第一通道 + 100
// ---------------------------------------------------------------------------
inline void t2_channel_offset()
{
    printf("[T2] 双通道 4x4 3x3 stride1：c 分块偏移 = K*K = 9\n");

    Matrix im = makeImage4x4();
    Matrix two(1, 32);
    for (size_t i = 0; i < 16; ++i) {
        two(0, i) = im(0, i);               // c = 0
        two(0, 16 + i) = im(0, i) + 100.0;  // c = 1
    }

    Matrix A = two.im2col(1, 2, 4, 4, 3, 1);
    T_CHECK(A.rows() == 4 && A.cols() == 18,
            "形状：期望 4x18（列 = C*K*K = 18），实得 %zux%zu", A.rows(), A.cols());
    if (A.rows() != 4 || A.cols() != 18) return;

    const double* e = expectPatches();   // 仍是 4 x 9，按 (c, 行) 取
    size_t bad0 = 0, bad1 = 0;
    for (size_t r = 0; r < 4; ++r)
        for (size_t k = 0; k < 9; ++k) {
            const double want0 = e[r * 9 + k];
            if (!dstest::exact(A(r, 0 * 9 + k), want0))      ++bad0;
            if (!dstest::exact(A(r, 1 * 9 + k), want0 + 100.0)) ++bad1;
        }
    if (bad0) {
        ++dstest::fails();
        printf("  [FAIL] T2 通道 0 的块（列 0..8）有 %zu 个格子不对\n", bad0);
    }
    if (bad1) {
        ++dstest::fails();
        printf("  [FAIL] T2 通道 1 的块（列 9..17）有 %zu 个格子不对"
               "（说明 c 的分块偏移不是 K*K = 9）\n", bad1);
    }
}

// ---------------------------------------------------------------------------
// T3：五种形状逐元素对拍。
//     参照值直接从原图取：X(n, c*H*W + (oh*stride+kh)*W + (ow*stride+kw))
//     这条参照完全不碰 im2col 的循环，所以是真·独立参照。
// ---------------------------------------------------------------------------
inline void t3_elementwise_many_shapes()
{
    printf("[T3] 多形状逐元素对拍（参照 = 直接从原图取像素）\n");

    struct Case { size_t N, C, H, W, K, stride; };
    const Case cases[] = {
        { 2, 1,  4,  4, 3, 1 },
        { 1, 3,  5,  5, 3, 2 },
        { 2, 2,  7,  7, 3, 1 },
        { 1, 1, 28, 28, 5, 2 },
        { 2, 1,  9,  7, 4, 2 },
    };

    for (const Case& cs : cases) {
        Matrix X = Matrix::randnMatrix(cs.N, cs.C * cs.H * cs.W);
        Matrix A = X.im2col(cs.N, cs.C, cs.H, cs.W, cs.K, cs.stride);

        const size_t OH = outH(cs.H, cs.K, cs.stride);
        const size_t OW = outW(cs.W, cs.K, cs.stride);
        const size_t expRows = cs.N * OH * OW;
        const size_t expCols = cs.C * cs.K * cs.K;

        T_CHECK(A.rows() == expRows && A.cols() == expCols,
                "形状 N=%zu C=%zu H=%zu W=%zu K=%zu s=%zu：期望 %zux%zu，实得 %zux%zu",
                cs.N, cs.C, cs.H, cs.W, cs.K, cs.stride,
                expRows, expCols, A.rows(), A.cols());
        if (A.rows() != expRows || A.cols() != expCols) continue;

        size_t bad = 0;
        for (size_t n = 0; n < cs.N; ++n)
        for (size_t oh = 0; oh < OH; ++oh)
        for (size_t ow = 0; ow < OW; ++ow)
        for (size_t c = 0; c < cs.C; ++c)
        for (size_t kh = 0; kh < cs.K; ++kh)
        for (size_t kw = 0; kw < cs.K; ++kw) {
            const size_t r = rowIdx(OH, OW, n, oh, ow);
            const size_t col = colIdx(cs.K, c, kh, kw);
            const size_t src = c * cs.H * cs.W
                             + (oh * cs.stride + kh) * cs.W
                             + (ow * cs.stride + kw);
            const double want = X(n, src);
            if (!dstest::exact(A(r, col), want)) {
                ++bad;
                if (bad <= 4)
                    printf("  [FAIL] n=%zu oh=%zu ow=%zu c=%zu kh=%zu kw=%zu"
                           " -> A(%zu,%zu)：期望 %.6f，实得 %.6f\n",
                           n, oh, ow, c, kh, kw, r, col, want, A(r, col));
            }
        }
        if (bad) {
            ++dstest::fails();
            printf("  [FAIL] 形状 N=%zu C=%zu H=%zu W=%zu K=%zu s=%zu：%zu 个格子不对\n",
                   cs.N, cs.C, cs.H, cs.W, cs.K, cs.stride, bad);
        }
    }
}

// ---------------------------------------------------------------------------
// 朴素卷积（独立参照，不碰 im2col）
//   X  : (N, C*H*W)      每行是一张图，行主序 (c, h, w)
//   Wt : (OC, C*K*K)     每个输出通道一行，行主序 (c, kh, kw)
//   返回 (N, OC*OH*OW)   行主序 (oc, oh, ow)
// ---------------------------------------------------------------------------
inline Matrix naiveConv2D(const Matrix& X, const Matrix& Wt,
                          size_t N, size_t C, size_t H, size_t W,
                          size_t OC, size_t K, size_t stride)
{
    const size_t OH = outH(H, K, stride);
    const size_t OW = outW(W, K, stride);
    Matrix out(N, OC * OH * OW);
    for (size_t n = 0; n < N; ++n)
    for (size_t oc = 0; oc < OC; ++oc)
    for (size_t oh = 0; oh < OH; ++oh)
    for (size_t ow = 0; ow < OW; ++ow) {
        double s = 0.0;
        for (size_t c = 0; c < C; ++c)
        for (size_t kh = 0; kh < K; ++kh)
        for (size_t kw = 0; kw < K; ++kw)
            s += X(n, c * H * W + (oh * stride + kh) * W + (ow * stride + kw))
               * Wt(oc, c * K * K + kh * K + kw);
        out(n, oc * OH * OW + oh * OW + ow) = s;
    }
    return out;
}

// ---------------------------------------------------------------------------
// T4：C=1 的完整通路。A(4x9) x W(1x9)^T -> out(4x1)
//     卷积核按 (O, I*K*K) 存，正好喂给 matmul_nt，不用真的转置。
// ---------------------------------------------------------------------------
inline void t4_pipeline_single_channel()
{
    printf("[T4] C=1 通路：matmul_nt(A(4x9), W(1x9)) -> out(4x1)\n");

    Matrix im = makeImage4x4();
    Matrix Wk(1, 9);
    for (size_t i = 0; i < 9; ++i) Wk(0, i) = (double)(i + 1);   // 1..9

    Matrix A = im.im2col(1, 1, 4, 4, 3, 1);
    T_CHECK(A.rows() == 4 && A.cols() == 9,
            "形状：期望 4x9，实得 %zux%zu", A.rows(), A.cols());
    if (A.rows() != 4 || A.cols() != 9) return;

    Matrix out(4, 1);                    // 零初始化；nt 是累加语义，必须为零
    Matrix::matmul_nt(A, Wk, out);

    // 手算：patch·kernel，四个位置依次 178, 153, 178, 183
    const double want[4] = { 178.0, 153.0, 178.0, 183.0 };
    for (size_t i = 0; i < 4; ++i)
        T_CHECK(dstest::close(out(i, 0), want[i]),
                "输出[%zu]：期望 %.1f，实得 %.1f", i, want[i], out(i, 0));

    // 再用随机数据与朴素卷积对拍一次
    Matrix X  = Matrix::randnMatrix(1, 16);
    Matrix Wr = Matrix::randnMatrix(1, 9);
    Matrix A2 = X.im2col(1, 1, 4, 4, 3, 1);
    Matrix o2(4, 1);
    Matrix::matmul_nt(A2, Wr, o2);
    Matrix ref = naiveConv2D(X, Wr, 1, 1, 4, 4, 1, 3, 1);
    for (size_t i = 0; i < 4; ++i)
        T_CHECK(dstest::close(o2(i, 0), ref(0, i)),
                "随机数据输出[%zu]：朴素卷积 %.6f，通路 %.6f", i, ref(0, i), o2(i, 0));
}

// ---------------------------------------------------------------------------
// T5：C>1 的通路。A(M, C*K*K) x W(OC, C*K*K)^T -> out(M, OC)
//     注意 out 的行内是 o 连续（NHWC 语义）；要变 NCHW 得另外搬运。
// ---------------------------------------------------------------------------
inline void t5_pipeline_multi_channel()
{
    printf("[T5] C=3 OC=2 通路：matmul_nt(A(M,27), W(2,27)) -> out(M,2) vs 朴素卷积\n");

    const size_t N = 2, C = 3, H = 5, W = 5, K = 3, stride = 2, OC = 2;
    const size_t OH = outH(H, K, stride), OW = outW(W, K, stride);
    const size_t M = N * OH * OW;
    const size_t V = C * K * K;

    Matrix X  = Matrix::randnMatrix(N, C * H * W);
    Matrix Wt = Matrix::randnMatrix(OC, V);

    Matrix A = X.im2col(N, C, H, W, K, stride);
    T_CHECK(A.rows() == M && A.cols() == V,
            "形状：期望 %zux%zu，实得 %zux%zu", M, V, A.rows(), A.cols());
    if (A.rows() != M || A.cols() != V) return;

    Matrix out(M, OC);
    Matrix::matmul_nt(A, Wt, out);

    Matrix ref = naiveConv2D(X, Wt, N, C, H, W, OC, K, stride);

    double maxErr = 0.0;
    for (size_t n = 0; n < N; ++n)
    for (size_t oh = 0; oh < OH; ++oh)
    for (size_t ow = 0; ow < OW; ++ow)
    for (size_t oc = 0; oc < OC; ++oc) {
        const size_t r = rowIdx(OH, OW, n, oh, ow);
        // 升到 double 再比：float 版 std::fabs 返回 float，喂给 std::max(double, float) 编不过
        const double d = std::fabs((double)out(r, oc) - (double)ref(n, oc * OH * OW + oh * OW + ow));
        maxErr = std::max(maxErr, d);
    }
    T_CHECK(maxErr < dstest::kFloatTol, "最大误差 %.3e（应为 0）", maxErr);
}

// ---------------------------------------------------------------------------
// T6：padding 版。4x4 输入、K=3、stride=1、padding=1 —— same padding，输出仍是 4x4
//   padded 之后是 6x6（上下左右各补一条 0）：
//       0 0 0 0 0 0
//       0 1 2 3 4 0
//       0 5 6 7 8 0
//       0 9 0 1 2 0
//       0 3 4 5 6 0
//       0 0 0 0 0 0
//   所以 (oh,ow)=(0,0) 那一行是左上角含三个 0 的窗口，
//   而 (1,1) 那一行恰好又是没 padding 时的 patch(0,0)。
// ---------------------------------------------------------------------------
inline void t6_padding_same()
{
    printf("[T6] padding 版：4x4 K=3 s=1 p=1（same padding）\n");

    Matrix im = makeImage4x4();
    Matrix A = im.im2col(1, 1, 4, 4, 3, 1, 1);

    T_CHECK(A.rows() == 16 && A.cols() == 9,
            "形状：期望 16x9（OH=OW=4），实得 %zux%zu", A.rows(), A.cols());
    if (A.rows() != 16 || A.cols() != 9) return;

    static const double r0[9] = { 0,0,0, 0,1,2, 0,5,6 };   // (oh,ow)=(0,0)
    static const double r1[9] = { 0,0,0, 1,2,3, 5,6,7 };   // (0,1)
    static const double r4[9] = { 0,1,2, 0,5,6, 0,9,0 };   // (1,0)
    static const double r5[9] = { 1,2,3, 5,6,7, 9,0,1 };   // (1,1) == 无 padding 的 patch(0,0)
    const double* want[4] = { r0, r1, r4, r5 };
    const size_t  rows_[4] = { 0, 1, 4, 5 };

    for (size_t t = 0; t < 4; ++t)
        for (size_t k = 0; k < 9; ++k)
            T_CHECK(dstest::exact(A(rows_[t], k), want[t][k]),
                    "A(%zu,%zu)：期望 %.0f，实得 %.0f",
                    rows_[t], k, want[t][k], A(rows_[t], k));
}

// ---------------------------------------------------------------------------
// T7：padding 多形状扫描。
//   参照用【有符号坐标】独立判断越界，不抄实现里的 size_t 写法——只有这样才能
//   抓到「先减后加导致无符号回绕」这类边界错。含 W < K 靠 padding 补足的退化例。
// ---------------------------------------------------------------------------
inline void t7_padding_many_shapes()
{
    printf("[T7] padding 多形状扫描（参照 = 有符号坐标独立判断）\n");

    struct Case { size_t N, C, H, W, K, stride, padding; };
    const Case cases[] = {
        { 1, 1,  4,  4, 3, 1, 0 },   // p=0 应与无 padding 版行为一致
        { 1, 1,  4,  4, 3, 1, 1 },   // same
        { 2, 3,  5,  5, 3, 2, 1 },
        { 1, 2,  7,  7, 3, 1, 2 },
        { 1, 1,  2,  2, 3, 1, 1 },   // W < K，靠 padding 补足
        { 1, 1,  1,  1, 3, 1, 1 },   // 极端退化：窗口里只有 1 个真实像素
        { 1, 1, 28, 28, 5, 2, 2 },
    };

    for (const Case& cs : cases) {
        const int64_t OH64 = ((int64_t)cs.H + 2 * (int64_t)cs.padding - (int64_t)cs.K) / (int64_t)cs.stride + 1;
        const int64_t OW64 = ((int64_t)cs.W + 2 * (int64_t)cs.padding - (int64_t)cs.K) / (int64_t)cs.stride + 1;
        const size_t OH = (size_t)OH64, OW = (size_t)OW64;
        const size_t M = cs.N * OH * OW, V = cs.C * cs.K * cs.K;

        Matrix X = Matrix::randnMatrix(cs.N, cs.C * cs.H * cs.W);
        Matrix A = X.im2col(cs.N, cs.C, cs.H, cs.W, cs.K, cs.stride, cs.padding);

        T_CHECK(A.rows() == M && A.cols() == V,
                "形状 N=%zu C=%zu H=%zu W=%zu K=%zu s=%zu p=%zu：期望 %zux%zu，实得 %zux%zu",
                cs.N, cs.C, cs.H, cs.W, cs.K, cs.stride, cs.padding, M, V, A.rows(), A.cols());
        if (A.rows() != M || A.cols() != V) continue;

        size_t bad = 0;
        for (size_t n = 0; n < cs.N; ++n)
        for (size_t oh = 0; oh < OH; ++oh)
        for (size_t ow = 0; ow < OW; ++ow)
        for (size_t c = 0; c < cs.C; ++c)
        for (size_t kh = 0; kh < cs.K; ++kh)
        for (size_t kw = 0; kw < cs.K; ++kw) {
            const int64_t h = (int64_t)(oh * cs.stride + kh) - (int64_t)cs.padding;
            const int64_t w = (int64_t)(ow * cs.stride + kw) - (int64_t)cs.padding;
            const bool outside = (h < 0 || h >= (int64_t)cs.H || w < 0 || w >= (int64_t)cs.W);
            const double want = outside
                ? 0.0
                : X(n, c * cs.H * cs.W + (size_t)h * cs.W + (size_t)w);

            const size_t r = rowIdx(OH, OW, n, oh, ow);
            const size_t col = colIdx(cs.K, c, kh, kw);
            if (!dstest::exact(A(r, col), want)) {
                ++bad;
                if (bad <= 4)
                    printf("  [FAIL] n=%zu oh=%zu ow=%zu c=%zu kh=%zu kw=%zu"
                           "（padded 坐标 h=%lld w=%lld）期望 %.6f，实得 %.6f\n",
                           n, oh, ow, c, kh, kw, (long long)h, (long long)w, want, A(r, col));
            }
        }
        if (bad) {
            ++dstest::fails();
            printf("  [FAIL] 形状 N=%zu C=%zu H=%zu W=%zu K=%zu s=%zu p=%zu：%zu 个格子不对\n",
                   cs.N, cs.C, cs.H, cs.W, cs.K, cs.stride, cs.padding, bad);
        }
    }
}

// ---------------------------------------------------------------------------
// 入口
// ---------------------------------------------------------------------------
inline int runIm2ColTests()
{
    dstest::resetFails();
    printf("\n===== im2col 测试开始 =====\n");
    t1_layout_single_channel();
    t2_channel_offset();
    t3_elementwise_many_shapes();
    t4_pipeline_single_channel();
    t5_pipeline_multi_channel();
    t6_padding_same();
    t7_padding_many_shapes();
    printf("===== im2col 测试结束：失败 %d 条 =====\n\n", dstest::fails());
    return dstest::fails();
}

} // namespace im2col_test
