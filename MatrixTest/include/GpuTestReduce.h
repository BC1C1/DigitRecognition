#pragma once
// ============================================================================
//  GpuTestReduce.h —— 归约算子的对拍
//
//  这三个最容易错的地方不是"和算得对不对"，而是**输出长度**：
//      launchSumRows(ga, gb, M, N)  输出长度 M —— 每行一个和（列向量）
//      launchSumCols(ga, gb, M, N)  输出长度 N —— 每列一个和（行向量）
//
//  这两个的名字是按**结果的形状**取的，不是按"沿哪个方向求和"。所以
//  sumRows 里线程数按 M 算、sumCols 里按 N 算 —— 一旦写反，输出长度就错，
//  而调用方（Conv2D / Linear 求 db）拿到的是一根长度不对的向量，
//  短了是结果错、长了就是**显存越界写**。所以这里对长度的检查是重点：
//  除了对拍数值，还要确认输出缓冲的头尾都没被多写。
//
//  【测试清单】
//      R1  launchSumRows  长度 M + 数值
//      R2  launchSumCols  长度 N + 数值
//      R3  launchSumKernel 全元素和（double）
//      R4  长度敏感性：M != N 且都跨过 256 线程边界
// ============================================================================

#include "GpuTestCommon.h"

namespace gputest {

// ---------------------------------------------------------------------------
// R1：sumRows —— 输出长度 M，每个 = 该行之和
// ---------------------------------------------------------------------------
inline void r1_sum_rows()
{
    printf("[R1] launchSumRows（输出长度应为 M）\n");

    struct Shape { size_t M, N; };
    const Shape shapes[] = { {1, 1}, {1, 7}, {7, 1}, {16, 16}, {17, 19}, {100, 80}, {300, 5} };

    for (const Shape& s : shapes) {
        Matrix a = makeMatrix(s.M, s.N, 501);
        Matrix out(s.M, 1);
        // 输出缓冲刻意只给 M 个元素，多写一个就会踩到后面的显存
        upload(out, std::vector<float>(s.M, -999.0f));

        launchSumRows(a.data_gpu(), out.data_gpu(), s.M, s.N);
        if (!runAndCheck("launchSumRows")) {
            G_CHECK(false, "M=%zu N=%zu kernel 失败", s.M, s.N);
            continue;
        }

        const std::vector<float> ha = download(a);
        const std::vector<float> got = download(out);

        std::vector<float> want(s.M, 0.0f);
        for (size_t i = 0; i < s.M; ++i) {
            double acc = 0.0;
            for (size_t j = 0; j < s.N; ++j) acc += (double)ha[i * s.N + j];
            want[i] = (float)acc;
        }

        const double e = maxDiff(got, want);
        G_CHECK(e < kFloatTol, "M=%zu N=%zu 最大误差 %.3e", s.M, s.N, e);

        size_t leftover = 0;
        for (float x : got) if (std::fabs(x + 999.0f) < kExactTol) ++leftover;
        G_CHECK(leftover == 0, "M=%zu N=%zu 有 %zu 个输出位置没被写", s.M, s.N, leftover);
    }
}

// ---------------------------------------------------------------------------
// R2：sumCols —— 输出长度 N，每个 = 该列之和
// ---------------------------------------------------------------------------
inline void r2_sum_cols()
{
    printf("[R2] launchSumCols（输出长度应为 N）\n");

    struct Shape { size_t M, N; };
    const Shape shapes[] = { {1, 1}, {1, 7}, {7, 1}, {16, 16}, {19, 17}, {80, 100}, {5, 300} };

    for (const Shape& s : shapes) {
        Matrix a = makeMatrix(s.M, s.N, 601);
        Matrix out(1, s.N);
        upload(out, std::vector<float>(s.N, -999.0f));

        launchSumCols(a.data_gpu(), out.data_gpu(), s.M, s.N);
        if (!runAndCheck("launchSumCols")) {
            G_CHECK(false, "M=%zu N=%zu kernel 失败", s.M, s.N);
            continue;
        }

        const std::vector<float> ha = download(a);
        const std::vector<float> got = download(out);

        std::vector<float> want(s.N, 0.0f);
        for (size_t j = 0; j < s.N; ++j) {
            double acc = 0.0;
            for (size_t i = 0; i < s.M; ++i) acc += (double)ha[i * s.N + j];
            want[j] = (float)acc;
        }

        const double e = maxDiff(got, want);
        G_CHECK(e < kFloatTol, "M=%zu N=%zu 最大误差 %.3e", s.M, s.N, e);

        size_t leftover = 0;
        for (float x : got) if (std::fabs(x + 999.0f) < kExactTol) ++leftover;
        G_CHECK(leftover == 0, "M=%zu N=%zu 有 %zu 个输出位置没被写", s.M, s.N, leftover);
    }
}

// ---------------------------------------------------------------------------
// R3：全元素和。kernel 内部会自己 malloc / memset / D2H / free，
//     返回的是 host 上的 double —— 这一步本身就是一个同步点。
// ---------------------------------------------------------------------------
inline void r3_sum_kernel()
{
    printf("[R3] launchSumKernel（全元素和，返回 double）\n");

    const size_t sizes[] = { 1, 17, 256, 1000, 4096, 100000 };

    for (size_t n : sizes) {
        Matrix a = makeMatrix(1, n, 701);
        const std::vector<float> ha = download(a);

        double want = 0.0;
        for (float x : ha) want += (double)x;

        const double got = launchSumKernel(a.data_gpu(), n);
        if (!runAndCheck("launchSumKernel")) {
            G_CHECK(false, "n=%zu kernel 失败", n);
            continue;
        }

        // 两边都是 double 累加，但顺序不同；给一个相对宽松的绝对容差
        const double e = std::fabs(got - want);
        const double tol = 1e-3 * std::max(1.0, std::fabs(want));
        G_CHECK(e < tol, "n=%zu 求和 %.6f，期望 %.6f，差 %.3e", n, got, want, e);
    }
}

// ---------------------------------------------------------------------------
// R4：长度敏感性。M 和 N 都跨过 256 线程边界，而且一大一小 ——
//     如果两个 kernel 的 grid 用错了维度，这里一定红，比单个形状更灵敏。
// ---------------------------------------------------------------------------
inline void r4_shape_sensitivity()
{
    printf("[R4] 长度敏感性：M/N 都跨 256 边界且一大一小\n");

    struct Shape { size_t M, N; };
    const Shape shapes[] = { {257, 3}, {3, 257}, {300, 260}, {260, 300}, {1000, 7}, {7, 1000} };

    for (const Shape& s : shapes) {
        Matrix a = makeMatrix(s.M, s.N, 801);
        const std::vector<float> ha = download(a);

        Matrix rows(s.M, 1);
        Matrix cols(1, s.N);
        launchSumRows(a.data_gpu(), rows.data_gpu(), s.M, s.N);
        launchSumCols(a.data_gpu(), cols.data_gpu(), s.M, s.N);
        if (!runAndCheck("launchSumRows/Cols")) {
            G_CHECK(false, "M=%zu N=%zu kernel 失败", s.M, s.N);
            continue;
        }

        const std::vector<float> gr = download(rows), gc = download(cols);

        std::vector<float> wantR(s.M, 0.0f), wantC(s.N, 0.0f);
        for (size_t i = 0; i < s.M; ++i)
            for (size_t j = 0; j < s.N; ++j) {
                wantR[i] += ha[i * s.N + j];
                wantC[j] += ha[i * s.N + j];
            }

        G_CHECK(maxDiff(gr, wantR) < kFloatTol, "M=%zu N=%zu sumRows 误差 %.3e",
                s.M, s.N, maxDiff(gr, wantR));
        G_CHECK(maxDiff(gc, wantC) < kFloatTol, "M=%zu N=%zu sumCols 误差 %.3e",
                s.M, s.N, maxDiff(gc, wantC));

        // 两条和相等：总量守恒。这一条不依赖任何一方的实现，是独立交叉验证
        double sumR = 0.0, sumC = 0.0;
        for (float x : gr) sumR += (double)x;
        for (float x : gc) sumC += (double)x;
        G_CHECK(std::fabs(sumR - sumC) < kFloatTol * std::max(1.0, std::fabs(sumR)),
                "M=%zu N=%zu 行和总和 %.6f 与列和总和 %.6f 不等", s.M, s.N, sumR, sumC);
    }
}

// ---------------------------------------------------------------------------
// 入口
// ---------------------------------------------------------------------------
inline int runReduceTests()
{
    printf("\n===== 归约算子 =====\n");
    r1_sum_rows();
    r2_sum_cols();
    r3_sum_kernel();
    r4_shape_sensitivity();
    return fails();
}

} // namespace gputest
