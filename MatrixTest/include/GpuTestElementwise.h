#pragma once
// ============================================================================
//  GpuTestElementwise.h —— 元素级算子的对拍
//
//  这几个 kernel 的形态都一样：一维线程网格，每个线程处理一个元素，
//  边界检查是 `if (i >= n) return;`。所以它们最值得测的不是"算得对不对"
//  （那太容易看出来了），而是**边界**：
//      n = 1              线程数远大于元素数
//      n = 255/256/257    正好跨 block 边界
//      n = 1000           多个 block
//  边界写错的表现不是崩，而是**最后一个 block 越界写**或者**末尾没写**，
//  后者看起来像"结果对了但有几个位置是旧值"。
//
//  另外 in-place 版本要额外钉一件事：**第二个操作数不能被修改**。
//  launchSubInPlace(ga, gb) 只该动 ga；哪天写成两个都动，这里是唯一会红的地方。
//
//  【测试清单】
//      E1  launchFillInplace
//      E2  launchAdd
//      E3  launchSub
//      E4  launchSubInPlace        含"gb 未被修改"
//      E5  launchScaleMul
//      E6  launchScaleMulInPlace
//      E7  launchElementwiseMul
//      E8  launchElementwiseMulInPlace  含"gb 未被修改"
// ============================================================================

#include "GpuTestCommon.h"

namespace gputest {

// 元素级测试共用的尺寸表：1 / 边界 / 跨块 / 多块
static const size_t kSizes[] = { 1, 17, 255, 256, 257, 1000, 4096 };

// ---------------------------------------------------------------------------
// E1：fill
// ---------------------------------------------------------------------------
inline void e1_fill()
{
    printf("[E1] launchFillInplace\n");

    for (size_t n : kSizes) {
        Matrix m = makeMatrix(1, n, 11);
        const float value = -3.25f;
        launchFillInplace(m.data_gpu(), n, value);
        if (!runAndCheck("launchFillInplace")) {
            G_CHECK(false, "n=%zu kernel 失败", n);
            continue;
        }
        const std::vector<float> got = download(m);
        double e = 0.0;
        for (float x : got) e = std::max(e, std::fabs((double)x - (double)value));
        G_CHECK(e < kExactTol, "n=%zu 最大误差 %.3e", n, e);
    }
}

// ---------------------------------------------------------------------------
// E2：a + b
// ---------------------------------------------------------------------------
inline void e2_add()
{
    printf("[E2] launchAdd\n");

    for (size_t n : kSizes) {
        Matrix a = makeMatrix(1, n, 21);
        Matrix b = makeMatrix(1, n, 22);
        Matrix c(1, n);

        // 预填一个哨兵值：如果 kernel 漏写末尾元素，这里会留下 -999
        std::vector<float> sentinel(n, -999.0f);
        upload(c, sentinel);

        launchAdd(a.data_gpu(), b.data_gpu(), c.data_gpu(), n);
        if (!runAndCheck("launchAdd")) { G_CHECK(false, "n=%zu kernel 失败", n); continue; }

        const std::vector<float> ha = download(a), hb = download(b), hc = download(c);
        std::vector<float> want(n);
        for (size_t i = 0; i < n; ++i) want[i] = ha[i] + hb[i];

        const double e = maxDiff(hc, want);
        G_CHECK(e < kFloatTol, "n=%zu 最大误差 %.3e（首个不同位置 %zu）",
                n, e, firstDiff(hc, want));
    }
}

// ---------------------------------------------------------------------------
// E3：a - b
// ---------------------------------------------------------------------------
inline void e3_sub()
{
    printf("[E3] launchSub\n");

    for (size_t n : kSizes) {
        Matrix a = makeMatrix(1, n, 31);
        Matrix b = makeMatrix(1, n, 32);
        Matrix c(1, n);
        upload(c, std::vector<float>(n, -999.0f));

        launchSub(a.data_gpu(), b.data_gpu(), c.data_gpu(), n);
        if (!runAndCheck("launchSub")) { G_CHECK(false, "n=%zu kernel 失败", n); continue; }

        const std::vector<float> ha = download(a), hb = download(b), hc = download(c);
        std::vector<float> want(n);
        for (size_t i = 0; i < n; ++i) want[i] = ha[i] - hb[i];

        const double e = maxDiff(hc, want);
        G_CHECK(e < kFloatTol, "n=%zu 最大误差 %.3e", n, e);
    }
}

// ---------------------------------------------------------------------------
// E4：a -= b（原地）。特别检查 b 没被动过。
// ---------------------------------------------------------------------------
inline void e4_sub_inplace()
{
    printf("[E4] launchSubInPlace（含：gb 不应被修改）\n");

    for (size_t n : kSizes) {
        Matrix a = makeMatrix(1, n, 41);
        Matrix b = makeMatrix(1, n, 42);

        const std::vector<float> ha = download(a), hb = download(b);

        launchSubInPlace(a.data_gpu(), b.data_gpu(), n);
        if (!runAndCheck("launchSubInPlace")) { G_CHECK(false, "n=%zu kernel 失败", n); continue; }

        const std::vector<float> ga = download(a), gb = download(b);

        std::vector<float> want(n);
        for (size_t i = 0; i < n; ++i) want[i] = ha[i] - hb[i];
        G_CHECK(maxDiff(ga, want) < kFloatTol, "n=%zu a 的结果不对，误差 %.3e",
                n, maxDiff(ga, want));

        // 第二个操作数必须原封不动
        G_CHECK(maxDiff(gb, hb) < kExactTol, "n=%zu 第二个操作数被修改了！误差 %.3e",
                n, maxDiff(gb, hb));
    }
}

// ---------------------------------------------------------------------------
// E5：a * value
// ---------------------------------------------------------------------------
inline void e5_scale_mul()
{
    printf("[E5] launchScaleMul\n");

    const float value = -1.75f;
    for (size_t n : kSizes) {
        Matrix a = makeMatrix(1, n, 51);
        Matrix b(1, n);
        upload(b, std::vector<float>(n, -999.0f));

        launchScaleMul(a.data_gpu(), b.data_gpu(), n, value);
        if (!runAndCheck("launchScaleMul")) { G_CHECK(false, "n=%zu kernel 失败", n); continue; }

        const std::vector<float> ha = download(a), hb = download(b);
        std::vector<float> want(n);
        for (size_t i = 0; i < n; ++i) want[i] = ha[i] * value;

        const double e = maxDiff(hb, want);
        G_CHECK(e < kFloatTol, "n=%zu 最大误差 %.3e", n, e);
    }
}

// ---------------------------------------------------------------------------
// E6：a *= value（原地）
// ---------------------------------------------------------------------------
inline void e6_scale_mul_inplace()
{
    printf("[E6] launchScaleMulInPlace\n");

    const float value = 2.5f;
    for (size_t n : kSizes) {
        Matrix a = makeMatrix(1, n, 61);
        const std::vector<float> ha = download(a);

        launchScaleMulInPlace(a.data_gpu(), n, value);
        if (!runAndCheck("launchScaleMulInPlace")) { G_CHECK(false, "n=%zu kernel 失败", n); continue; }

        const std::vector<float> ga = download(a);
        std::vector<float> want(n);
        for (size_t i = 0; i < n; ++i) want[i] = ha[i] * value;

        const double e = maxDiff(ga, want);
        G_CHECK(e < kFloatTol, "n=%zu 最大误差 %.3e", n, e);
    }
}

// ---------------------------------------------------------------------------
// E7：a .* b
// ---------------------------------------------------------------------------
inline void e7_elementwise_mul()
{
    printf("[E7] launchElementwiseMul\n");

    for (size_t n : kSizes) {
        Matrix a = makeMatrix(1, n, 71);
        Matrix b = makeMatrix(1, n, 72);
        Matrix c(1, n);
        upload(c, std::vector<float>(n, -999.0f));

        launchElementwiseMul(a.data_gpu(), b.data_gpu(), c.data_gpu(), n);
        if (!runAndCheck("launchElementwiseMul")) { G_CHECK(false, "n=%zu kernel 失败", n); continue; }

        const std::vector<float> ha = download(a), hb = download(b), hc = download(c);
        std::vector<float> want(n);
        for (size_t i = 0; i < n; ++i) want[i] = ha[i] * hb[i];

        const double e = maxDiff(hc, want);
        G_CHECK(e < kFloatTol, "n=%zu 最大误差 %.3e", n, e);
    }
}

// ---------------------------------------------------------------------------
// E8：a .*= b（原地）。同样检查 b 没被动过。
// ---------------------------------------------------------------------------
inline void e8_elementwise_mul_inplace()
{
    printf("[E8] launchElementwiseMulInPlace（含：gb 不应被修改）\n");

    for (size_t n : kSizes) {
        Matrix a = makeMatrix(1, n, 81);
        Matrix b = makeMatrix(1, n, 82);

        const std::vector<float> ha = download(a), hb = download(b);

        launchElementwiseMulInPlace(a.data_gpu(), b.data_gpu(), n);
        if (!runAndCheck("launchElementwiseMulInPlace")) { G_CHECK(false, "n=%zu kernel 失败", n); continue; }

        const std::vector<float> ga = download(a), gb = download(b);

        std::vector<float> want(n);
        for (size_t i = 0; i < n; ++i) want[i] = ha[i] * hb[i];
        G_CHECK(maxDiff(ga, want) < kFloatTol, "n=%zu a 的结果不对，误差 %.3e",
                n, maxDiff(ga, want));

        G_CHECK(maxDiff(gb, hb) < kExactTol, "n=%zu 第二个操作数被修改了！误差 %.3e",
                n, maxDiff(gb, hb));
    }
}

// ---------------------------------------------------------------------------
// 入口
// ---------------------------------------------------------------------------
inline int runElementwiseTests()
{
    printf("\n===== 元素级算子 =====\n");
    e1_fill();
    e2_add();
    e3_sub();
    e4_sub_inplace();
    e5_scale_mul();
    e6_scale_mul_inplace();
    e7_elementwise_mul();
    e8_elementwise_mul_inplace();
    return fails();
}

} // namespace gputest
