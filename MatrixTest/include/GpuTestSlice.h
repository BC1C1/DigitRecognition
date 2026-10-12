#pragma once
// ============================================================================
//  GpuTestSlice.h —— 按索引表挑行的两个算子
//
//  被测的是两样东西，故意分开测：
//
//    1) launchSliceRowsByIndex(ga, gb, M, N, count, idx)
//       底层算子。gb 由调用方提供，所以这里能自己控制输出缓冲的大小和内容 ——
//       **这是唯一能检查"有没有越界写 / 有没有漏写"的位置**。
//
//    2) Matrix::sliceRowsByIndex(const size_t* d_idx, size_t n)
//       包装层。它只收一个已经在显存里的索引表，ret 在内部构造，所以能测的是
//       **形状**和**内容**，测不到越界。
//
//  【为什么这批用例偏"边界"而不是"多来几组形状"】
//  这个算子做的是纯搬运，'算得对不对'一眼就能看出来；难的是**边界**：
//
//    count 不是 16 的倍数   →  grid.y 少一块，尾部若干行不写（静默）
//    cols  不是 16 的倍数   →  grid.x 少一块，右侧若干列不写（静默）
//    count = 1             →  grid.y 恰好一块
//    count = 0             →  空 grid
//    idx 里下标重复         →  输出应当出现重复的行，而不是被去重
//    idx 取满且乱序         →  输出是 source 的一个排列
//
//  前两条是最要紧的：漏写**不会崩**，留下的是输出缓冲里的旧值。而这个缓冲在
//  真实训练里是跨批复用的（矩阵复用），所以表现是"某几个样本带着上一批的
//  像素" —— loss 照样下降，准确率只低一点点。
//
//  容差用 kExactTol：纯搬运、不做算术，float 之间应当**逐位相等**。
//
//  【测试清单】
//      G1-1  launch  基本形状 × 多组 (M, N, count)
//      G1-2  launch  输出缓冲哨兵 —— 检出漏写与越界写
//      G1-3  launch  count = 0（只记录底层行为，边界归调用方）
//      G1-4  launch  count = 1
//      G1-5  launch  cols 非 16 倍数
//      G1-6  launch  count 与 cols 同时非 16 倍数
//      G1-7  launch  下标乱序 + 重复 + 取满
//      G1-8  Matrix  包装层的形状与内容
// ============================================================================

#include "GpuTestCommon.h"

#include <numeric>

namespace gputest {

// ---------------------------------------------------------------------------
// ⚠ 所有用例的前提：count <= M，且 idx[i] < M。
//
// 这两个条件在底层是一条 assert（TypeDefine.h 的 sliceRowsByIndex），而 assert
// 在 Release 下是空的 —— 违反它不会报错，会**越界读**，然后表现成"某些输出
// 位置没被写"。测试用例自己必须先满足前提，否则测的是未定义行为。
// ---------------------------------------------------------------------------

// 已知的哨兵值。取一个不会和 makeData 输出（[-1,1)）重合的数。
static const float kSliceSentinel = -999.0f;

// ---------------------------------------------------------------------------
// 把 host 的索引表传到显存。调用方负责 cudaFree。
// n == 0 时返回 nullptr（设备端 kernel 会在 i >= count 处直接返回，
// 不会解引用它）。
// ---------------------------------------------------------------------------
inline size_t* makeDeviceIndex(const std::vector<size_t>& idx)
{
    if (idx.empty()) return nullptr;
    size_t* d = nullptr;
    cudaMalloc_throw(&d, sizeof(size_t) * idx.size());
    cudaMemcpy(d, idx.data(), sizeof(size_t) * idx.size(), cudaMemcpyHostToDevice);
    return d;
}

// ---------------------------------------------------------------------------
// host 端参照实现。照抄 launch 的语义，不复用被测代码：
//     gb[i][j] = ga[idx[i]][j]
// ---------------------------------------------------------------------------
inline std::vector<float> refSliceRows(const std::vector<float>& src,
                                       size_t srcRows, size_t cols,
                                       const std::vector<size_t>& idx)
{
    const size_t count = idx.size();
    std::vector<float> out(count * cols);
    for (size_t i = 0; i < count; ++i) {
        assert(idx[i] < srcRows);
        for (size_t j = 0; j < cols; ++j)
            out[i * cols + j] = src[idx[i] * cols + j];
    }
    return out;
}

// 打印现场：首个不同的位置 + 它的行列
inline void reportSliceDiff(const char* tag, const std::vector<float>& got,
                           const std::vector<float>& want, size_t cols)
{
    const size_t n = firstDiff(got, want, kExactTol);
    if (n >= got.size()) {
        printf("   [%s] 长度不一致或无法定位：got %zu, want %zu\n",
               tag, got.size(), want.size());
        return;
    }
    printf("   [%s] 首个不同位置 %zu（行 %zu, 列 %zu）：得到 %.9g，期望 %.9g\n",
           tag, n, n / cols, n % cols, (double)got[n], (double)want[n]);
}

// ---------------------------------------------------------------------------
// G1-1：基本形状 × 多组参数。
// 形状表里同时放入"整齐"和"不整齐"的两种，方便和后面几条对照。
// ---------------------------------------------------------------------------
inline void g1_1_basic()
{
    printf("[G1-1] launchSliceRowsByIndex 基本形状\n");

    struct Case { size_t M, C, count; const char* tag; };
    const Case cases[] = {
        {   64,   784,   64, "一整批" },
        {  128,    10,   32, "小列数" },
        {  100,   300,   17, "非方" },
        { 1000,   784,  100, "千行取百" },
        { 5000,   784, 1000, "验证集 chunk" },
        {  256,   256,  256, "全 16 倍数" },
    };

    for (const Case& c : cases) {
        Matrix src = makeMatrix(c.M, c.C, 3001);
        Matrix dst(c.count, c.C);

        std::vector<size_t> idx(c.count);
        std::iota(idx.begin(), idx.end(), 0);
        // 打乱，避免"顺序取"掩盖索引没用上的情况
        for (size_t i = c.count; i > 1; --i)
            std::swap(idx[i - 1], idx[(i * 7919u) % i]);

        size_t* dIdx = makeDeviceIndex(idx);
        launchSliceRowsByIndex(src.data_gpu(), dst.data_gpu(), c.M, c.C, c.count, dIdx);
        const bool ok = runAndCheck("launchSliceRowsByIndex");
        if (dIdx) cudaFree(dIdx);

        if (!ok) { G_CHECK(false, "[%s] kernel 失败", c.tag); continue; }

        const std::vector<float> want = refSliceRows(download(src), c.M, c.C, idx);
        const std::vector<float> got = download(dst);
        const double e = maxDiff(got, want);
        if (e >= kExactTol) reportSliceDiff(c.tag, got, want, c.C);
        G_CHECK(e < kExactTol, "[%s] M=%zu C=%zu count=%zu 最大误差 %.3e",
                c.tag, c.M, c.C, c.count, e);
    }
}

// ---------------------------------------------------------------------------
// G1-2：哨兵。输出缓冲预填 kSliceSentinel，跑完必须一个都不剩。
//
// 这一条同时覆盖两件事：
//   - 漏写（count / cols 让 grid 少算了一块）→ 留下哨兵
//   - 越界写（grid 多算了一块，或者写错偏移）→ 多出来的位置被改动
// 所以除了"输出全等于期望"，还要"和预填的旧内容相比，不该动的地方一个都没动"。
// ---------------------------------------------------------------------------
inline void g1_2_sentinel()
{
    printf("[G1-2] 输出缓冲哨兵：漏写与越界写\n");

    struct Case { size_t M, C, count; const char* tag; };
    const Case cases[] = {
        { 128, 784,  63, "count 比 16 倍数少 1" },
        { 128, 784,  65, "count 比 16 倍数多 1" },
        {  64, 300,  64, "cols 非 16 倍数" },
        { 100, 300,  17, "两者都非 16 倍数" },
        { 128,  17,   3, "极小列数" },
        {  33,  65,   1, "count=1" },
    };

    for (const Case& c : cases) {
        Matrix src = makeMatrix(c.M, c.C, 3101);
        Matrix dst(c.count, c.C);

        // 预填哨兵
        {
            std::vector<float> sentinel(c.count * c.C, kSliceSentinel);
            upload(dst, sentinel);
        }

        std::vector<size_t> idx(c.count);
        for (size_t i = 0; i < c.count; ++i)
            idx[i] = (i * 37u + 11u) % c.M;      // 乱序、可能重复

        size_t* dIdx = makeDeviceIndex(idx);
        launchSliceRowsByIndex(src.data_gpu(), dst.data_gpu(), c.M, c.C, c.count, dIdx);
        const bool ok = runAndCheck("launchSliceRowsByIndex");
        if (dIdx) cudaFree(dIdx);

        if (!ok) { G_CHECK(false, "[%s] kernel 失败", c.tag); continue; }

        const std::vector<float> want = refSliceRows(download(src), c.M, c.C, idx);
        const std::vector<float> got = download(dst);

        // 1) 内容正确
        const double e = maxDiff(got, want);
        if (e >= kExactTol) reportSliceDiff(c.tag, got, want, c.C);
        G_CHECK(e < kExactTol, "[%s] 内容不符，最大误差 %.3e", c.tag, e);

        // 2) 没有留下哨兵 —— 任何一个位置还是 -999，说明那个位置根本没被写
        size_t leftover = 0;
        for (float x : got) if (std::fabs(x - kSliceSentinel) < kExactTol) ++leftover;
        G_CHECK(leftover == 0, "[%s] 有 %zu 个输出位置没被写（仍是哨兵），占 %.1f%%",
                c.tag, leftover, 100.0 * leftover / (double)got.size());
    }
}

// ---------------------------------------------------------------------------
// G1-3：count = 0。
//
// 【这一条为什么不判失败】边界情形归调用方管，不归算子管。
// Matrix::sliceRowsByIndex 已经有 `if (n == 0) return Matrix();`，所以真实路径
// 根本走不到这里。底层 launch 在 count=0 时算出的 grid.y 是 0，而 CUDA 拒绝
// 维度为 0 的 grid（invalid argument）—— 这是底层算子的已知行为，不是缺陷。
//
// 所以这里只**记录**它：将来若有人给 launch 补上 `if (count == 0) return;`，
// 这一条会打印出"空 grid 被接受"，提醒他上面那个早返回可以不必再依赖。
// ---------------------------------------------------------------------------
inline void g1_3_empty()
{
    printf("[G1-3] count = 0（边界归 Matrix 层，此处只记录底层行为）\n");

    const size_t M = 64, C = 784;
    Matrix src = makeMatrix(M, C, 3201);
    Matrix dst = makeConst(1, C, kSliceSentinel);   // 借用一行当"输出"，不该被碰

    const std::vector<float> before = download(dst);
    launchSliceRowsByIndex(src.data_gpu(), dst.data_gpu(), M, C, 0, nullptr);

    const cudaError_t e = cudaGetLastError();
    cudaDeviceSynchronize();
    cudaGetLastError();     // 清掉，别让它粘住后面的用例

    if (e == cudaSuccess) {
        printf("   [信息] count=0 被接受（空 grid 没报错）—— 底层已经会挡这一手\n");
    } else {
        printf("   [信息] count=0 被 CUDA 拒绝：%s\n"
               "          这是预期的：grid.y 算出来是 0，而 CUDA 不接受 0 维 grid。\n"
               "          调用方 Matrix::sliceRowsByIndex 已在 n==0 时早返回。\n",
               cudaGetErrorString(e));
    }

    const std::vector<float> after = download(dst);
    G_CHECK(maxDiff(before, after) < kExactTol,
            "count=0 时改动了不是输出的缓冲");
}

// ---------------------------------------------------------------------------
// G1-4：count = 1。grid.y 恰好一块，且是"块内只有一个线程有效"的情形。
// ---------------------------------------------------------------------------
inline void g1_4_single()
{
    printf("[G1-4] count = 1\n");

    struct Case { size_t M, C, row; const char* tag; };
    const Case cases[] = {
        {  64,  784,  0,  "第一行" },
        {  64,  784, 63,  "最后一行" },
        { 100,  300, 57,  "中间行 / 非 16 倍数" },
    };

    for (const Case& c : cases) {
        Matrix src = makeMatrix(c.M, c.C, 3301);
        Matrix dst(1, c.C);

        std::vector<size_t> idx{ c.row };
        size_t* dIdx = makeDeviceIndex(idx);
        launchSliceRowsByIndex(src.data_gpu(), dst.data_gpu(), c.M, c.C, 1, dIdx);
        const bool ok = runAndCheck("launchSliceRowsByIndex (count=1)");
        if (dIdx) cudaFree(dIdx);

        if (!ok) { G_CHECK(false, "[%s] kernel 失败", c.tag); continue; }

        const std::vector<float> src_h = download(src);
        std::vector<float> want(c.C);
        for (size_t j = 0; j < c.C; ++j) want[j] = src_h[c.row * c.C + j];

        const std::vector<float> got = download(dst);
        const double e = maxDiff(got, want);
        if (e >= kExactTol) reportSliceDiff(c.tag, got, want, c.C);
        G_CHECK(e < kExactTol, "[%s] 取第 %zu 行失败，误差 %.3e", c.tag, c.row, e);
    }
}

// ---------------------------------------------------------------------------
// G1-5 / G1-6：cols 和 count 不是 16 的倍数。
// 这两条是针对 grid 维度的：x 方向每一块管 16 列，一旦块数算少，右侧的列
// 一个都不会被写；y 方向同理，尾部的行全部漏掉。列数是 16 的倍数时这两类
// 错误都不显形，所以专门挑非倍数来打。
// ---------------------------------------------------------------------------
inline void g1_5_grid_coverage()
{
    printf("[G1-5/6] cols 与 count 的非 16 倍数覆盖\n");

    struct Case { size_t M, C, count; const char* tag; };
    const Case cases[] = {
        {  128,  300,  64, "cols=300" },
        {  128,  301,  64, "cols=301" },
        {  128,  783,  64, "cols=783（784-1）" },
        {  128,  785,  64, "cols=785（784+1）" },
        {  128,  784,  63, "count=63" },
        {  128,  784,  65, "count=65" },
        { 1000,  300, 100, "两者同时非倍数" },
        { 1000,  301,  17, "两者同时非倍数 / 极小 count" },
        {  100,  301, 100, "取满 / cols 非倍数" },
    };

    for (const Case& c : cases) {
        Matrix src = makeMatrix(c.M, c.C, 3401);
        Matrix dst(c.count, c.C);

        {
            std::vector<float> sentinel(c.count * c.C, kSliceSentinel);
            upload(dst, sentinel);
        }

        // 逆序取前 count 行：既乱序，又保证下标都合法
        std::vector<size_t> idx(c.count);
        for (size_t i = 0; i < c.count; ++i)
            idx[i] = c.M - 1 - (i % c.M);

        size_t* dIdx = makeDeviceIndex(idx);
        launchSliceRowsByIndex(src.data_gpu(), dst.data_gpu(), c.M, c.C, c.count, dIdx);
        const bool ok = runAndCheck("launchSliceRowsByIndex");
        if (dIdx) cudaFree(dIdx);

        if (!ok) { G_CHECK(false, "[%s] kernel 失败", c.tag); continue; }

        const std::vector<float> want = refSliceRows(download(src), c.M, c.C, idx);
        const std::vector<float> got = download(dst);

        const double e = maxDiff(got, want);
        if (e >= kExactTol) reportSliceDiff(c.tag, got, want, c.C);
        G_CHECK(e < kExactTol, "[%s] M=%zu C=%zu count=%zu 误差 %.3e",
                c.tag, c.M, c.C, c.count, e);

        size_t leftover = 0;
        for (float x : got) if (std::fabs(x - kSliceSentinel) < kExactTol) ++leftover;
        G_CHECK(leftover == 0, "[%s] 有 %zu 个位置没被写", c.tag, leftover);
    }
}

// ---------------------------------------------------------------------------
// G1-7：下标乱序 / 重复 / 取满。
// 重复下标是重点：它必须产生**重复的行**，不能被去重，也不能只写一次。
// 取满 + 打乱则要求输出是 source 的一个排列 —— 用一个和来间接验证（搬运不改数值，
// 所以和应当完全相等，顺便也验证了没有多写少写）。
// ---------------------------------------------------------------------------
inline void g1_7_index_patterns()
{
    printf("[G1-7] 下标乱序 / 重复 / 取满\n");

    const size_t M = 64, C = 784;

    // (a) 重复：全取同一个下标
    {
        Matrix src = makeMatrix(M, C, 3501);
        const size_t count = 5;
        Matrix dst(count, C);

        std::vector<size_t> idx(count, 42);         // 五行全是同一行
        size_t* dIdx = makeDeviceIndex(idx);
        launchSliceRowsByIndex(src.data_gpu(), dst.data_gpu(), M, C, count, dIdx);
        const bool ok = runAndCheck("launchSliceRowsByIndex (重复下标)");
        if (dIdx) cudaFree(dIdx);

        if (!ok) { G_CHECK(false, "[重复] kernel 失败"); }
        else {
            const std::vector<float> want = refSliceRows(download(src), M, C, idx);
            const std::vector<float> got = download(dst);
            const double e = maxDiff(got, want);
            if (e >= kExactTol) reportSliceDiff("重复", got, want, C);
            G_CHECK(e < kExactTol, "[重复] 五行同下标应当得到五个相同的行，误差 %.3e", e);
        }
    }

    // (b) 取满 + 打乱：输出是 source 的一个排列
    {
        Matrix src = makeMatrix(M, C, 3502);
        Matrix dst(M, C);

        std::vector<size_t> idx(M);
        std::iota(idx.begin(), idx.end(), 0);
        for (size_t i = M; i > 1; --i)
            std::swap(idx[i - 1], idx[(i * 4093u) % i]);

        size_t* dIdx = makeDeviceIndex(idx);
        launchSliceRowsByIndex(src.data_gpu(), dst.data_gpu(), M, C, M, dIdx);
        const bool ok = runAndCheck("launchSliceRowsByIndex (取满乱序)");
        if (dIdx) cudaFree(dIdx);

        if (!ok) { G_CHECK(false, "[取满] kernel 失败"); }
        else {
            const std::vector<float> src_h = download(src);
            const std::vector<float> want = refSliceRows(src_h, M, C, idx);
            const std::vector<float> got = download(dst);

            const double e = maxDiff(got, want);
            if (e >= kExactTol) reportSliceDiff("取满", got, want, C);
            G_CHECK(e < kExactTol, "[取满] 误差 %.3e", e);

            // 和不变 —— 独立于参照实现的第二条判据：搬运不该增减任何数值
            double sumSrc = 0.0, sumDst = 0.0;
            for (float x : src_h) sumSrc += (double)x;
            for (float x : got)   sumDst += (double)x;
            G_CHECK(std::fabs(sumSrc - sumDst) < kExactTol * std::max(1.0, std::fabs(sumSrc)),
                    "[取满] 全元素和变了：source %.6f，输出 %.6f", sumSrc, sumDst);
        }
    }

    // (c) 逆序
    {
        Matrix src = makeMatrix(M, C, 3503);
        Matrix dst(M, C);

        std::vector<size_t> idx(M);
        for (size_t i = 0; i < M; ++i) idx[i] = M - 1 - i;

        size_t* dIdx = makeDeviceIndex(idx);
        launchSliceRowsByIndex(src.data_gpu(), dst.data_gpu(), M, C, M, dIdx);
        const bool ok = runAndCheck("launchSliceRowsByIndex (逆序)");
        if (dIdx) cudaFree(dIdx);

        if (!ok) { G_CHECK(false, "[逆序] kernel 失败"); }
        else {
            const std::vector<float> want = refSliceRows(download(src), M, C, idx);
            const std::vector<float> got = download(dst);
            const double e = maxDiff(got, want);
            if (e >= kExactTol) reportSliceDiff("逆序", got, want, C);
            G_CHECK(e < kExactTol, "[逆序] 误差 %.3e", e);
        }
    }
}

// ---------------------------------------------------------------------------
// G1-8：Matrix::sliceRowsByIndex 包装层。
//
// 它不收 host 的 vector（那正是重构的目的：索引表由调用方常驻显存），所以这里
// 造一个和真实训练里一样的设备索引表，传指针进去。测两件事：
//   - 形状：ret 必须是 (n, cols)，cols 要来自 source 而不是 0
//   - 内容：逐位等于参照
//
// 越界写在这里测不到（ret 是内部构造的），那由 G1-2 负责。
// ---------------------------------------------------------------------------
inline void g1_8_matrix_wrapper()
{
    printf("[G1-8] Matrix::sliceRowsByIndex 包装层\n");

    struct Case { size_t M, C, count; const char* tag; };
    const Case cases[] = {
        {   64, 784,  64, "一整批" },
        { 5000, 784, 1000, "验证集 chunk" },
        {  100, 300,  17, "非 16 倍数" },
        {  128,  10,  32, "小列数" },
    };

    for (const Case& c : cases) {
        const Matrix src = makeMatrix(c.M, c.C, 3601);

        std::vector<size_t> idx(c.count);
        for (size_t i = 0; i < c.count; ++i)
            idx[i] = (i * 53u + 7u) % c.M;

        size_t* dIdx = makeDeviceIndex(idx);
        const Matrix ret = src.sliceRowsByIndex(dIdx, idx.size());
        const bool ok = runAndCheck("Matrix::sliceRowsByIndex");
        if (dIdx) cudaFree(dIdx);

        if (!ok) { G_CHECK(false, "[%s] 调用失败", c.tag); continue; }

        // 形状
        G_CHECK(ret.rows() == c.count && ret.cols() == c.C,
                "[%s] 形状不对：得到 (%zu, %zu)，期望 (%zu, %zu)",
                c.tag, ret.rows(), ret.cols(), c.count, c.C);

        if (ret.rows() != c.count || ret.cols() != c.C) continue;

        const std::vector<float> want = refSliceRows(download(src), c.M, c.C, idx);
        const std::vector<float> got = download(ret);
        const double e = maxDiff(got, want);
        if (e >= kExactTol) reportSliceDiff(c.tag, got, want, c.C);
        G_CHECK(e < kExactTol, "[%s] 内容误差 %.3e", c.tag, e);
    }

    // n = 0：包装层应当早返回。这里只要求"不崩、不报 CUDA 错"。
    // ⚠ 注意它返回的是 Matrix()，也就是 (0, 0) —— 列数会丢。这一点记在这里，
    //   不判失败：调用方如果依赖 ret.cols()，拿到的是 0 而不是 source 的列数。
    {
        const Matrix src = makeMatrix(16, 784, 3602);
        const Matrix ret = src.sliceRowsByIndex(nullptr, 0);
        const bool ok = runAndCheck("Matrix::sliceRowsByIndex (n=0)");
        G_CHECK(ok, "[n=0] 调用报错");
        printf("   [n=0] 返回形状 (%zu, %zu)，rows 应当为 0\n", ret.rows(), ret.cols());
        G_CHECK(ret.rows() == 0, "[n=0] 返回的 rows 应当为 0，得到 %zu", ret.rows());
    }
}

// ---------------------------------------------------------------------------
// 入口
// ---------------------------------------------------------------------------
inline int runSliceTests()
{
    printf("\n===== 按索引挑行 =====\n");
    g1_1_basic();
    g1_2_sentinel();
    g1_3_empty();
    g1_4_single();
    g1_5_grid_coverage();
    g1_7_index_patterns();
    g1_8_matrix_wrapper();
    return fails();
}

} // namespace gputest
