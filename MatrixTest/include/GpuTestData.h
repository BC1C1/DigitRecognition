#pragma once
// ============================================================================
//  GpuTestData.h —— 数据侧算子：平移增强、随机填充、随机矩阵
//
//  这一组的共同点：它们的输入或输出是"数据"，不是网络张量。所以测得住的
//  东西和前面的算子不一样 —— 不是"算得准不准"，而是**映射关系对不对**、
//  **边界怎么处理**、**随机数是否符合约定**。
//
//  三类各测什么：
//
//  D 组  launchShiftImg —— 平移增强
//        输出对输入是 **置换 + 补零**，不做算术，所以用 kExactTol 对拍。
//        关键是 random_list 由调用方提供 —— 所以测试能**自己构造**位移，
//        把边界值（正好 0、正好 1、maxShift 本身）都钉到。
//        这是本次唯一能精确对拍的"增强"路径。
//
//  F 组  launchFillUniform / Matrix::randnMatrix —— 随机填充
//        随机数没法逐位对拍（序列本身没有参照），所以换三条判据：
//        值域、分布的粗略矩、以及"生成器是有状态的"。
//        randnMatrix 额外要覆盖**奇数 n 的慢路径** —— 那条路会多生成一个数
//        再拷回来，走的是 launchFillNormal 的 tmp 缓冲分支。
//
//  【全局状态提醒】
//  getRandnGenerator() 是进程级单例、种子固定 42、**有状态**。所以本文件的
//  用例之间存在顺序依赖：换了执行顺序，各用例拿到的随机数就不同。对同一次
//  构建来说结果是确定的（这正是"同代码跑两次逐位相同"的来源），但**不要
//  在用例之间插入新的随机数消费者**。
//
//  【测试清单】
//      D1  launchShiftImg  基本对拍 × 多组 (N, H, W, maxShift)
//      D2  launchShiftImg  极端位移：全零、只留一角
//      D3  launchShiftImg  H != W 非方形
//      D4  launchShiftImg  N / H / W 非 grid 倍数
//      D5  launchShiftImg  maxShift = 0（恒等）
//      D6  launchShiftImg  位移为 (0,0) 时不改动输出冗余
//      F1  Matrix::randnMatrix  形状 / 统计矩 / 奇数 n 慢路径
//      F2  launchFillUniform    值域 + 均匀性 + 生成器有状态
// ============================================================================

#include "GpuTestCommon.h"

namespace gputest {

// ---------------------------------------------------------------------------
// D 组：平移增强
// ---------------------------------------------------------------------------

// host 端参照实现。照抄旧 CPU 版（main.cpp 里那个 augmentShift）的语义：
//     sy = y - dy, sx = x - dx；越界的位置填 0
// 这里不复用被测 kernel，也不调用任何 Matrix 成员。
inline std::vector<float> refShiftImg(const std::vector<float>& src,
                                      size_t N, size_t H, size_t W,
                                      const std::vector<float>& rand01, int maxShift)
{
    const int span = 2 * maxShift + 1;
    std::vector<float> out(N * H * W, 0.0f);
    for (size_t n = 0; n < N; ++n) {
        // 与 kernel 完全一致的取整方式（截断，不是四舍五入）
        const int dw = (int)(rand01[2 * n]     * span) - maxShift;
        const int dh = (int)(rand01[2 * n + 1] * span) - maxShift;
        for (int y = 0; y < (int)H; ++y) {
            const int sy = y - dh;
            if (sy < 0 || sy >= (int)H) continue;
            for (int x = 0; x < (int)W; ++x) {
                const int sx = x - dw;
                if (sx < 0 || sx >= (int)W) continue;
                out[n * H * W + (size_t)y * W + (size_t)x] =
                    src[n * H * W + (size_t)sy * W + (size_t)sx];
            }
        }
    }
    return out;
}

// 造一张 (N, H, W) 的图，host 侧一堆 [-1,1)
inline std::vector<float> makeImgData(size_t n, unsigned seed)
{
    return makeData(n, seed);
}

// 上传 random_list（host 已定值），执行一次 shift，收回输出
inline std::vector<float> runShift(const std::vector<float>& src_h,
                                   size_t N, size_t H, size_t W,
                                   const std::vector<float>& rand01, int maxShift,
                                   const char* what)
{
    Matrix src(N, H * W);
    upload(src, src_h);
    Matrix dst(N, H * W);
    zeroDevice(dst);                    // 先清零：漏写的位置会保持 0，而 0 也是参照的合法值

    float* dRand = nullptr;
    cudaMallocAsync_throw(&dRand, sizeof(float) * rand01.size());
    cudaMemcpy(dRand, rand01.data(), sizeof(float) * rand01.size(), cudaMemcpyHostToDevice);

    launchShiftImg(src.data_gpu(), dst.data_gpu(), N, 1, H, W, dRand, maxShift);
    runAndCheck(what);
    cudaFree(dRand);

    return download(dst);
}

// ---------------------------------------------------------------------------
// D1：基本对拍
// ---------------------------------------------------------------------------
inline void d1_shift_basic()
{
    printf("[D1] launchShiftImg 基本对拍\n");

    struct Case { size_t N, H, W; int maxShift; const char* tag; };
    const Case cases[] = {
        {  64, 28, 28, 3, "MNIST 一批 / ±3" },
        {   1, 28, 28, 3, "单张" },
        {  64, 28, 28, 0, "maxShift=0" },
        {  64, 28, 28, 1, "±1" },
        {  64, 28, 28, 7, "±7（比通常大）" },
        {  16,  8,  8, 2, "小图" },
    };

    for (const Case& c : cases) {
        const std::vector<float> src = makeImgData(c.N * c.H * c.W, 4001);

        // 随机的随机表（覆盖 [0,1) 的各个区间）
        std::vector<float> r(2 * c.N);
        for (size_t i = 0; i < r.size(); ++i)
            r[i] = (float)((i * 7919u + 13u) % 1000u) / 1000.0f;

        const std::vector<float> got = runShift(src, c.N, c.H, c.W, r, c.maxShift, "launchShiftImg");
        const std::vector<float> want = refShiftImg(src, c.N, c.H, c.W, r, c.maxShift);

        const double e = maxDiff(got, want);
        if (e >= kExactTol) {
            const size_t k = firstDiff(got, want, kExactTol);
            printf("   [%s] 首个不同 %zu（样本 %zu, 行 %zu, 列 %zu）：得到 %.9g，期望 %.9g\n",
                   c.tag, k, k / (c.H * c.W), (k / c.W) % c.H, k % c.W,
                   (double)got[k], (double)want[k]);
        }
        G_CHECK(e < kExactTol, "[%s] 误差 %.3e", c.tag, e);
    }
}

// ---------------------------------------------------------------------------
// D2：极端位移。0 和 1 是取整边界：
//     rand=0   → dw = 0*span - maxShift = -maxShift
//     rand→1   → dw = (span-1) - maxShift = +maxShift
//   所以整批偏移到极限时，要么几乎全零，要么只保留一角 —— 这两种都必须是
//   精确的，不能靠容差盖过去。
// ---------------------------------------------------------------------------
inline void d2_shift_extremes()
{
    printf("[D2] launchShiftImg 极端位移（0 与 1 的取整边界）\n");

    const size_t N = 8, H = 28, W = 28;
    const int maxShift = 3;
    const std::vector<float> src = makeImgData(N * H * W, 4101);

    struct Case { float v; const char* tag; };
    const Case cases[] = {
        { 0.0f,     "rand=0 → 位移 -maxShift" },
        { 0.9999f,  "rand→1 → 位移 +maxShift" },
        { 0.5f,     "rand=0.5 → 位移 0 附近" },
    };

    for (const Case& c : cases) {
        std::vector<float> r(2 * N, c.v);
        const std::vector<float> got = runShift(src, N, H, W, r, maxShift, "launchShiftImg");
        const std::vector<float> want = refShiftImg(src, N, H, W, r, maxShift);

        const double e = maxDiff(got, want);
        if (e >= kExactTol) {
            const size_t k = firstDiff(got, want, kExactTol);
            printf("   [%s] 首个不同 %zu：得到 %.9g，期望 %.9g\n", c.tag, k, (double)got[k], (double)want[k]);
        }
        G_CHECK(e < kExactTol, "[%s] 误差 %.3e", c.tag, e);
    }

    // maxShift >= H 时，位移必然把整张图推出边界 → 输出应当全零。
    // 这是"上界"的独立判据：不依赖参照实现，直接看输出。
    {
        const size_t N2 = 4;
        std::vector<float> r(2 * N2, 0.0f);        // 位移 -maxShift
        const std::vector<float> src2 = makeImgData(N2 * H * W, 4102);
        const int bigShift = (int)H;               // 整张推出
        const std::vector<float> got = runShift(src2, N2, H, W, r, bigShift, "launchShiftImg");

        size_t nonzero = 0;
        for (float x : got) if (x != 0.0f) ++nonzero;
        G_CHECK(nonzero == 0, "maxShift=%d（= H）时应当整批为 0，却有 %zu 个非零", bigShift, nonzero);
    }
}

// ---------------------------------------------------------------------------
// D3：非方形 H != W。行和列的越界判断是两段独立代码，方形图上一次错两个方向
//     会互相掩盖。
// ---------------------------------------------------------------------------
inline void d3_shift_non_square()
{
    printf("[D3] launchShiftImg 非方形 H != W\n");

    struct Case { size_t N, H, W; int maxShift; const char* tag; };
    const Case cases[] = {
        { 32,  7, 28, 2, "H < W" },
        { 32, 28,  7, 2, "H > W" },
        { 32,  1, 16, 1, "H = 1" },
        { 32, 16,  1, 1, "W = 1" },
        {  1,  5, 13, 4, "单张非方" },
    };

    for (const Case& c : cases) {
        const std::vector<float> src = makeImgData(c.N * c.H * c.W, 4201);
        std::vector<float> r(2 * c.N);
        for (size_t i = 0; i < r.size(); ++i)
            r[i] = (float)((i * 4093u + 7u) % 997u) / 997.0f;

        const std::vector<float> got = runShift(src, c.N, c.H, c.W, r, c.maxShift, "launchShiftImg");
        const std::vector<float> want = refShiftImg(src, c.N, c.H, c.W, r, c.maxShift);

        const double e = maxDiff(got, want);
        if (e >= kExactTol) {
            const size_t k = firstDiff(got, want, kExactTol);
            printf("   [%s] 首个不同 %zu：得到 %.9g，期望 %.9g\n", c.tag, k, (double)got[k], (double)want[k]);
        }
        G_CHECK(e < kExactTol, "[%s] H=%zu W=%zu 误差 %.3e", c.tag, c.H, c.W, e);
    }
}

// ---------------------------------------------------------------------------
// D4：N / H / W 不是 grid 维度的倍数。
//   threads = (16, 4, 4) —— x 管 W、y 管 H、z 管 N。
//   28 恰好是 4 的倍数，所以 MNIST 上"多算一块"这类错误完全不显形。
//   这里专挑非倍数：块数少算一块，就会有一整片像素保持 0。
// ---------------------------------------------------------------------------
inline void d4_shift_grid_coverage()
{
    printf("[D4] launchShiftImg 非 grid 倍数（threads = 16,4,4）\n");

    struct Case { size_t N, H, W; int maxShift; const char* tag; };
    const Case cases[] = {
        {  63, 27, 27, 2, "三者都非倍数" },
        {  65, 29, 29, 2, "三者都非倍数（+1）" },
        {   1,  1,  1, 0, "退化 1x1x1" },
        {   3, 30, 30, 3, "N 非 4 倍数" },
        {   4, 17, 19, 2, "H/W 非 4 和 16 倍数" },
        {  33,  5,  3, 1, "极小图" },
    };

    for (const Case& c : cases) {
        const std::vector<float> src = makeImgData(c.N * c.H * c.W, 4301);
        std::vector<float> r(2 * c.N);
        for (size_t i = 0; i < r.size(); ++i)
            r[i] = (float)((i * 6151u + 3u) % 991u) / 991.0f;

        const std::vector<float> got = runShift(src, c.N, c.H, c.W, r, c.maxShift, "launchShiftImg");
        const std::vector<float> want = refShiftImg(src, c.N, c.H, c.W, r, c.maxShift);

        const double e = maxDiff(got, want);
        if (e >= kExactTol) {
            const size_t k = firstDiff(got, want, kExactTol);
            printf("   [%s] 首个不同 %zu（样本 %zu 行 %zu 列 %zu）：得到 %.9g，期望 %.9g\n",
                   c.tag, k, k / (c.H * c.W), (k / c.W) % c.H, k % c.W,
                   (double)got[k], (double)want[k]);
        }
        G_CHECK(e < kExactTol, "[%s] N=%zu H=%zu W=%zu 误差 %.3e", c.tag, c.N, c.H, c.W, e);
    }
}

// ---------------------------------------------------------------------------
// D5：maxShift = 0。span = 1，rand 乘 1 取整必然是 0，所以位移恒为 (0,0)。
//   这一条不依赖参照实现也能断言：输出必须**逐位等于输入**。
// ---------------------------------------------------------------------------
inline void d5_shift_identity()
{
    printf("[D5] launchShiftImg maxShift = 0（恒等）\n");

    const size_t N = 16, H = 9, W = 11;
    const std::vector<float> src = makeImgData(N * H * W, 4401);

    std::vector<float> r(2 * N);
    for (size_t i = 0; i < r.size(); ++i) r[i] = (float)(i % 7) / 7.0f;   // 故意各不相同

    const std::vector<float> got = runShift(src, N, H, W, r, 0, "launchShiftImg");

    const double e = maxDiff(got, src);
    if (e >= kExactTol) {
        const size_t k = firstDiff(got, src, kExactTol);
        printf("   首个不同 %zu：得到 %.9g，输入 %.9g\n", k, (double)got[k], (double)src[k]);
    }
    G_CHECK(e < kExactTol, "maxShift=0 时应当是恒等映射，误差 %.3e", e);
}

// ---------------------------------------------------------------------------
// D6：位移恰好为 (0,0) 时（rand*span 落在中间那格），输出应当等于输入。
//   和 D5 的区别：这里是 span=7 但取整结果为 0，走的是"正常路径 + 零位移"。
//   maxShift=3 时 rand=0.5 → (int)(0.5*7)=3 → dw = 3-3 = 0。
// ---------------------------------------------------------------------------
inline void d6_shift_zero_offset()
{
    printf("[D6] launchShiftImg 零位移（maxShift=3, rand=0.5）\n");

    const size_t N = 8, H = 6, W = 6;
    const std::vector<float> src = makeImgData(N * H * W, 4501);
    std::vector<float> r(2 * N, 0.5f);

    const std::vector<float> got = runShift(src, N, H, W, r, 3, "launchShiftImg");

    const double e = maxDiff(got, src);
    G_CHECK(e < kExactTol, "rand=0.5 且 maxShift=3 时位移应为 0，误差 %.3e", e);
}

// ---------------------------------------------------------------------------
// F 组：随机填充
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// F1：Matrix::randnMatrix —— 形状、统计矩、奇数 n 的慢路径。
//
// 随机序列没有参照可以逐位对拍，所以只能查性质：
//   - 形状对
//   - 均值 ≈ 0、标准差 ≈ scale（用大样本，容差给得保守）
//   - **奇数 n 不报错且值有效** —— 这条走 launchFillNormal 的 tmp 缓冲分支，
//     是新增代码里最容易出错的一段（多生成一个、拷贝前 n 个）
// ---------------------------------------------------------------------------
inline void f1_randn_matrix()
{
    printf("[F1] Matrix::randnMatrix（含奇数 n 慢路径）\n");

    // (a) 形状 + 统计矩。样本取大一点，矩才有意义。
    {
        const size_t R = 256, C = 256;
        const float scale = 0.5f;
        const Matrix m = Matrix::randnMatrix(R, C, scale);
        if (!runAndCheck("randnMatrix")) { G_CHECK(false, "randnMatrix 报错"); }
        else {
            G_CHECK(m.rows() == R && m.cols() == C,
                    "形状不对：得到 (%zu, %zu)，期望 (%zu, %zu)", m.rows(), m.cols(), R, C);

            const std::vector<float> h = download(m);
            double sum = 0.0, sumSq = 0.0;
            for (float x : h) { sum += (double)x; sumSq += (double)x * (double)x; }
            const double mean = sum / (double)h.size();
            const double var = sumSq / (double)h.size() - mean * mean;
            const double sd = std::sqrt(var > 0 ? var : 0.0);

            // 样本 65536，均值的标准误约 scale/256 ≈ 0.002；容差给 0.05 已经极宽
            G_CHECK(std::fabs(mean) < 0.05, "均值偏离 0 太多：%.6f", mean);
            G_CHECK(std::fabs(sd - (double)scale) < 0.05,
                    "标准差偏离 scale 太多：得到 %.6f，期望 %.6f", sd, (double)scale);
            printf("   [统计] 65536 个样本：均值 %.6f，标准差 %.6f（期望 0 / %.2f）\n",
                   mean, sd, (double)scale);
        }
    }

    // (b) 奇数 n 慢路径。挑几个奇数尺寸，逐个验证形状 + 值全为有限数。
    {
        struct Case { size_t R, C; const char* tag; };
        const Case cases[] = {
            {   1,   1, "1x1（n=1，最小奇数）" },
            {   1,   3, "1x3（n=3）" },
            {   3,   3, "3x3（n=9）" },
            {   5,   7, "5x7（n=35）" },
            { 129,   1, "129x1（n=129）" },
            {   1, 784, "1x784（偶数，对照走直路）" },
            { 128,  64, "128x64（偶数，对照走直路）" },
        };

        for (const Case& c : cases) {
            const Matrix m = Matrix::randnMatrix(c.R, c.C, 1.0f);
            if (!runAndCheck("randnMatrix (奇数路径)")) {
                G_CHECK(false, "[%s] 报错", c.tag);
                continue;
            }
            G_CHECK(m.rows() == c.R && m.cols() == c.C,
                    "[%s] 形状不对：(%zu, %zu)", c.tag, m.rows(), m.cols());

            const std::vector<float> h = download(m);
            size_t bad = 0;
            for (float x : h) if (!std::isfinite(x)) ++bad;
            G_CHECK(bad == 0, "[%s] 有 %zu 个非有限值", c.tag, bad);

            // 全 0 也是可疑的（说明 tmp 拷贝没生效）
            size_t zero = 0;
            for (float x : h) if (x == 0.0f) ++zero;
            G_CHECK(zero < h.size(), "[%s] 输出全是 0，慢路径可能没把值拷回来", c.tag);
        }
    }
}

// ---------------------------------------------------------------------------
// F2：launchFillUniform —— 值域、均匀性、生成器有状态。
//
// 值域这条要注意：curandGenerateUniform 的文档说生成 (0,1)，但实现上常见的是
// [0,1)。这里按**闭区间**判，只钉住"不该出现负数、不该出现 >1"；同时统计
// 是否真的取到过 0 —— 这个观察结果决定了 shiftImgKernel 里 dw 能不能等于
// -maxShift（即 `(int)(0.0f * span)` 这一路）。
// ---------------------------------------------------------------------------
inline void f2_fill_uniform()
{
    printf("[F2] launchFillUniform\n");

    const size_t n = 65536;
    Matrix m(1, n);
    launchFillUniform(m.data_gpu(), n);
    if (!runAndCheck("launchFillUniform (1)")) { G_CHECK(false, "报错"); return; }

    const std::vector<float> a = download(m);

    // (a) 值域
    size_t neg = 0, above = 0, exact0 = 0, exact1 = 0;
    for (float x : a) {
        if (x < 0.0f)      ++neg;
        if (x > 1.0f)      ++above;
        if (x == 0.0f)     ++exact0;
        if (x == 1.0f)     ++exact1;
    }
    G_CHECK(neg == 0,   "出现 %zu 个负值", neg);
    G_CHECK(above == 0, "出现 %zu 个大于 1 的值", above);

    // (b) 均匀性：分 10 桶，每桶期望 10%。样本 65536 时单桶标准差约 0.12%，
    //     容差给 1% 已经很宽（约 8 个标准差）。
    size_t bucket[10] = { 0 };
    for (float x : a) {
        size_t b = (size_t)(x * 10.0f);
        if (b > 9) b = 9;
        ++bucket[b];
    }
    for (int i = 0; i < 10; ++i) {
        const double frac = (double)bucket[i] / (double)n;
        G_CHECK(std::fabs(frac - 0.1) < 0.01,
                "第 %d 桶占比 %.4f，偏离 0.1 超过 1%%（桶计数 %zu）", i, frac, bucket[i]);
    }

    printf("   [观察] 取到 0 的次数 %zu，取到 1 的次数 %zu（共 %zu 个样本）\n",
           exact0, exact1, n);

    // (c) 生成器有状态：连续两次调用应当不同。
    //     这一条同时钉住"每次调用没有偷偷重置种子" —— 如果哪天改成每次重置，
    //     同一次训练里所有批会拿到完全相同的位移，而准确率照样能看。
    Matrix m2(1, n);
    launchFillUniform(m2.data_gpu(), n);
    if (!runAndCheck("launchFillUniform (2)")) { G_CHECK(false, "第二次报错"); return; }

    const std::vector<float> b2 = download(m2);
    G_CHECK(maxDiff(a, b2) > kExactTol,
            "连续两次 launchFillUniform 结果完全相同 —— 生成器状态可能被重置了");
}

// ---------------------------------------------------------------------------
// 入口
// ---------------------------------------------------------------------------
inline int runDataTests()
{
    printf("\n===== 数据侧算子（平移 / 随机填充）=====\n");
    d1_shift_basic();
    d2_shift_extremes();
    d3_shift_non_square();
    d4_shift_grid_coverage();
    d5_shift_identity();
    d6_shift_zero_offset();
    f1_randn_matrix();
    f2_fill_uniform();
    return fails();
}

} // namespace gputest
