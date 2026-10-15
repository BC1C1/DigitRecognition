#pragma once
// ============================================================================
//  GpuTestMatmul.h —— 三个 GEMM 变体的对拍
//
//  规格（都来自 cuda_bridge.h 和 Matrix.cu 的实现）：
//      launchMatmulNN(ga, gb, gc, M, N, K, BM, BN, BK)   ga(M,K)  gb(K,N)  -> gc(M,N)
//      launchMatmulNT(ga, gb, gc, M, N, K)               ga(M,K)  gb(N,K)  -> gc(M,N)
//      launchMatmulTN(ga, gb, gc, M, N, K)               ga(K,M)  gb(K,N)  -> gc(M,N)
//
//  ⚠ NN 已经换成**分块版**（shared memory 搬运 + 寄存器累加），所以它的签名比
//    NT / TN 多三个块尺寸。块尺寸不是纯调优旋钮 —— 它和 launch 里的 blockDim
//    绑死：
//
//        launchMatmulNN 里   dim3 threads = { 16, 16 };            ← 硬编码
//        kernel 计算阶段     blocka[ty * BK + k] * blockb[k * BN + tx]
//        kernel 写回         row = ti * BM + ty,  col = tj * BN + tx
//
//    ty 只能取 [0,16)、tx 只能取 [0,16)，所以每个 block 实际只覆盖 BM=16 行、
//    BN=16 列。BM/BN 传 16 以外的值**不会算错，而是有一批输出压根没人写**
//    （哨兵留在原地，能一眼看出来）。M5 后半段就是钉这一条：把 launch 里的
//    threads 改成 { BN, BM } 之后，那几条会从红变绿。
//
//    BK 不受这个限制：搬运循环的 idx < BM*BK / BK*BN 和计算循环的 k < BK
//    都按它走，取任何值是自由的（M5 前半段扫它）。
//
//  ⚠ 三个都是**覆盖写**（gc[...] = value），不是累加。这和 CPU 版的
//    Matrix::matmul_* 语义不同（那边内部写 out += ...，调用方要先清零）。
//    M4 专门钉这一条：把输出预填成非零再跑，结果必须完全被覆盖。
//
//  ⚠ 分块版还有个隐式前提：分块搬运要把越界的格点填 0（加法的单位元），
//    而不是跳过。M6 用 15/16/17 这套"块尺寸前后一格"的形状压它 ——
//    少填一格、或者填错方向，这里第一个红。
//
//  形状表里有意思的几个：
//      1x1x1        退化
//      15/16/17     block 尺寸 16 的前后一格 —— 上一轮 grid 算错就是在这里露的
//      63x65 / 65x63  M 和 N 一大一小，能看出两个方向的块数有没有各自算对
//      100x80x120   和 maxpool/im2col 里出现的真实量级接近
//      128x128x128  全 16 的倍数，一路顺畅
//
//  【测试清单】
//      M1  launchMatmulNN 多形状
//      M2  launchMatmulNT 多形状
//      M3  launchMatmulTN 多形状
//      M4  覆盖写语义（输出预填哨兵）
//      M5  块尺寸扫描：BK 自由变化 / BM·BN 是否真的只能取 16
//      M6  边界余数：M/N/K 落在块尺寸两侧
//      M7  K 远大于 BK：多轮 tk 的搬运、累加与 __syncthreads
// ============================================================================

#include "GpuTestCommon.h"

namespace gputest {

struct Dim { size_t M, N, K; };

static const Dim kDims[] = {
    {   1,   1,    1 },
    {   1,   1,    5 },
    {   3,   5,    7 },
    {  15,  16,   17 },
    {  16,  16,   16 },
    {  17,  17,   17 },
    {  33,  17,   64 },
    {  63,  65,   64 },
    {  65,  63,   66 },
    { 100,  80,  120 },
    { 128, 128,  128 },
};

// 当前 launch 唯一支持的块尺寸组合。
// 单独拎出来是为了让"其他地方想换块尺寸"这件事有一个显眼的落点。
static const size_t kBM = 16, kBN = 16, kBK = 16;

// 输出预填的哨兵。用它而不是 0：0 是合法的乘积结果，分不出"没写"和"真算了 0"。
constexpr float kSentinel = -999.0f;

// ---------------------------------------------------------------------------
// 三个 host 端参照实现。一律用 double 累加 —— 不是为了让结果更"准"，
// 而是让参照自身的舍入误差比被测 kernel 低几个数量级，免得两边都糊。
// ---------------------------------------------------------------------------
inline std::vector<float> refNN(const std::vector<float>& A, const std::vector<float>& B,
                                size_t M, size_t N, size_t K)
{
    std::vector<float> C(M * N, 0.0f);
    for (size_t i = 0; i < M; ++i)
        for (size_t j = 0; j < N; ++j) {
            double s = 0.0;
            for (size_t k = 0; k < K; ++k) s += (double)A[i * K + k] * (double)B[k * N + j];
            C[i * N + j] = (float)s;
        }
    return C;
}

inline std::vector<float> refNT(const std::vector<float>& A, const std::vector<float>& B,
                                size_t M, size_t N, size_t K)
{
    // A(M,K), B(N,K)
    std::vector<float> C(M * N, 0.0f);
    for (size_t i = 0; i < M; ++i)
        for (size_t j = 0; j < N; ++j) {
            double s = 0.0;
            for (size_t k = 0; k < K; ++k) s += (double)A[i * K + k] * (double)B[j * K + k];
            C[i * N + j] = (float)s;
        }
    return C;
}

inline std::vector<float> refTN(const std::vector<float>& A, const std::vector<float>& B,
                                size_t M, size_t N, size_t K)
{
    // A(K,M), B(K,N)
    std::vector<float> C(M * N, 0.0f);
    for (size_t i = 0; i < M; ++i)
        for (size_t j = 0; j < N; ++j) {
            double s = 0.0;
            for (size_t k = 0; k < K; ++k) s += (double)A[k * M + i] * (double)B[k * N + j];
            C[i * N + j] = (float)s;
        }
    return C;
}

// 数一数有多少个输出位置还原封不动地留着哨兵 —— 也就是"没人写过这里"。
// 分块 kernel 的块边界算错时，症状就是这个，而不是数值偏差。
inline size_t countUntouched(const std::vector<float>& v)
{
    size_t n = 0;
    for (float x : v) if (std::fabs((double)x - (double)kSentinel) < kExactTol) ++n;
    return n;
}

// ---------------------------------------------------------------------------
// M1：launchMatmulNN   A(M,K) x B(K,N) -> C(M,N)
// ---------------------------------------------------------------------------
inline void m1_matmul_nn()
{
    printf("[M1] launchMatmulNN\n");

    for (const Dim& d : kDims) {
        Matrix A = makeMatrix(d.M, d.K, 101);
        Matrix B = makeMatrix(d.K, d.N, 102);
        Matrix C(d.M, d.N);

        // 预填哨兵：漏写的元素会留下它，能一眼看出"少算了哪块"
        upload(C, std::vector<float>(d.M * d.N, kSentinel));

        launchMatmulNN(A.data_gpu(), B.data_gpu(), C.data_gpu(), d.M, d.N, d.K,
                       kBM, kBN, kBK);
        if (!runAndCheck("launchMatmulNN")) {
            G_CHECK(false, "M=%zu N=%zu K=%zu kernel 失败", d.M, d.N, d.K);
            continue;
        }

        const std::vector<float> want = refNN(download(A), download(B), d.M, d.N, d.K);
        const std::vector<float> got = download(C);
        const double e = maxDiff(got, want);
        const size_t untouched = countUntouched(got);
        G_CHECK(untouched == 0, "nn  M=%zu N=%zu K=%zu 有 %zu 个输出没被写",
                d.M, d.N, d.K, untouched);
        G_CHECK(e < kFloatTol, "nn  M=%zu N=%zu K=%zu 最大误差 %.3e（首个不同 %zu）",
                d.M, d.N, d.K, e, firstDiff(got, want));
    }
}

// ---------------------------------------------------------------------------
// M2：launchMatmulNT   A(M,K) x B(N,K) -> C(M,N)
// ---------------------------------------------------------------------------
inline void m2_matmul_nt()
{
    printf("[M2] launchMatmulNT\n");

    for (const Dim& d : kDims) {
        Matrix A = makeMatrix(d.M, d.K, 201);
        Matrix B = makeMatrix(d.N, d.K, 202);   // 注意：B 是 (N,K)
        Matrix C(d.M, d.N);
        upload(C, std::vector<float>(d.M * d.N, kSentinel));

        launchMatmulNT(A.data_gpu(), B.data_gpu(), C.data_gpu(), d.M, d.N, d.K);
        if (!runAndCheck("launchMatmulNT")) {
            G_CHECK(false, "M=%zu N=%zu K=%zu kernel 失败", d.M, d.N, d.K);
            continue;
        }

        const std::vector<float> want = refNT(download(A), download(B), d.M, d.N, d.K);
        const std::vector<float> got = download(C);
        const double e = maxDiff(got, want);
        G_CHECK(e < kFloatTol, "nt  M=%zu N=%zu K=%zu 最大误差 %.3e（首个不同 %zu）",
                d.M, d.N, d.K, e, firstDiff(got, want));
    }
}

// ---------------------------------------------------------------------------
// M3：launchMatmulTN   A(K,M) x B(K,N) -> C(M,N)
// ---------------------------------------------------------------------------
inline void m3_matmul_tn()
{
    printf("[M3] launchMatmulTN\n");

    for (const Dim& d : kDims) {
        Matrix A = makeMatrix(d.K, d.M, 301);   // 注意：A 是 (K,M)
        Matrix B = makeMatrix(d.K, d.N, 302);   // B 是 (K,N)
        Matrix C(d.M, d.N);
        upload(C, std::vector<float>(d.M * d.N, kSentinel));

        launchMatmulTN(A.data_gpu(), B.data_gpu(), C.data_gpu(), d.M, d.N, d.K);
        if (!runAndCheck("launchMatmulTN")) {
            G_CHECK(false, "M=%zu N=%zu K=%zu kernel 失败", d.M, d.N, d.K);
            continue;
        }

        const std::vector<float> want = refTN(download(A), download(B), d.M, d.N, d.K);
        const std::vector<float> got = download(C);
        const double e = maxDiff(got, want);
        G_CHECK(e < kFloatTol, "tn  M=%zu N=%zu K=%zu 最大误差 %.3e（首个不同 %zu）",
                d.M, d.N, d.K, e, firstDiff(got, want));
    }
}

// ---------------------------------------------------------------------------
// M4：覆盖写语义。输出预填 123，跑完必须**完全等于** A*B —— 那个 123 一点不剩。
//     如果哪天有人把 kernel 改成累加（好和 CPU 版对齐），这里会红，
//     提醒他调用方（Linear / Conv2D 现在都按"输出会被覆盖"写）要一起改。
// ---------------------------------------------------------------------------
inline void m4_overwrite_semantics()
{
    printf("[M4] 覆盖写语义：输出预填非零，结果应完全被覆盖\n");

    const size_t M = 9, N = 7, K = 5;
    Matrix A = makeMatrix(M, K, 401);
    Matrix B = makeMatrix(K, N, 402);
    Matrix C(M, N);

    const float sentinel = 123.0f;
    upload(C, std::vector<float>(M * N, sentinel));

    launchMatmulNN(A.data_gpu(), B.data_gpu(), C.data_gpu(), M, N, K, kBM, kBN, kBK);
    if (!runAndCheck("launchMatmulNN")) { G_CHECK(false, "kernel 失败"); return; }

    const std::vector<float> want = refNN(download(A), download(B), M, N, K);
    const std::vector<float> got = download(C);

    // 和"纯 A*B"比：覆盖写应当完全一致
    const double eOver = maxDiff(got, want);
    G_CHECK(eOver < kFloatTol,
            "应当是覆盖写：结果应完全等于 A*B，实测差 %.3e", eOver);

    // 和"A*B + 哨兵"比：如果命中了这个，说明它是累加语义
    std::vector<float> wantAcc(want);
    for (float& x : wantAcc) x += sentinel;
    const double eAcc = maxDiff(got, wantAcc);
    if (eAcc < kFloatTol) {
        G_CHECK(false, "检测到累加语义（结果 = A*B + 初值）—— 和 cuda_bridge.h 里写的覆盖写不符");
    }

    // 哨兵残留检查：任何一个输出位置如果还是 123，说明那个位置根本没被写
    size_t leftover = 0;
    for (float x : got) if (std::fabs(x - sentinel) < kExactTol) ++leftover;
    G_CHECK(leftover == 0, "有 %zu 个输出位置完全没被写（还是初值）", leftover);
}

// ---------------------------------------------------------------------------
// M5：块尺寸扫描
//
//   前半段 —— BK 自由。搬运循环的边界是 BM*BK / BK*BN，计算循环是 k < BK，
//   所以 BK 取 1、取到比 K 还大、或者正好等于 K，都该是对的。顺带把两种
//   极端都压到：BK=1 是 TK=K 轮、每轮只搬一格；BK>K 是一轮里大部分格点
//   越界、全靠"填 0"兜住。
//
//   后半段 —— BM / BN 是否真的只能取 16。这里不跟参照比数值，先数哨兵：
//   凡是没被写的输出都还在原地。当前 launch 写死 threads = {16,16}，
//   所以 BM=32 那一行会整片留着哨兵。
// ---------------------------------------------------------------------------
inline void m5_block_sizes()
{
    printf("[M5] 块尺寸扫描\n");

    const size_t M = 64, N = 48, K = 96;

    // --- 前半段：BK 自由 ---
    static const size_t kBKs[] = { 1, 8, 16, 32, 96, 128 };
    for (size_t bk : kBKs) {
        Matrix A = makeMatrix(M, K, 501);
        Matrix B = makeMatrix(K, N, 502);
        Matrix C(M, N);
        upload(C, std::vector<float>(M * N, kSentinel));

        launchMatmulNN(A.data_gpu(), B.data_gpu(), C.data_gpu(), M, N, K, 16, 16, bk);
        if (!runAndCheck("launchMatmulNN")) {
            G_CHECK(false, "BK=%zu kernel 失败（shared = (16*%zu + %zu*16)*4 字节）",
                    bk, bk, bk);
            continue;
        }

        const std::vector<float> want = refNN(download(A), download(B), M, N, K);
        const std::vector<float> got = download(C);
        const double e = maxDiff(got, want);
        G_CHECK(countUntouched(got) == 0, "BK=%zu 有 %zu 个输出没被写",
                bk, countUntouched(got));
        G_CHECK(e < kFloatTol, "BK=%zu 最大误差 %.3e（首个不同 %zu）", bk, e, firstDiff(got, want));
    }

    // --- 后半段：BM / BN ---
    // 期望 16/16 绿，其余红并指出"有多少输出没人写"。
    static const size_t kBlocks[] = { 16, 32 };
    for (size_t bm : kBlocks) {
        for (size_t bn : kBlocks) {
            Matrix A = makeMatrix(M, K, 511);
            Matrix B = makeMatrix(K, N, 512);
            Matrix C(M, N);
            upload(C, std::vector<float>(M * N, kSentinel));

            launchMatmulNN(A.data_gpu(), B.data_gpu(), C.data_gpu(), M, N, K, bm, bn, 16);
            if (!runAndCheck("launchMatmulNN")) {
                G_CHECK(false, "BM=%zu BN=%zu kernel 失败", bm, bn);
                continue;
            }

            const std::vector<float> got = download(C);
            const size_t untouched = countUntouched(got);
            if (untouched != 0) {
                G_CHECK(false,
                        "BM=%zu BN=%zu：有 %zu/%zu 个输出没被写 —— launch 里 blockDim 写死了 (16,16)，"
                        "而 kernel 的写回是 row = ti*BM + ty、col = tj*BN + tx，"
                        "ty/tx 只到 15。把 threads 改成 { BN, BM } 就能放开这一条",
                        bm, bn, untouched, got.size());
                continue;
            }

            const std::vector<float> want = refNN(download(A), download(B), M, N, K);
            const double e = maxDiff(got, want);
            G_CHECK(e < kFloatTol, "BM=%zu BN=%zu 最大误差 %.3e（首个不同 %zu）",
                    bm, bn, e, firstDiff(got, want));
        }
    }
}

// ---------------------------------------------------------------------------
// M6：边界余数。块尺寸固定 16，把 M/N/K 摆在它两侧 —— 15（整块差一行）、
//     16（正好）、17（整块多一行）。分块 kernel 最容易死的地方就是这里：
//     越界的格点必须填 0 而不是跳过，且 A 的列方向 / B 的行方向都要按各自
//     的块偏移判断，写反一条就会读到别人的数据（数值错，而不只是精度差）。
// ---------------------------------------------------------------------------
inline void m6_edge_remainders()
{
    printf("[M6] 边界余数：M/N/K 落在块尺寸 16 两侧\n");

    static const Dim kEdge[] = {
        {   1,   1,   1 },
        {   1,   1,  17 },
        {  17,   1,   1 },
        {  15,  15,  15 },
        {  16,  16,  16 },
        {  17,  17,  17 },
        {  15,  16,  17 },
        {  17,  16,  15 },
        {  16,  17,  15 },
        {  15,  17,  16 },
        {  31,  33,  32 },
        {  33,  31,  32 },
        {  32,  33,  31 },
        {  47,  48,  49 },
    };

    for (const Dim& d : kEdge) {
        Matrix A = makeMatrix(d.M, d.K, 601);
        Matrix B = makeMatrix(d.K, d.N, 602);
        Matrix C(d.M, d.N);
        upload(C, std::vector<float>(d.M * d.N, kSentinel));

        launchMatmulNN(A.data_gpu(), B.data_gpu(), C.data_gpu(), d.M, d.N, d.K,
                       kBM, kBN, kBK);
        if (!runAndCheck("launchMatmulNN")) {
            G_CHECK(false, "M=%zu N=%zu K=%zu kernel 失败", d.M, d.N, d.K);
            continue;
        }

        const std::vector<float> want = refNN(download(A), download(B), d.M, d.N, d.K);
        const std::vector<float> got = download(C);
        const size_t untouched = countUntouched(got);
        G_CHECK(untouched == 0, "M=%zu N=%zu K=%zu 有 %zu 个输出没被写",
                d.M, d.N, d.K, untouched);
        const double e = maxDiff(got, want);
        G_CHECK(e < kFloatTol, "M=%zu N=%zu K=%zu 最大误差 %.3e（首个不同 %zu）",
                d.M, d.N, d.K, e, firstDiff(got, want));
    }
}

// ---------------------------------------------------------------------------
// M7：K 远大于 BK。形状故意选成非块倍数（M=N=20），让"最后一轮 tk 不满"
//     和"tk 轮数很多"两件事同时发生。
//     两处 __syncthreads() 的职责在这里才看得出来：少了后面那个，下一轮
//     搬运会覆盖掉还在被读的 smem。
// ---------------------------------------------------------------------------
inline void m7_deep_k()
{
    printf("[M7] K 远大于 BK：多轮 tk\n");

    const size_t M = 20, N = 20;
    static const size_t kKs[] = { 17, 64, 256, 1000 };

    for (size_t K : kKs) {
        Matrix A = makeMatrix(M, K, 701);
        Matrix B = makeMatrix(K, N, 702);
        Matrix C(M, N);
        upload(C, std::vector<float>(M * N, kSentinel));

        launchMatmulNN(A.data_gpu(), B.data_gpu(), C.data_gpu(), M, N, K, kBM, kBN, kBK);
        if (!runAndCheck("launchMatmulNN")) {
            G_CHECK(false, "K=%zu kernel 失败", K);
            continue;
        }

        // K 越大，float 累加的绝对误差按 eps*sqrt(K) 量级走，所以这里用
        // 随量级缩放的容差，而不是固定的 kFloatTol。
        const std::vector<float> want = refNN(download(A), download(B), M, N, K);
        const std::vector<float> got = download(C);
        const double e = maxDiff(got, want);
        double worst = 0.0;
        for (float x : want) worst = std::max(worst, std::fabs((double)x));
        G_CHECK(e < relTol(worst),
                "K=%zu 最大误差 %.3e（结果量级 %.1f，容差 %.3e；首个不同 %zu）",
                K, e, worst, relTol(worst), firstDiff(got, want));
    }
}

// ---------------------------------------------------------------------------
// 入口
// ---------------------------------------------------------------------------
inline int runMatmulTests()
{
    printf("\n===== GEMM 三变体 =====\n");
    m1_matmul_nn();
    m2_matmul_nt();
    m3_matmul_tn();
    m4_overwrite_semantics();
    m5_block_sizes();
    m6_edge_remainders();
    m7_deep_k();
    return fails();
}

} // namespace gputest
