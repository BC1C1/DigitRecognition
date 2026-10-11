#pragma once
// ============================================================================
//  GpuTestMatmul.h —— 三个 GEMM 变体的对拍
//
//  规格（都来自 cuda_bridge.h 和 Matrix.cu 的实现）：
//      launchMatmulNN(ga, gb, gc, M, N, K)   ga(M,K)  gb(K,N)  -> gc(M,N)
//      launchMatmulNT(ga, gb, gc, M, N, K)   ga(M,K)  gb(N,K)  -> gc(M,N)
//      launchMatmulTN(ga, gb, gc, M, N, K)   ga(K,M)  gb(K,N)  -> gc(M,N)
//
//  ⚠ 三个都是**覆盖写**（gc[...] = value），不是累加。这和 CPU 版的
//    Matrix::matmul_* 语义不同（那边内部写 out += ...，调用方要先清零）。
//    M4 专门钉这一条：把输出预填成非零再跑，结果必须完全被覆盖。
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
        upload(C, std::vector<float>(d.M * d.N, -999.0f));

        launchMatmulNN(A.data_gpu(), B.data_gpu(), C.data_gpu(), d.M, d.N, d.K);
        if (!runAndCheck("launchMatmulNN")) {
            G_CHECK(false, "M=%zu N=%zu K=%zu kernel 失败", d.M, d.N, d.K);
            continue;
        }

        const std::vector<float> want = refNN(download(A), download(B), d.M, d.N, d.K);
        const std::vector<float> got = download(C);
        const double e = maxDiff(got, want);
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
        upload(C, std::vector<float>(d.M * d.N, -999.0f));

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
        upload(C, std::vector<float>(d.M * d.N, -999.0f));

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

    launchMatmulNN(A.data_gpu(), B.data_gpu(), C.data_gpu(), M, N, K);
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
// 入口
// ---------------------------------------------------------------------------
inline int runMatmulTests()
{
    printf("\n===== GEMM 三变体 =====\n");
    m1_matmul_nn();
    m2_matmul_nt();
    m3_matmul_tn();
    m4_overwrite_semantics();
    return fails();
}

} // namespace gputest
