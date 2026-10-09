#pragma once
// ============================================================================
//  MatmulTest.h —— GEMM 内核的对拍测试
//
//  被测量的三个内核（签名都是 void(const Matrix&, const Matrix&, Matrix&)）：
//      matmul_nn(A, B, out)   A(M,K)  B(K,N)  ->  out(M,N) = A * B
//      matmul_nt(A, B, out)   A(M,K)  B(N,K)  ->  out(M,N) = A * B^T
//      matmul_tn(A, B, out)   A(K,M)  B(K,N)  ->  out(M,N) = A^T * B
//
//  ⚠ 三个内核都是【累加】语义：内部写的是 out += ...，所以 out 必须预先清零。
//    S2 就是把这个约定钉成测试；哪天内核改成覆盖写，S2 会红，提醒你回头改
//    调用方（Conv2D 之类的）。
//
//  为什么值得单独测：这三个是 Conv2D 的计算引擎。它们错了卷积也跟着错，而且
//  错得不明显——因为「参照实现」很可能就是其中一个内核。所以下面所有参照都
//  用朴素三重循环重写，绝不复用被测量的内核。
//
//  【测试清单】
//      S1  12 组维度三内核对拍（含 1、非 4 倍数、跨 blocksize 的 63/64/65）
//      S2  累加语义：out 初值非零时，结果 = 正确值 + 初值
//      S3  nt 与 transpose()+nn 交叉对拍（两条独立路径互证）
// ============================================================================

#include "TestCommon.h"

namespace matmul_test {

// ---------------------------------------------------------------------------
// 朴素参照：三重循环，不复用任何被测量的内核
// ---------------------------------------------------------------------------
inline Matrix naiveNN(const Matrix& A, const Matrix& B)      // (M,N) = A(M,K) * B(K,N)
{
    Matrix C(A.rows(), B.cols());
    for (size_t i = 0; i < A.rows(); ++i)
        for (size_t j = 0; j < B.cols(); ++j) {
            double s = 0.0;
            for (size_t k = 0; k < A.cols(); ++k) s += A(i, k) * B(k, j);
            C(i, j) = s;
        }
    return C;
}

inline Matrix naiveNT(const Matrix& A, const Matrix& B)      // (M,N) = A(M,K) * B(N,K)^T
{
    Matrix C(A.rows(), B.rows());
    for (size_t i = 0; i < A.rows(); ++i)
        for (size_t j = 0; j < B.rows(); ++j) {
            double s = 0.0;
            for (size_t k = 0; k < A.cols(); ++k) s += A(i, k) * B(j, k);
            C(i, j) = s;
        }
    return C;
}

inline Matrix naiveTN(const Matrix& A, const Matrix& B)      // (M,N) = A(K,M)^T * B(K,N)
{
    Matrix C(A.cols(), B.cols());
    for (size_t i = 0; i < A.cols(); ++i)
        for (size_t j = 0; j < B.cols(); ++j) {
            double s = 0.0;
            for (size_t k = 0; k < A.rows(); ++k) s += A(k, i) * B(k, j);
            C(i, j) = s;
        }
    return C;
}

// ---------------------------------------------------------------------------
// S1：维度扫描。包含 1（退化）、3/5（小于向量宽度 4）、17/63/65（非块的整数倍）
// ---------------------------------------------------------------------------
inline void s1_shapes()
{
    printf("[S1] 12 组维度 x 3 内核，与朴素三重循环对拍\n");

    struct Dim { size_t M, N, K; };
    const Dim dims[] = {
        {  1,   1,   1 }, {  1,   1,   5 }, {  2,   3,   5 }, {  3,   5,   7 },
        {  4,   4,   4 }, {  5,   4,   3 }, { 16,  16,  16 }, { 17,  17,  17 },
        { 63,  65,  64 }, { 64,  64,  64 }, { 65,  63,  66 }, { 100, 80, 120 },
    };

    for (const Dim& d : dims) {
        Matrix A   = Matrix::randnMatrix(d.M, d.K);

        // --- nn: A(M,K) x B(K,N)
        Matrix Bnn = Matrix::randnMatrix(d.K, d.N);
        Matrix o1(d.M, d.N);
        Matrix::matmul_nn(A, Bnn, o1);
        const double e1 = dstest::maxDiff(o1, naiveNN(A, Bnn));
        T_CHECK(e1 < dstest::kFloatTol, "nn  M=%zu N=%zu K=%zu 最大误差 %.3e", d.M, d.N, d.K, e1);

        // --- nt: A(M,K) x B(N,K)^T
        Matrix Bnt = Matrix::randnMatrix(d.N, d.K);
        Matrix o2(d.M, d.N);
        Matrix::matmul_nt(A, Bnt, o2);
        const double e2 = dstest::maxDiff(o2, naiveNT(A, Bnt));
        T_CHECK(e2 < dstest::kFloatTol, "nt  M=%zu N=%zu K=%zu 最大误差 %.3e", d.M, d.N, d.K, e2);

        // --- tn: A(K,M)^T x B(K,N)
        Matrix Atn = Matrix::randnMatrix(d.K, d.M);
        Matrix o3(d.M, d.N);
        Matrix::matmul_tn(Atn, Bnn, o3);
        const double e3 = dstest::maxDiff(o3, naiveTN(Atn, Bnn));
        T_CHECK(e3 < dstest::kFloatTol, "tn  M=%zu N=%zu K=%zu 最大误差 %.3e", d.M, d.N, d.K, e3);
    }
}

// ---------------------------------------------------------------------------
// S2：累加语义。out 先填 1，跑完之后应当等于「正确值 + 1」。
//     这条测试是双向的：它既验证内核在累加，也提醒调用方必须清零。
// ---------------------------------------------------------------------------
inline void s2_accumulate_semantics()
{
    printf("[S2] 累加语义：out 初值非零 -> 结果 = 正确值 + 初值\n");

    const size_t M = 5, N = 4, K = 7;
    Matrix A  = Matrix::randnMatrix(M, K);
    Matrix B  = Matrix::randnMatrix(K, N);      // nn 用
    Matrix Bn = Matrix::randnMatrix(N, K);      // nt 用
    Matrix At = Matrix::randnMatrix(K, M);      // tn 用

    auto seeded = [](size_t r, size_t c) {
        Matrix o(r, c);
        for (size_t i = 0; i < r; ++i)
            for (size_t j = 0; j < c; ++j) o(i, j) = 1.0;
        return o;
    };
    auto plusOne = [](Matrix w) {
        for (size_t i = 0; i < w.rows(); ++i)
            for (size_t j = 0; j < w.cols(); ++j) w(i, j) += 1.0;
        return w;
    };

    Matrix o1 = seeded(M, N);
    Matrix::matmul_nn(A, B, o1);
    const double e1 = dstest::maxDiff(o1, plusOne(naiveNN(A, B)));
    T_CHECK(e1 < dstest::kFloatTol, "nn 初值为 1 时误差 %.3e（内核应做 out += ，不是覆盖写）", e1);

    Matrix o2 = seeded(M, N);
    Matrix::matmul_nt(A, Bn, o2);
    const double e2 = dstest::maxDiff(o2, plusOne(naiveNT(A, Bn)));
    T_CHECK(e2 < dstest::kFloatTol, "nt 初值为 1 时误差 %.3e（内核应做 out += ，不是覆盖写）", e2);

    Matrix o3 = seeded(M, N);
    Matrix::matmul_tn(At, B, o3);
    const double e3 = dstest::maxDiff(o3, plusOne(naiveTN(At, B)));
    T_CHECK(e3 < dstest::kFloatTol, "tn 初值为 1 时误差 %.3e（内核应做 out += ，不是覆盖写）", e3);
}

// ---------------------------------------------------------------------------
// S3：nt 与 transpose()+nn 交叉对拍。
//     A * B^T 有两条独立走法，两条都对才是真的对。
// ---------------------------------------------------------------------------
inline void s3_nt_vs_transpose_nn()
{
    printf("[S3] nt 与 transpose()+nn 交叉对拍\n");

    struct Dim { size_t M, N, K; };
    const Dim dims[] = { { 4, 9, 9 }, { 7, 5, 13 }, { 33, 17, 64 }, { 64, 64, 65 } };

    for (const Dim& d : dims) {
        Matrix A  = Matrix::randnMatrix(d.M, d.K);
        Matrix B  = Matrix::randnMatrix(d.N, d.K);   // nt 的 B：按行读
        Matrix o1(d.M, d.N), o2(d.M, d.N);

        Matrix::matmul_nt(A, B, o1);

        Matrix Bt = B.transpose();                   // (K,N)
        Matrix::matmul_nn(A, Bt, o2);

        const double e = dstest::maxDiff(o1, o2);
        T_CHECK(e < dstest::kFloatTol, "M=%zu N=%zu K=%zu 两条路径差 %.3e", d.M, d.N, d.K, e);
    }
}

// ---------------------------------------------------------------------------
// 入口
// ---------------------------------------------------------------------------
inline int runMatmulTests()
{
    dstest::resetFails();
    printf("\n===== GEMM 内核测试开始 =====\n");
    s1_shapes();
    s2_accumulate_semantics();
    s3_nt_vs_transpose_nn();
    printf("===== GEMM 内核测试结束：失败 %d 条 =====\n\n", dstest::fails());
    return dstest::fails();
}

} // namespace matmul_test
