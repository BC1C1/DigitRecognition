#pragma once
// ============================================================================
//  Dispatchers.h —— 给 MatrixTest 用的矩阵乘法"分派器"与候选实现
//
//  为什么单独一个文件：下面这几个函数**只存在于基准里**，生产代码里没有，
//  所以放在这里不会和 TypeDefine.h 冲突，也不用改它。
//
//  对照的核心是这一对：
//      matmulOut_nt_strided()    —— 生产代码 matmul_nt 的做法：B 按行读
//      matmulOut_nt_transposed() —— 候选做法：先用 transpose() 转好，再走 nn
//  两者算出来的结果必须一致（main.cpp 的 MODE 0 会对拍验证）。
// ============================================================================

#include "TypeDefine.h"
#include <algorithm>

// 可调块大小（贯穿所有内核，方便扫参）
inline int64_t& g_blocksize() { static int64_t bs = 32; return bs; }
// 可调线程数（0 = 让 OpenMP 自己决定 = 吃满逻辑核）
inline int& g_threads() { static int t = 16; return t; }

// ---------------------------------------------------------------------------
// 现场取证：维度校验
//   堆损坏的特点是「报了不一定在案发现场」。所以在每个内核入口先校验维度，
//   不符时立刻打印出「哪个内核、哪个矩阵、期望什么形状、实际什么形状」，
//   而不是等后面某次 new 才崩。
// ---------------------------------------------------------------------------
#include <cstdio>
#include <cstring>

inline bool shapeOk(const char* kernel, const char* name, const Matrix& m,
                    size_t expR, size_t expC)
{
    if (m.rows() == expR && m.cols() == expC) return true;

    // 同一个内核只报一次，避免在计时循环里刷屏把结果冲掉
    static const char* seen[16] = { nullptr };
    static int seenCount = 0;
    for (int i = 0; i < seenCount; ++i)
        if (seen[i] == kernel) return false;
    if (seenCount < 16) seen[seenCount++] = kernel;

    printf("\n  [维度错误] 内核 %s 的参数 %s：期望 (%zu x %zu)，实际 (%zu x %zu)\n",
           kernel, name, expR, expC, m.rows(), m.cols());
    return false;
}

// 注意：那句"已中止"也必须一起去重，否则还是会在计时循环里刷屏
#define CHECK_SHAPE(kernel, name, m, r, c)                                     \
    do {                                                                       \
        if (!shapeOk(kernel, name, m, (r), (c))) {                             \
            static bool reported_##__LINE__ = false;                           \
            if (!reported_##__LINE__) {                                        \
                reported_##__LINE__ = true;                                    \
                printf("  → 已中止该内核，避免越界写坏堆\n");                  \
            }                                                                  \
            return;                                                            \
        }                                                                      \
    } while (0)


// ---------------------------------------------------------------------------
// 内核 1：out = A * B^T   —— B 按「行」读（生产代码 matmul_nt 的形态）
//   A(M,K)  B(N,K)  out(M,N)
//   注意：out 必须是已经清零的缓冲（与生产代码一致，内核只做 +=）
// ---------------------------------------------------------------------------
inline void matmulOut_nt_strided(const Matrix& A, const Matrix& B, Matrix& out)
{
    const int64_t M = (int64_t)A.rows();
    const int64_t K = (int64_t)A.cols();
    const int64_t N = (int64_t)B.rows();
    const int64_t bs = g_blocksize();
    const int nt = g_threads();

    CHECK_SHAPE("nt_strided", "B", B, (size_t)N, (size_t)K);
    CHECK_SHAPE("nt_strided", "out", out, (size_t)M, (size_t)N);

#pragma omp parallel num_threads(nt)
    {
#pragma omp for schedule(dynamic)
        for (int64_t bi = 0; bi < M; bi += bs) {
            const int64_t Mend = std::min(M, bi + bs);
            for (int64_t bj = 0; bj < N; bj += bs) {
                const int64_t Nend = std::min(N, bj + bs);
                for (int64_t bk = 0; bk < K; bk += bs) {
                    const int64_t Kend = std::min(K, bk + bs);
                    for (int64_t i = bi; i < Mend; ++i) {
                        for (int64_t j = bj; j < Nend; ++j) {
                            __m256d vc = _mm256_setzero_pd();
                            int64_t k = bk;
                            for (; k + 4 <= Kend; k += 4) {
                                __m256d va = _mm256_loadu_pd(&A(i, k));
                                __m256d vb = _mm256_loadu_pd(&B(j, k));
                                vc = _mm256_fmadd_pd(va, vb, vc);
                            }
                            // 水平归约
                            __m128d lo = _mm256_castpd256_pd128(vc);
                            __m128d hi = _mm256_extractf128_pd(vc, 1);
                            __m128d s = _mm_add_pd(lo, hi);
                            double sum = _mm_cvtsd_f64(s) + _mm_cvtsd_f64(_mm_unpackhi_pd(s, s));
                            for (; k < Kend; ++k) sum += A(i, k) * B(j, k);
                            out(i, j) += sum;
                        }
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// 内核 2：out = A * B^T   —— 候选做法：先把 B 转置成 NT(M,K)，再走"行对行"的 nn
//   out(i,j) = sum_k A(i,k) * BT(j,k)，BT 是标准行主序 → 访存连续
//   转置成本只算在调用方（一次性），这里只管算
// ---------------------------------------------------------------------------
inline void matmulOut_nt_transposed(const Matrix& A, const Matrix& BT, Matrix& out)
{
    const int64_t M = (int64_t)A.rows();
    const int64_t K = (int64_t)A.cols();
    const int64_t N = (int64_t)out.cols();   // 输出列数 = N（不能从 BT 推，否则是自证循环）
    const int64_t bs = g_blocksize();
    const int nt = g_threads();

    // 语义：out(i,j) = sum_k A(i,k) * BT(j,k)  →  BT 必须是 (N, K)
    CHECK_SHAPE("nt_transposed", "BT", BT, (size_t)N, (size_t)K);
    CHECK_SHAPE("nt_transposed", "out", out, (size_t)M, (size_t)N);

#pragma omp parallel num_threads(nt)
    {
#pragma omp for schedule(dynamic)
        for (int64_t bi = 0; bi < M; bi += bs) {
            const int64_t Mend = std::min(M, bi + bs);
            for (int64_t bj = 0; bj < N; bj += bs) {
                const int64_t Nend = std::min(N, bj + bs);
                for (int64_t bk = 0; bk < K; bk += bs) {
                    const int64_t Kend = std::min(K, bk + bs);
                    for (int64_t i = bi; i < Mend; ++i) {
                        const double* Ai = &A(i, 0);
                        for (int64_t j = bj; j < Nend; ++j) {
                            const double* Bj = &BT(j, 0);
                            __m256d vc = _mm256_setzero_pd();
                            int64_t k = bk;
                            for (; k + 4 <= Kend; k += 4)
                                vc = _mm256_fmadd_pd(_mm256_loadu_pd(Ai + k),
                                                     _mm256_loadu_pd(Bj + k), vc);
                            __m128d lo = _mm256_castpd256_pd128(vc);
                            __m128d hi = _mm256_extractf128_pd(vc, 1);
                            __m128d s = _mm_add_pd(lo, hi);
                            double sum = _mm_cvtsd_f64(s) + _mm_cvtsd_f64(_mm_unpackhi_pd(s, s));
                            for (; k < Kend; ++k) sum += Ai[k] * Bj[k];
                            out(i, j) += sum;
                        }
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// 内核 3：out = A^T * B   —— 生产代码 matmul_tn 的形态（按列访存 A）
//   A(K,M)  B(K,N)  out(M,N)
// ---------------------------------------------------------------------------
inline void matmulOut_tn_strided(const Matrix& A, const Matrix& B, Matrix& out)
{
    const int64_t K = (int64_t)A.rows();
    const int64_t M = (int64_t)A.cols();
    const int64_t N = (int64_t)B.cols();
    const int64_t bs = g_blocksize();
    const int nt = g_threads();

    CHECK_SHAPE("tn_strided", "B", B, (size_t)K, (size_t)N);
    CHECK_SHAPE("tn_strided", "out", out, (size_t)M, (size_t)N);

#pragma omp parallel num_threads(nt)
    {
#pragma omp for schedule(dynamic)
        for (int64_t bi = 0; bi < M; bi += bs) {
            const int64_t Mend = std::min(M, bi + bs);
            for (int64_t bj = 0; bj < N; bj += bs) {
                const int64_t Nend = std::min(N, bj + bs);
                for (int64_t bk = 0; bk < K; bk += bs) {
                    const int64_t Kend = std::min(K, bk + bs);
                    for (int64_t i = bi; i < Mend; ++i) {
                        for (int64_t j = bj; j < Nend; ++j) {
                            double sum = 0.0;
                            for (int64_t k = bk; k < Kend; ++k)
                                sum += A(k, i) * B(k, j);     // ← A 按列取，跨步
                            out(i, j) += sum;
                        }
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// 内核 4：out = A^T * B   —— 候选：先把 A 转置成 AT(M,K)，再走"行对行"
// ---------------------------------------------------------------------------
inline void matmulOut_tn_transposed(const Matrix& AT, const Matrix& B, Matrix& out)
{
    const int64_t M = (int64_t)AT.rows();
    const int64_t K = (int64_t)AT.cols();
    const int64_t N = (int64_t)B.cols();
    const int64_t bs = g_blocksize();
    const int nt = g_threads();

    // K 必须等于 B 的行数（否则 out = A^T * B 不成立）
    CHECK_SHAPE("tn_transposed", "B", B, (size_t)K, (size_t)N);
    CHECK_SHAPE("tn_transposed", "out", out, (size_t)M, (size_t)N);

#pragma omp parallel num_threads(nt)
    {
#pragma omp for schedule(dynamic)
        for (int64_t bi = 0; bi < M; bi += bs) {
            const int64_t Mend = std::min(M, bi + bs);
            for (int64_t bj = 0; bj < N; bj += bs) {
                const int64_t Nend = std::min(N, bj + bs);
                for (int64_t bk = 0; bk < K; bk += bs) {
                    const int64_t Kend = std::min(K, bk + bs);
                    for (int64_t i = bi; i < Mend; ++i) {
                        const double* Ai = &AT(i, 0);
                        for (int64_t j = bj; j < Nend; ++j) {
                            double sum = 0.0;
                            for (int64_t k = bk; k < Kend; ++k)
                                sum += Ai[k] * B(k, j);
                            out(i, j) += sum;
                        }
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// 内核 0：out = A * B   —— 与「生产版 Matrix::matmul」同一套循环，
//   但线程数与块大小由 g_threads() / g_blocksize() 控制。
//
//   ⚠️ 为什么必须单独写一个：老版 TypeDefine.h 的 matmul 里写死了
//      `#pragma omp parallel num_threads(16)`（第 181 行），而 num_threads 子句
//      优先于 omp_set_num_threads / OMP_NUM_THREADS。直接调它做线程扫描，
//      会得到"所有档位一样快"的假结论 —— 因为实际始终是 16 线程。
//
//   A(M,K)  B(K,N)  out(M,N)，out 需已清零
// ---------------------------------------------------------------------------
inline void matmulOut_nn_dispatch(const Matrix& A, const Matrix& B, Matrix& out)
{
    const int64_t M = (int64_t)A.rows();
    const int64_t K = (int64_t)A.cols();
    const int64_t N = (int64_t)B.cols();
    const int64_t bs = g_blocksize();
    const int nt = g_threads();

    CHECK_SHAPE("nn_dispatch", "B", B, (size_t)K, (size_t)N);
    CHECK_SHAPE("nn_dispatch", "out", out, (size_t)M, (size_t)N);

#pragma omp parallel num_threads(nt)
    {
#pragma omp for schedule(dynamic)
        for (int64_t bi = 0; bi < M; bi += bs) {
            const int64_t Mend = std::min(M, bi + bs);
            for (int64_t bj = 0; bj < N; bj += bs) {
                const int64_t Nend = std::min(N, bj + bs);
                for (int64_t bk = 0; bk < K; bk += bs) {
                    const int64_t Kend = std::min(K, bk + bs);
                    for (int64_t i = bi; i < Mend; ++i) {
                        for (int64_t j = bj; j < Nend; ++j) {
                            __m256d vc = _mm256_setzero_pd();
                            int64_t k = bk;
                            for (; k + 4 <= Kend; k += 4)
                                vc = _mm256_fmadd_pd(_mm256_loadu_pd(&A(i, k)),
                                                     _mm256_loadu_pd(&B(k, j)), vc);
                            __m128d lo = _mm256_castpd256_pd128(vc);
                            __m128d hi = _mm256_extractf128_pd(vc, 1);
                            __m128d s = _mm_add_pd(lo, hi);
                            double sum = _mm_cvtsd_f64(s) + _mm_cvtsd_f64(_mm_unpackhi_pd(s, s));
                            for (; k < Kend; ++k) sum += A(i, k) * B(k, j);
                            out(i, j) += sum;
                        }
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// 正确性基准：朴素三重循环（只用于对拍）
// ---------------------------------------------------------------------------
inline Matrix matmulOut_naive(const Matrix& A, const Matrix& B)
{
    Matrix C(A.rows(), B.cols());
    for (size_t i = 0; i < A.rows(); i++)
        for (size_t j = 0; j < B.cols(); j++) {
            double s = 0;
            for (size_t k = 0; k < A.cols(); k++) s += A(i, k) * B(k, j);
            C(i, j) = s;
        }
    return C;
}
inline Matrix matmulOut_naive_nt(const Matrix& A, const Matrix& B)
{
    Matrix C(A.rows(), B.rows());
    for (size_t i = 0; i < A.rows(); i++)
        for (size_t j = 0; j < B.rows(); j++) {
            double s = 0;
            for (size_t k = 0; k < A.cols(); k++) s += A(i, k) * B(j, k);
            C(i, j) = s;
        }
    return C;
}
inline Matrix matmulOut_naive_tn(const Matrix& A, const Matrix& B)
{
    Matrix C(A.cols(), B.cols());
    for (size_t i = 0; i < A.cols(); i++)
        for (size_t j = 0; j < B.cols(); j++) {
            double s = 0;
            for (size_t k = 0; k < A.rows(); k++) s += A(k, i) * B(k, j);
            C(i, j) = s;
        }
    return C;
}

// 比较两个矩阵，返回最大误差
inline double maxAbsDiff(const Matrix& X, const Matrix& Y)
{
    if (X.rows() != Y.rows() || X.cols() != Y.cols()) return 1e308;
    double m = 0.0;
    for (size_t i = 0; i < X.rows(); i++)
        for (size_t j = 0; j < X.cols(); j++)
            m = std::max(m, std::fabs(X(i, j) - Y(i, j)));
    return m;
}
