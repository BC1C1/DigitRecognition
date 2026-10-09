#pragma once
// ============================================================================
//  Col2ImTest.h —— col2im 的正确性测试（含 padding）
//
//  【规格】col2im 是 im2col 的【转置（伴随）算子】：
//      输入 (N*OH*OW, C*K*K)  ->  输出 (N, C*H*W)
//          OH = (H + 2p - K)/stride + 1，OW 同理
//      两条硬性要求：
//        1) 输出是【累加】的（stride < K 时窗口重叠，一个像素被多个 patch 命中）
//        2) padding 区的贡献必须【丢弃】（那里是虚拟像素，梯度没有落点）
//
//  【测试清单】
//      C1  恒等式 col2im(im2col(X)) == X ⊙ count：p=0 / p=1 两组 count 都手算硬编码
//      C2  伴随检验 <im2col(X), G> == <X, col2im(G)>：对任意 G 成立，定义性
//      C3  多形状扫描（含 padding），count 由反向算法独立现算
//      C4  直接构造 dA：padding 位置的贡献必须为 0；col2im(全 1) 等于覆盖次数
// ============================================================================

#include "TestCommon.h"

namespace col2im_test {

// ---------------------------------------------------------------------------
// 工具
// ---------------------------------------------------------------------------
// 带 padding 的输出尺寸。用 int64_t 算，免得 (H - K) 这种写法在 H < K 时回绕
inline int64_t outDim(int64_t v, int64_t K, int64_t stride, int64_t padding)
{
    return (v + 2 * padding - K) / stride + 1;
}

// 反向算法：按 padded 坐标枚举，数某个真实像素被几个 patch 命中
inline size_t countPixel(size_t h, size_t w, size_t OH, size_t OW,
                         size_t K, size_t stride, size_t padding)
{
    size_t cnt = 0;
    for (size_t oh = 0; oh < OH; ++oh)
        for (size_t ow = 0; ow < OW; ++ow)
            for (size_t kh = 0; kh < K; ++kh)
                for (size_t kw = 0; kw < K; ++kw)
                    if (oh * stride + kh == h + padding &&
                        ow * stride + kw == w + padding) ++cnt;
    return cnt;
}

// 全 1 矩阵。
// ⚠ 不能用 fill(1.0)：这份 TypeDefine.h 里 fill(double) 是 const 版，返回新矩阵
//   而不是原地写（主项目那份才有原地的 void fill(double)），用了等于没填。
inline Matrix ones(size_t r, size_t c)
{
    Matrix m(r, c);
    for (size_t i = 0; i < r; ++i)
        for (size_t j = 0; j < c; ++j) m(i, j) = 1.0;
    return m;
}

// ---------------------------------------------------------------------------
// C1：恒等式。两组 count 都手算：
//   p=0（4x4 K=3 s=1）：角 1、边 2、内部 4，总和 36 = 4 个 patch × 9
//   p=1（same）      ：窗口可以"悬"到图外，真实像素被覆盖的次数变多，
//                      总和 100；16 个 patch 共 144 次落格，另 44 次落在 padding 区
// ---------------------------------------------------------------------------
inline void c1_identity()
{
    printf("[C1] 恒等式 col2im(im2col(X)) == X ⊙ count（p=0 与 p=1）\n");

    static const double cntP0[16] = {
        1, 2, 2, 1,
        2, 4, 4, 2,
        2, 4, 4, 2,
        1, 2, 2, 1
    };
    static const double cntP1[16] = {
        4, 6, 6, 4,
        6, 9, 9, 6,
        6, 9, 9, 6,
        4, 6, 6, 4
    };

    struct Case { size_t padding; const double* cnt; double total; const char* tag; };
    const Case cases[] = {
        { 0, cntP0,  36.0, "p=0" },
        { 1, cntP1, 100.0, "p=1(same)" },
    };

    for (const Case& cs : cases) {
        const size_t H = 4, W = 4, K = 3, stride = 1;
        const size_t OH = (size_t)outDim((int64_t)H, (int64_t)K, (int64_t)stride, (int64_t)cs.padding);
        const size_t OW = (size_t)outDim((int64_t)W, (int64_t)K, (int64_t)stride, (int64_t)cs.padding);

        // 先确认硬编码 count 与反向算法一致，免得 count 错了让恒等式假绿
        bool countOk = true;
        double total = 0.0;
        for (size_t h = 0; h < H; ++h)
            for (size_t w = 0; w < W; ++w) {
                const double c = (double)countPixel(h, w, OH, OW, K, stride, cs.padding);
                total += c;
                if (!dstest::exact(c, cs.cnt[h * W + w])) {
                    countOk = false;
                    printf("  [FAIL] %s count(%zu,%zu)：硬编码 %.0f，反向算法 %.0f\n",
                           cs.tag, h, w, cs.cnt[h * W + w], c);
                }
            }
        T_CHECK(countOk, "%s 的 count 两种算法不一致（先修 count，再看恒等式）", cs.tag);
        T_CHECK(dstest::exact(total, cs.total),
                "%s count 总和应为 %.0f，实得 %.0f", cs.tag, cs.total, total);

        Matrix X = Matrix::randnMatrix(1, H * W);
        Matrix A = X.im2col(1, 1, H, W, K, stride, cs.padding);
        T_CHECK(A.rows() == OH * OW && A.cols() == K * K,
                "%s im2col 形状：期望 %zux%zu，实得 %zux%zu",
                cs.tag, OH * OW, K * K, A.rows(), A.cols());
        if (A.rows() != OH * OW || A.cols() != K * K) continue;

        Matrix back = A.col2im(1, 1, H, W, K, stride, cs.padding);
        T_CHECK(back.rows() == 1 && back.cols() == H * W,
                "%s col2im 形状：期望 1x%zu，实得 %zux%zu", cs.tag, H * W, back.rows(), back.cols());
        if (back.rows() != 1 || back.cols() != H * W) continue;

        size_t bad = 0;
        for (size_t i = 0; i < H * W; ++i) {
            const double want = X(0, i) * cs.cnt[i];
            if (!dstest::close(back(0, i), want, dstest::kFloatTol)) {
                ++bad;
                if (bad <= 4)
                    printf("  [FAIL] %s back(0,%zu)：期望 %.6f（count=%.0f），实得 %.6f\n",
                           cs.tag, i, want, cs.cnt[i], back(0, i));
            }
        }
        if (bad) {
            ++dstest::fails();
            printf("  [FAIL] %s：%zu 个像素里有 %zu 个不等于 X ⊙ count\n", cs.tag, H * W, bad);
        }
    }
}

// ---------------------------------------------------------------------------
// C2：伴随检验。im2col 是线性映射，col2im 必须是它的转置：
//       <im2col(X), G> == <X, col2im(G)>
//     对任意 G 成立，不依赖 count，也不要求 X 特殊。
// ---------------------------------------------------------------------------
inline void c2_adjoint()
{
    printf("[C2] 伴随检验 <im2col(X), G> == <X, col2im(G)>\n");

    struct Case { size_t N, C, H, W, K, stride, padding; const char* tag; };
    const Case cases[] = {
        { 1, 1, 4, 4, 3, 1, 0, "无 padding" },
        { 1, 1, 4, 4, 3, 1, 1, "same padding" },
        { 1, 1, 4, 4, 2, 2, 0, "stride=K 完全不重叠" },
        { 2, 3, 5, 5, 3, 2, 1, "多 batch 多通道 + padding" },
        { 1, 2, 7, 7, 3, 1, 2, "padding=2" },
    };

    for (const Case& cs : cases) {
        const size_t OH = (size_t)outDim((int64_t)cs.H, (int64_t)cs.K, (int64_t)cs.stride, (int64_t)cs.padding);
        const size_t OW = (size_t)outDim((int64_t)cs.W, (int64_t)cs.K, (int64_t)cs.stride, (int64_t)cs.padding);
        const size_t M = cs.N * OH * OW, V = cs.C * cs.K * cs.K;

        Matrix X = Matrix::randnMatrix(cs.N, cs.C * cs.H * cs.W);
        Matrix G = Matrix::randnMatrix(M, V);

        Matrix A = X.im2col(cs.N, cs.C, cs.H, cs.W, cs.K, cs.stride, cs.padding);
        T_CHECK(A.rows() == M && A.cols() == V,
                "[%s] im2col 形状：期望 %zux%zu，实得 %zux%zu",
                cs.tag, M, V, A.rows(), A.cols());
        if (A.rows() != M || A.cols() != V) continue;

        Matrix Xg = G.col2im(cs.N, cs.C, cs.H, cs.W, cs.K, cs.stride, cs.padding);
        T_CHECK(Xg.rows() == cs.N && Xg.cols() == cs.C * cs.H * cs.W,
                "[%s] col2im 形状：期望 %zux%zu，实得 %zux%zu",
                cs.tag, cs.N, cs.C * cs.H * cs.W, Xg.rows(), Xg.cols());
        if (Xg.rows() != cs.N || Xg.cols() != cs.C * cs.H * cs.W) continue;

        double lhs = 0.0, rhs = 0.0;
        for (size_t i = 0; i < M; ++i)
            for (size_t j = 0; j < V; ++j)
                lhs += A(i, j) * G(i, j);
        for (size_t i = 0; i < cs.N; ++i)
            for (size_t j = 0; j < cs.C * cs.H * cs.W; ++j)
                rhs += X(i, j) * Xg(i, j);

        T_CHECK(dstest::close(lhs, rhs, dstest::kFloatTol),
                "[%s] 内积不等：<im2col(X),G>=%.6f，<X,col2im(G)>=%.6f，差 %.3e",
                cs.tag, lhs, rhs, std::fabs(lhs - rhs));
    }
}

// ---------------------------------------------------------------------------
// C3：多形状的恒等式扫描。count 用反向算法现算，不依赖 col2im。
// ---------------------------------------------------------------------------
inline void c3_shapes()
{
    printf("[C3] 多形状恒等式扫描（含 padding 与退化形状）\n");

    struct Case { size_t N, C, H, W, K, stride, padding; };
    const Case cases[] = {
        { 1, 1,  1,  1, 1, 1, 0 },
        { 1, 1,  4,  4, 3, 1, 0 },
        { 1, 1,  4,  4, 3, 1, 1 },
        { 1, 1,  4,  4, 2, 2, 0 },
        { 2, 3,  5,  5, 3, 1, 1 },
        { 1, 2,  7,  7, 3, 2, 1 },
        { 1, 1,  2,  2, 3, 1, 1 },   // W < K，靠 padding 补足
        { 1, 1, 28, 28, 5, 2, 2 },
    };

    for (const Case& cs : cases) {
        const size_t OH = (size_t)outDim((int64_t)cs.H, (int64_t)cs.K, (int64_t)cs.stride, (int64_t)cs.padding);
        const size_t OW = (size_t)outDim((int64_t)cs.W, (int64_t)cs.K, (int64_t)cs.stride, (int64_t)cs.padding);
        const size_t M = cs.N * OH * OW, V = cs.C * cs.K * cs.K;

        Matrix X = Matrix::randnMatrix(cs.N, cs.C * cs.H * cs.W);
        Matrix A = X.im2col(cs.N, cs.C, cs.H, cs.W, cs.K, cs.stride, cs.padding);
        T_CHECK(A.rows() == M && A.cols() == V,
                "im2col 形状 N=%zu C=%zu H=%zu W=%zu K=%zu s=%zu p=%zu：期望 %zux%zu，实得 %zux%zu",
                cs.N, cs.C, cs.H, cs.W, cs.K, cs.stride, cs.padding, M, V, A.rows(), A.cols());
        if (A.rows() != M || A.cols() != V) continue;

        Matrix back = A.col2im(cs.N, cs.C, cs.H, cs.W, cs.K, cs.stride, cs.padding);
        T_CHECK(back.rows() == cs.N && back.cols() == cs.C * cs.H * cs.W,
                "col2im 形状：期望 %zux%zu，实得 %zux%zu",
                cs.N, cs.C * cs.H * cs.W, back.rows(), back.cols());
        if (back.rows() != cs.N || back.cols() != cs.C * cs.H * cs.W) continue;

        size_t bad = 0;
        for (size_t c = 0; c < cs.C; ++c)
            for (size_t h = 0; h < cs.H; ++h)
                for (size_t w = 0; w < cs.W; ++w) {
                    const double cnt = (double)countPixel(h, w, OH, OW, cs.K, cs.stride, cs.padding);
                    const size_t idx = c * cs.H * cs.W + h * cs.W + w;
                    for (size_t n = 0; n < cs.N; ++n) {
                        const double want = X(n, idx) * cnt;   // 每张图不同，别用 X(0,·)
                        if (!dstest::close(back(n, idx), want, dstest::kFloatTol)) {
                            ++bad;
                            if (bad <= 3)
                                printf("  [FAIL] n=%zu c=%zu h=%zu w=%zu：期望 %.6f（count=%.0f），实得 %.6f\n",
                                       n, c, h, w, want, cnt, back(n, idx));
                        }
                    }
                }
        if (bad) {
            ++dstest::fails();
            printf("  [FAIL] 形状 N=%zu C=%zu H=%zu W=%zu K=%zu s=%zu p=%zu：%zu 个像素不对\n",
                   cs.N, cs.C, cs.H, cs.W, cs.K, cs.stride, cs.padding, bad);
        }
    }
}

// ---------------------------------------------------------------------------
// C4：直接构造 dA，钉死两件事
//   ① padding 位置的贡献必须为 0（虚拟像素没有落点）
//   ② col2im(全 1) == 覆盖次数（纯计数语义，不掺 im2col）
// ---------------------------------------------------------------------------
inline void c4_padding_dropped()
{
    printf("[C4] padding 位置的贡献必须为 0；col2im(全 1) == 覆盖次数\n");

    // H=W=2, K=3, s=1, p=1 -> padded 4x4，OH=OW=2，共 4 个 patch
    const size_t H = 2, W = 2, K = 3, stride = 1, padding = 1;
    const size_t OH = 2, OW = 2, M = OH * OW, V = K * K;

    // ① (oh,ow)=(0,0) 的 (kh,kw)=(0,0) 映射到 padded (0,0)：在 padding 区
    Matrix dA(M, V);
    dA(0, 0) = 1.0;
    Matrix dX = dA.col2im(1, 1, H, W, K, stride, padding);
    T_CHECK(dX.rows() == 1 && dX.cols() == H * W,
            "形状：期望 1x%zu，实得 %zux%zu", H * W, dX.rows(), dX.cols());
    if (dX.rows() == 1 && dX.cols() == H * W) {
        double s = 0.0;
        for (size_t j = 0; j < H * W; ++j) s += std::fabs(dX(0, j));
        T_CHECK(dstest::exact(s, 0.0),
                "padding 位置的 1 不该产生任何 dX，实得绝对值总和 %.6f", s);
    }

    // 对照：(kh,kw)=(1,1) 映射到 padded (1,1)，对应真实像素 (0,0)
    Matrix dA2(M, V);
    dA2(0, 1 * K + 1) = 1.0;
    Matrix dX2 = dA2.col2im(1, 1, H, W, K, stride, padding);
    bool ok = (dX2.rows() == 1 && dX2.cols() == H * W);
    if (!ok) {
        T_CHECK(false, "对照形状：期望 1x%zu，实得 %zux%zu", H * W, dX2.rows(), dX2.cols());
    } else {
        for (size_t j = 0; j < H * W; ++j) {
            const double want = (j == 0) ? 1.0 : 0.0;
            if (!dstest::exact(dX2(0, j), want)) {
                ok = false;
                printf("  [FAIL] 对照：dX(0,%zu) 期望 %.1f，实得 %.6f\n", j, want, dX2(0, j));
            }
        }
        T_CHECK(ok, "真实像素位置的贡献没落到 (0,0)");
    }

    // ② col2im(全 1) == 每个像素的覆盖次数
    Matrix all1 = ones(M, V);
    Matrix cnt = all1.col2im(1, 1, H, W, K, stride, padding);
    if (cnt.rows() != 1 || cnt.cols() != H * W) {
        T_CHECK(false, "col2im(全 1) 形状：期望 1x%zu，实得 %zux%zu", H * W, cnt.rows(), cnt.cols());
    } else {
        for (size_t h = 0; h < H; ++h)
            for (size_t w = 0; w < W; ++w) {
                const double want = (double)countPixel(h, w, OH, OW, K, stride, padding);
                T_CHECK(dstest::exact(cnt(0, h * W + w), want),
                        "col2im(全 1)(%zu,%zu)：期望 %.0f，实得 %.0f",
                        h, w, want, cnt(0, h * W + w));
            }
    }
}

// ---------------------------------------------------------------------------
// 入口
// ---------------------------------------------------------------------------
inline int runCol2ImTests()
{
    dstest::resetFails();
    printf("\n===== col2im 测试开始 =====\n");
    c1_identity();
    c2_adjoint();
    c3_shapes();
    c4_padding_dropped();
    printf("===== col2im 测试结束：失败 %d 条 =====\n\n", dstest::fails());
    return dstest::fails();
}

} // namespace col2im_test
