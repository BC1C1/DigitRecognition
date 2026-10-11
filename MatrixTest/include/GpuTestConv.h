#pragma once
// ============================================================================
//  GpuTestConv.h —— 卷积通路的对拍
//
//  布局约定（全部来自 Matrix.cu 的注释和实现）：
//      im2col:   ga (N, C*H*W)            -> gb ((N*OH*OW), C*K*K)
//      col2im:   ga ((N*OH*OW), C*K*K)    -> gb (N, C*H*W)   累加，gb 必须预清零
//      convGemm: A((N*OH*OW),(I*K*K)), kernel(O,(I*K*K)), bias(1,O)
//                                          -> out (N, O*OH*OW)
//                写入位置 out[i + o*OH*OW + n*OH*OW*O]
//      reshape:  grad (N, O*OH*OW)        -> dP ((N*OH*OW), O)
//      maxpool:  x (N, C*H*W)             -> out (N, C*OH*OW)，另出 maxIndex
//
//  这几条里最容易错的是**索引换算**：im2col 的下标是六重展开
//  （n,oh,ow,c,h,w -> 线性），col2im 是它的逆向散射，convGemm 的输出
//  位置又和 im2col 的输入布局不一样。所以对拍要覆盖：
//      padding=0 / 1 / 2，stride=1 / 2 / 3，K=1 / 2 / 3 / 5
//      W < K（靠 padding 补足）、C>1、N>1
//
//  【测试清单】
//      C1  launchIm2col
//      C2  launchCol2im（累加语义 + 预清零）
//      C3  im2col 与 col2im 互为伴随：<im2col(X),G> == <X,col2im(G)>
//      C4  launchConv2DGemmKernel
//      C5  launchConv2DBackwardReshape
//      C6  launchMaxPool2DForward（值 + 索引）
//      C7  launchMaxPool2DBackward（散射累加）
// ============================================================================

#include "GpuTestCommon.h"

namespace gputest {

struct ConvCase {
    size_t N, C, H, W, K, stride, padding;
    const char* tag;
};

static const ConvCase kConvCases[] = {
    { 1, 1,  4,  4, 3, 1, 0, "baseline" },
    { 1, 1,  4,  4, 3, 1, 1, "same padding" },
    { 1, 1,  4,  4, 2, 2, 0, "stride=K 不重叠" },
    { 2, 3,  5,  5, 3, 1, 1, "多 batch 多通道" },
    { 1, 2,  7,  7, 3, 2, 1, "stride=2 + padding" },
    { 1, 1,  1,  1, 1, 1, 0, "退化到 1x1" },
    { 1, 1,  2,  2, 3, 1, 1, "W < K 靠 padding 补足" },
    { 1, 2,  5,  5, 3, 1, 2, "padding=2" },
    { 1, 1, 28, 28, 5, 2, 2, "MNIST 真实量级" },
};

inline size_t outDim(size_t v, size_t K, size_t stride, size_t padding)
{
    return (v + 2 * padding - K) / stride + 1;
}

// ---------------------------------------------------------------------------
// host 端参照实现
// ---------------------------------------------------------------------------
inline std::vector<float> refIm2col(const std::vector<float>& X, const ConvCase& cs, size_t OH, size_t OW)
{
    const size_t k2 = cs.K * cs.K, Ck2 = cs.C * k2;
    const size_t HW = cs.H * cs.W, CHW = cs.C * HW;
    const size_t OHOW = OH * OW;
    std::vector<float> B(cs.N * OHOW * Ck2, 0.0f);

    for (size_t n = 0; n < cs.N; ++n)
    for (size_t oh = 0; oh < OH; ++oh)
    for (size_t ow = 0; ow < OW; ++ow)
    for (size_t c = 0; c < cs.C; ++c)
    for (size_t h = 0; h < cs.K; ++h)
    for (size_t w = 0; w < cs.K; ++w) {
        const long th = (long)(oh * cs.stride + h) - (long)cs.padding;
        const long tw = (long)(ow * cs.stride + w) - (long)cs.padding;
        float v = 0.0f;
        if (th >= 0 && th < (long)cs.H && tw >= 0 && tw < (long)cs.W)
            v = X[(size_t)th * cs.W + (size_t)tw + c * HW + n * CHW];
        const size_t bidx = w + h * cs.K + c * k2 + ow * Ck2 + oh * OW * Ck2 + n * OHOW * Ck2;
        B[bidx] = v;
    }
    return B;
}

inline std::vector<float> refCol2im(const std::vector<float>& A, const ConvCase& cs, size_t OH, size_t OW)
{
    const size_t k2 = cs.K * cs.K, Ck2 = cs.C * k2;
    const size_t HW = cs.H * cs.W, CHW = cs.C * HW;
    const size_t OHOW = OH * OW;
    std::vector<float> B(cs.N * CHW, 0.0f);

    for (size_t n = 0; n < cs.N; ++n)
    for (size_t oh = 0; oh < OH; ++oh)
    for (size_t ow = 0; ow < OW; ++ow)
    for (size_t c = 0; c < cs.C; ++c)
    for (size_t h = 0; h < cs.K; ++h)
    for (size_t w = 0; w < cs.K; ++w) {
        const long th = (long)(oh * cs.stride + h) - (long)cs.padding;
        const long tw = (long)(ow * cs.stride + w) - (long)cs.padding;
        if (th < 0 || th >= (long)cs.H || tw < 0 || tw >= (long)cs.W) continue;
        const size_t aidx = w + h * cs.K + c * k2 + ow * Ck2 + oh * OW * Ck2 + n * OHOW * Ck2;
        B[(size_t)th * cs.W + (size_t)tw + c * HW + n * CHW] += A[aidx];
    }
    return B;
}

// ---------------------------------------------------------------------------
// C1：im2col
// ---------------------------------------------------------------------------
inline void c1_im2col()
{
    printf("[C1] launchIm2col\n");

    for (const ConvCase& cs : kConvCases) {
        const size_t OH = outDim(cs.H, cs.K, cs.stride, cs.padding);
        const size_t OW = outDim(cs.W, cs.K, cs.stride, cs.padding);
        const size_t k2 = cs.K * cs.K;

        Matrix X = makeMatrix(cs.N, cs.C * cs.H * cs.W, 901);
        Matrix B(cs.N * OH * OW, cs.C * k2);
        upload(B, std::vector<float>(B.size(), -999.0f));

        launchIm2col(X.data_gpu(), B.data_gpu(), cs.N, cs.C, cs.H, cs.W,
                     OH, OW, cs.K, cs.stride, cs.padding);
        if (!runAndCheck("launchIm2col")) { G_CHECK(false, "[%s] kernel 失败", cs.tag); continue; }

        const std::vector<float> want = refIm2col(download(X), cs, OH, OW);
        const std::vector<float> got = download(B);
        const double e = maxDiff(got, want);
        G_CHECK(e < kExactTol, "[%s] im2col 是纯搬运，应逐位精确；误差 %.3e（首个不同 %zu）",
                cs.tag, e, firstDiff(got, want, kExactTol));
    }
}

// ---------------------------------------------------------------------------
// C2：col2im。它做的是散射累加（atomicAdd），所以输入缓冲必须预清零 ——
//     这一条本身也值得测：先预填非零，结果应当 = 参照 + 预填值。
// ---------------------------------------------------------------------------
inline void c2_col2im()
{
    printf("[C2] launchCol2im（累加语义，输出需预清零）\n");

    for (const ConvCase& cs : kConvCases) {
        const size_t OH = outDim(cs.H, cs.K, cs.stride, cs.padding);
        const size_t OW = outDim(cs.W, cs.K, cs.stride, cs.padding);
        const size_t k2 = cs.K * cs.K;

        Matrix A = makeMatrix(cs.N * OH * OW, cs.C * k2, 1001);
        Matrix B(cs.N, cs.C * cs.H * cs.W);
        zeroDevice(B);

        launchCol2im(A.data_gpu(), B.data_gpu(), cs.N, cs.C, cs.H, cs.W,
                     OH, OW, cs.K, cs.stride, cs.padding);
        if (!runAndCheck("launchCol2im")) { G_CHECK(false, "[%s] kernel 失败", cs.tag); continue; }

        const std::vector<float> want = refCol2im(download(A), cs, OH, OW);
        const std::vector<float> got = download(B);
        const double e = maxDiff(got, want);
        G_CHECK(e < kFloatTol, "[%s] col2im 误差 %.3e（首个不同 %zu）",
                cs.tag, e, firstDiff(got, want));
    }
}

// ---------------------------------------------------------------------------
// C3：伴随检验。<im2col(X), G> == <X, col2im(G)> 对任意 G 都成立 ——
//     这条不依赖任何具体数值写法，是两个算子互相定义的关系。
//     上次那批测试里它抓出过问题，所以这一条保留。
// ---------------------------------------------------------------------------
inline void c3_adjoint()
{
    printf("[C3] 伴随检验 <im2col(X),G> == <X,col2im(G)>\n");

    for (const ConvCase& cs : kConvCases) {
        const size_t OH = outDim(cs.H, cs.K, cs.stride, cs.padding);
        const size_t OW = outDim(cs.W, cs.K, cs.stride, cs.padding);
        const size_t M = cs.N * OH * OW, V = cs.C * cs.K * cs.K;

        Matrix X = makeMatrix(cs.N, cs.C * cs.H * cs.W, 1101);
        Matrix G = makeMatrix(M, V, 1102);
        Matrix A(M, V);
        Matrix Xg(cs.N, cs.C * cs.H * cs.W);
        zeroDevice(Xg);

        launchIm2col(X.data_gpu(), A.data_gpu(), cs.N, cs.C, cs.H, cs.W,
                     OH, OW, cs.K, cs.stride, cs.padding);
        launchCol2im(G.data_gpu(), Xg.data_gpu(), cs.N, cs.C, cs.H, cs.W,
                     OH, OW, cs.K, cs.stride, cs.padding);
        if (!runAndCheck("im2col/col2im 伴随")) { G_CHECK(false, "[%s] kernel 失败", cs.tag); continue; }

        const std::vector<float> hA = download(A), hG = download(G);
        const std::vector<float> hX = download(X), hXg = download(Xg);

        double lhs = 0.0, rhs = 0.0;
        for (size_t i = 0; i < M * V; ++i) lhs += (double)hA[i] * (double)hG[i];
        for (size_t i = 0; i < hX.size(); ++i) rhs += (double)hX[i] * (double)hXg[i];

        const double tol = kFloatTol * std::max(1.0, std::max(std::fabs(lhs), std::fabs(rhs)));
        G_CHECK(std::fabs(lhs - rhs) < tol,
                "[%s] 内积不等：%.6f vs %.6f，差 %.3e", cs.tag, lhs, rhs, std::fabs(lhs - rhs));
    }
}

// ---------------------------------------------------------------------------
// C4：convGemm —— A((N*OH*OW),(I*K*K)) x kernel(O,(I*K*K))^T + bias
//     输出布局 (N, O*OH*OW)，写入位置 i + o*OH*OW + n*OH*OW*O
// ---------------------------------------------------------------------------
inline void c4_conv_gemm()
{
    printf("[C4] launchConv2DGemmKernel\n");

    struct Case { size_t N, OH, OW, I, O, K; const char* tag; };
    const Case cases[] = {
        { 1,  2,  2, 1, 1, 3, "最小" },
        { 1,  4,  4, 1, 8, 3, "MNIST conv 层" },
        { 2,  3,  3, 2, 4, 3, "多 batch 多通道" },
        { 1,  1,  1, 1, 1, 1, "退化 1x1" },
        { 1,  7,  5, 3, 6, 2, "非方形 OH/OW" },
        { 1, 14, 14, 8, 3, 2, "pool 之后的真实量级" },
    };

    for (const Case& c : cases) {
        const size_t IKK = c.I * c.K * c.K;
        Matrix A = makeMatrix(c.N * c.OH * c.OW, IKK, 1201);
        Matrix ker = makeMatrix(c.O, IKK, 1202);
        Matrix bias = makeMatrix(1, c.O, 1203);
        Matrix out(c.N, c.O * c.OH * c.OW);
        upload(out, std::vector<float>(out.size(), -999.0f));

        launchConv2DGemmKernel(A.data_gpu(), ker.data_gpu(), bias.data_gpu(), out.data_gpu(),
                               c.N, c.OH, c.OW, c.I, c.O, c.K);
        if (!runAndCheck("launchConv2DGemmKernel")) { G_CHECK(false, "[%s] kernel 失败", c.tag); continue; }

        const std::vector<float> hA = download(A), hK = download(ker), hB = download(bias);
        std::vector<float> want(out.size(), 0.0f);
        for (size_t n = 0; n < c.N; ++n)
        for (size_t i = 0; i < c.OH * c.OW; ++i)
        for (size_t o = 0; o < c.O; ++o) {
            double s = 0.0;
            for (size_t k = 0; k < IKK; ++k)
                s += (double)hA[(n * c.OH * c.OW + i) * IKK + k] * (double)hK[o * IKK + k];
            s += (double)hB[o];
            want[i + o * c.OH * c.OW + n * c.OH * c.OW * c.O] = (float)s;
        }

        const std::vector<float> got = download(out);
        const double e = maxDiff(got, want);
        G_CHECK(e < kFloatTol, "[%s] 误差 %.3e（首个不同 %zu）", c.tag, e, firstDiff(got, want));
    }
}

// ---------------------------------------------------------------------------
// C5：backward 的逆重排  (N, O*OH*OW) -> ((N*OH*OW), O)
// ---------------------------------------------------------------------------
inline void c5_backward_reshape()
{
    printf("[C5] launchConv2DBackwardReshape\n");

    struct Case { size_t N, O, OH, OW; const char* tag; };
    const Case cases[] = {
        { 1, 1, 1, 1, "退化" },
        { 1, 8, 4, 4, "MNIST conv" },
        { 2, 4, 3, 5, "多 batch 非方形" },
        { 1, 3, 14, 14, "pool 之后" },
    };

    for (const Case& c : cases) {
        Matrix grad = makeMatrix(c.N, c.O * c.OH * c.OW, 1301);
        Matrix dP(c.N * c.OH * c.OW, c.O);
        upload(dP, std::vector<float>(dP.size(), -999.0f));

        launchConv2DBackwardReshape(grad.data_gpu(), dP.data_gpu(), c.N, c.O, c.OH, c.OW);
        if (!runAndCheck("launchConv2DBackwardReshape")) { G_CHECK(false, "[%s] kernel 失败", c.tag); continue; }

        const std::vector<float> hg = download(grad);
        std::vector<float> want(dP.size(), 0.0f);
        for (size_t n = 0; n < c.N; ++n)
        for (size_t i = 0; i < c.OH * c.OW; ++i)
        for (size_t o = 0; o < c.O; ++o)
            want[o + i * c.O + n * c.O * c.OH * c.OW] =
                hg[i + o * c.OH * c.OW + n * c.OH * c.OW * c.O];

        const std::vector<float> got = download(dP);
        const double e = maxDiff(got, want);
        G_CHECK(e < kExactTol, "[%s] 纯搬运应逐位精确，误差 %.3e", c.tag, e);
    }
}

// ---------------------------------------------------------------------------
// C6：maxpool forward —— 值和索引都要对。
//     索引错的表现很隐蔽：forward 的输出完全正确，但 backward 会把梯度
//     累加到错误的位置上，训练照样跑、loss 照样降（只是降得慢）。
//     所以这里把 maxIndex 单独对拍。
// ---------------------------------------------------------------------------
inline void c6_maxpool_forward()
{
    printf("[C6] launchMaxPool2DForward（值 + maxIndex）\n");

    struct Case { size_t N, C, H, W, K, stride; const char* tag; };
    const Case cases[] = {
        { 1, 1,  4,  4, 2, 2, "baseline" },
        { 1, 1,  5,  5, 3, 2, "非整除" },
        { 2, 3,  6,  6, 2, 2, "多 batch 多通道" },
        { 1, 1,  2,  2, 2, 2, "恰好一个窗口" },
        { 1, 1,  1,  1, 1, 1, "退化" },
        { 1, 8, 28, 28, 2, 2, "MNIST pool 层" },
    };

    for (const Case& c : cases) {
        const size_t OH = (c.H - c.K) / c.stride + 1;
        const size_t OW = (c.W - c.K) / c.stride + 1;
        const size_t cnt = c.N * c.C * OH * OW;

        Matrix x = makeMatrix(c.N, c.C * c.H * c.W, 1401);
        Matrix out(c.N, c.C * OH * OW);
        upload(out, std::vector<float>(out.size(), -999.0f));

        size_t* dIdx = nullptr;
        if (cudaMalloc((void**)&dIdx, sizeof(size_t) * cnt) != cudaSuccess) {
            G_CHECK(false, "[%s] maxIndex 分配失败", c.tag);
            continue;
        }

        launchMaxPool2DForward(x.data_gpu(), out.data_gpu(), c.N, c.C, c.H, c.W, c.K, c.stride, dIdx);
        const bool ok = runAndCheck("launchMaxPool2DForward");

        std::vector<size_t> hIdx(cnt, 0);
        cudaMemcpy(hIdx.data(), dIdx, sizeof(size_t) * cnt, cudaMemcpyDeviceToHost);
        cudaFree(dIdx);

        if (!ok) { G_CHECK(false, "[%s] kernel 失败", c.tag); continue; }

        const std::vector<float> hx = download(x);
        std::vector<float> wantOut(cnt, 0.0f);
        std::vector<size_t> wantIdx(cnt, 0);
        for (size_t n = 0; n < c.N; ++n)
        for (size_t ch = 0; ch < c.C; ++ch)
        for (size_t oh = 0; oh < OH; ++oh)
        for (size_t ow = 0; ow < OW; ++ow) {
            const size_t headh = oh * c.stride, headw = ow * c.stride;
            size_t best = headw + headh * c.W + ch * c.H * c.W + n * c.H * c.W * c.C;
            float bv = hx[best];
            for (size_t h = 0; h < c.K; ++h)
            for (size_t w = 0; w < c.K; ++w) {
                const size_t p = (headw + w) + (headh + h) * c.W + ch * c.H * c.W + n * c.H * c.W * c.C;
                if (hx[p] > bv) { bv = hx[p]; best = p; }   // 严格大于 -> 取第一个最大值
            }
            const size_t o = ow + oh * OW + ch * OH * OW + n * c.C * OH * OW;
            wantOut[o] = bv;
            wantIdx[o] = best;
        }

        const std::vector<float> got = download(out);
        G_CHECK(maxDiff(got, wantOut) < kFloatTol, "[%s] 输出误差 %.3e", c.tag, maxDiff(got, wantOut));

        size_t badIdx = 0;
        for (size_t i = 0; i < cnt; ++i) if (hIdx[i] != wantIdx[i]) ++badIdx;
        if (badIdx) {
            size_t first = 0;
            while (first < cnt && hIdx[first] == wantIdx[first]) ++first;
            G_CHECK(false, "[%s] maxIndex 有 %zu 个不对（首个位置 %zu：得到 %zu，期望 %zu）",
                    c.tag, badIdx, first, hIdx[first], wantIdx[first]);
        }
    }
}

// ---------------------------------------------------------------------------
// C7：maxpool backward —— 散射累加，输出必须预清零
// ---------------------------------------------------------------------------
inline void c7_maxpool_backward()
{
    printf("[C7] launchMaxPool2DBackward（散射累加）\n");

    struct Case { size_t N, C, H, W, K, stride; const char* tag; };
    const Case cases[] = {
        { 1, 1, 4, 4, 2, 2, "baseline" },
        { 2, 3, 6, 6, 2, 2, "多 batch 多通道" },
        { 1, 2, 5, 5, 3, 2, "非整除" },
        { 1, 1, 2, 2, 2, 2, "一个窗口（重叠）" },
    };

    for (const Case& c : cases) {
        const size_t OH = (c.H - c.K) / c.stride + 1;
        const size_t OW = (c.W - c.K) / c.stride + 1;
        const size_t cnt = c.N * c.C * OH * OW;

        Matrix x = makeMatrix(c.N, c.C * c.H * c.W, 1501);
        Matrix grad = makeMatrix(c.N, c.C * OH * OW, 1502);
        Matrix dx(c.N, c.C * c.H * c.W);
        zeroDevice(dx);

        size_t* dIdx = nullptr;
        if (cudaMalloc((void**)&dIdx, sizeof(size_t) * cnt) != cudaSuccess) {
            G_CHECK(false, "[%s] maxIndex 分配失败", c.tag);
            continue;
        }

        Matrix dummy(c.N, c.C * OH * OW);
        launchMaxPool2DForward(x.data_gpu(), dummy.data_gpu(), c.N, c.C, c.H, c.W, c.K, c.stride, dIdx);
        launchMaxPool2DBackward(grad.data_gpu(), dx.data_gpu(), c.N, c.C, c.H, c.W, c.K, c.stride, dIdx);
        const bool ok = runAndCheck("maxPool2D backward");

        std::vector<size_t> hIdx(cnt, 0);
        cudaMemcpy(hIdx.data(), dIdx, sizeof(size_t) * cnt, cudaMemcpyDeviceToHost);
        cudaFree(dIdx);

        if (!ok) { G_CHECK(false, "[%s] kernel 失败", c.tag); continue; }

        const std::vector<float> hg = download(grad);
        std::vector<float> want(dx.size(), 0.0f);
        for (size_t k = 0; k < cnt; ++k) want[hIdx[k]] += hg[k];

        const std::vector<float> got = download(dx);
        const double e = maxDiff(got, want);
        G_CHECK(e < kFloatTol, "[%s] 误差 %.3e（首个不同 %zu）", c.tag, e, firstDiff(got, want));
    }
}

// ---------------------------------------------------------------------------
// 入口
// ---------------------------------------------------------------------------
inline int runConvTests()
{
    printf("\n===== 卷积通路 =====\n");
    c1_im2col();
    c2_col2im();
    c3_adjoint();
    c4_conv_gemm();
    c5_backward_reshape();
    c6_maxpool_forward();
    c7_maxpool_backward();
    return fails();
}

} // namespace gputest
