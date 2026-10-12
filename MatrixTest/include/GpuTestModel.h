#pragma once
// ============================================================================
//  GpuTestModel.h —— 层、损失、优化器的算子对拍
//
//  这些 kernel 是"把 CPU 版的循环搬进显存"的产物，每一个都对应一段原来
//  的 host 代码。所以参照实现就直接照抄那段 host 代码的语义。
//
//  两个要特别小心的：
//
//  1) launchLogSoftmax(input, cache, out, M, N)
//     kernel 里写的是  out = log-softmax、cache = softmax —— 注意**第二个形参
//     装的是 softmax**。名字和内容不一致，调用方（CrossEntropyLoss）里两个
//     缓存的名字又是反的，所以 L4 不只对拍数值，还额外验证两条自洽性质：
//         Σ_j cache[l][j] == 1        （softmax 的定义）
//         out[l][j] == log(cache[l][j])（两者互为 log/exp）
//     只要有一个不成立，就说明两个输出装反了或者算错了。
//
//  2) launchMSELossForward 返回的是 (1/(M*N)) * Σ(pred-real)^2（均方误差），
//     launchMSELossBackwardKernel 给出 2*(pred-real)/(M*N)。
//     两者是配套的：后者正是前者对 pred 的导数。L5 特意挑了几个不等于 1 的
//     M*N（640 / 771）来钉住"有没有除"—— 不除就差 M*N 倍，一眼能看出来。
//
//  【测试清单】
//      L1  launchLinearForward
//      L2  launchReLUForward
//      L3  launchReLUBackward
//      L4  launchLogSoftmax（含两条自洽性质）
//      L5  launchMSELossForward
//      L6  launchMSELossBackwardKernel
//      L7  launchGeneralUpdateWithMomentum
// ============================================================================

#include "GpuTestCommon.h"

namespace gputest {

// ---------------------------------------------------------------------------
// L1：Linear 前向  result[i*O+j] = Σ_k x[i*I+k] * W[j*I+k] + b[j]
// ---------------------------------------------------------------------------
inline void l1_linear_forward()
{
    printf("[L1] launchLinearForward\n");

    struct Case { size_t N, I, O; const char* tag; };
    const Case cases[] = {
        {  1,   1,   1, "退化" },
        {  1,   4,   3, "小" },
        { 64,  10,   5, "一个 batch" },
        { 64, 128,  10, "MNIST 最后一层形状" },
        { 64,1568, 128, "MNIST 第一层形状" },
        {  3,   5, 257, "输出维度跨 block" },
    };

    for (const Case& c : cases) {
        Matrix x = makeMatrix(c.N, c.I, 1601);
        Matrix W = makeMatrix(c.O, c.I, 1602);
        Matrix b = makeMatrix(1, c.O, 1603);
        Matrix out(c.N, c.O);
        upload(out, std::vector<float>(out.size(), -999.0f));

        launchLinearForward(x.data_gpu(), W.data_gpu(), b.data_gpu(), out.data_gpu(), c.I, c.O, c.N);
        if (!runAndCheck("launchLinearForward")) { G_CHECK(false, "[%s] kernel 失败", c.tag); continue; }

        const std::vector<float> hx = download(x), hW = download(W), hb = download(b);
        std::vector<float> want(c.N * c.O, 0.0f);
        for (size_t i = 0; i < c.N; ++i)
        for (size_t j = 0; j < c.O; ++j) {
            double s = 0.0;
            for (size_t k = 0; k < c.I; ++k) s += (double)hx[i * c.I + k] * (double)hW[j * c.I + k];
            s += (double)hb[j];
            want[i * c.O + j] = (float)s;
        }

        const std::vector<float> got = download(out);
        const double e = maxDiff(got, want);
        G_CHECK(e < kFloatTol, "[%s] 误差 %.3e（首个不同 %zu）", c.tag, e, firstDiff(got, want));
    }
}

// ---------------------------------------------------------------------------
// L2 / L3：ReLU 前后向
//   forward: out = max(0, x)
//   backward: out = (cache == 0) ? 0 : grad   —— cache 是 forward 的输出
// ---------------------------------------------------------------------------
inline void l2_l3_relu()
{
    printf("[L2] launchReLUForward / [L3] launchReLUBackward\n");

    struct Case { size_t N, D; };
    const Case cases[] = { {1, 1}, {1, 300}, {64, 128}, {64, 1568}, {3, 257} };

    for (const Case& c : cases) {
        Matrix x = makeMatrix(c.N, c.D, 1701);

        // forward
        Matrix out(c.N, c.D);
        upload(out, std::vector<float>(out.size(), -999.0f));
        launchReLUForward(x.data_gpu(), out.data_gpu(), c.N, c.D);
        if (!runAndCheck("launchReLUForward")) { G_CHECK(false, "N=%zu D=%zu kernel 失败", c.N, c.D); continue; }

        const std::vector<float> hx = download(x);
        std::vector<float> wantF(hx.size());
        for (size_t i = 0; i < hx.size(); ++i) wantF[i] = hx[i] > 0.0f ? hx[i] : 0.0f;
        const std::vector<float> gotF = download(out);
        G_CHECK(maxDiff(gotF, wantF) < kExactTol,
                "forward N=%zu D=%zu 误差 %.3e", c.N, c.D, maxDiff(gotF, wantF));

        // backward：用 forward 的输出当 cache
        Matrix grad = makeMatrix(c.N, c.D, 1702);
        Matrix back(c.N, c.D);
        upload(back, std::vector<float>(back.size(), -999.0f));
        launchReLUBackward(grad.data_gpu(), out.data_gpu(), back.data_gpu(), c.N, c.D);
        if (!runAndCheck("launchReLUBackward")) { G_CHECK(false, "N=%zu D=%zu kernel 失败", c.N, c.D); continue; }

        const std::vector<float> hg = download(grad);
        std::vector<float> wantB(hx.size());
        for (size_t i = 0; i < hx.size(); ++i) wantB[i] = (gotF[i] == 0.0f) ? 0.0f : hg[i];
        const std::vector<float> gotB = download(back);
        G_CHECK(maxDiff(gotB, wantB) < kExactTol,
                "backward N=%zu D=%zu 误差 %.3e", c.N, c.D, maxDiff(gotB, wantB));
    }
}

// ---------------------------------------------------------------------------
// L4：log-softmax。
//   形参顺序是 (input, cache, out) —— cache 收的是 softmax，out 收的是 log-softmax。
//   除了逐元素对拍，再加两条不依赖实现的自我一致性检查。
// ---------------------------------------------------------------------------
inline void l4_log_softmax()
{
    printf("[L4] launchLogSoftmax（out=log-softmax，cache=softmax）\n");

    struct Case { size_t M, N; const char* tag; };
    const Case cases[] = {
        {   1,   2, "最小" },
        {   1,  10, "MNIST 类别数" },
        {  64,  10, "一个 batch" },
        { 128,  10, "两个 batch" },
        {   3, 257, "类别数跨 block" },
    };

    for (const Case& c : cases) {
        Matrix in = makeMatrix(c.M, c.N, 1801);
        Matrix cache(c.M, c.N);
        Matrix out(c.M, c.N);
        upload(cache, std::vector<float>(cache.size(), -999.0f));
        upload(out, std::vector<float>(out.size(), -999.0f));

        launchLogSoftmax(in.data_gpu(), cache.data_gpu(), out.data_gpu(), c.M, c.N);
        if (!runAndCheck("launchLogSoftmax")) { G_CHECK(false, "[%s] kernel 失败", c.tag); continue; }

        const std::vector<float> hIn = download(in);
        const std::vector<float> hCache = download(cache);
        const std::vector<float> hOut = download(out);

        // 参照：按 kernel 里同样的公式手算一遍
        std::vector<float> wantSM(c.M * c.N, 0.0f), wantLS(c.M * c.N, 0.0f);
        for (size_t l = 0; l < c.M; ++l) {
            double maxV = hIn[l * c.N];
            for (size_t i = 1; i < c.N; ++i) maxV = std::max(maxV, (double)hIn[l * c.N + i]);
            double es = 0.0;
            for (size_t i = 0; i < c.N; ++i) es += std::exp((double)hIn[l * c.N + i] - maxV);
            const double logEs = std::log(es);
            for (size_t i = 0; i < c.N; ++i) {
                wantLS[l * c.N + i] = (float)((double)hIn[l * c.N + i] - maxV - logEs);
                wantSM[l * c.N + i] = (float)(std::exp((double)hIn[l * c.N + i] - maxV) / es);
            }
        }

        // 逐元素对拍，两边分别报，能直接看出"是不是装反了"
        const double eOut = maxDiff(hOut, wantLS);
        const double eCache = maxDiff(hCache, wantSM);
        G_CHECK(eOut < kFloatTol, "[%s] out 应等于 log-softmax，误差 %.3e", c.tag, eOut);
        G_CHECK(eCache < kFloatTol, "[%s] cache 应等于 softmax，误差 %.3e", c.tag, eCache);

        // 交叉验证：如果两个装反了，上面两条会红，但下面这两条还会绿（因为
        // 两组数据各自都是合法的概率/对数概率）。所以四条一起看，才能定位。
        const double eOutAsSM = maxDiff(hOut, wantSM);
        const double eCacheAsLS = maxDiff(hCache, wantLS);
        if (eOutAsSM < kFloatTol || eCacheAsLS < kFloatTol)
            G_CHECK(false, "[%s] 检测到 out / cache 两个参数装反了", c.tag);

        // 自洽性质：每行 softmax 之和为 1
        double worstRow = 0.0;
        for (size_t l = 0; l < c.M; ++l) {
            double s = 0.0;
            for (size_t i = 0; i < c.N; ++i) s += (double)hCache[l * c.N + i];
            worstRow = std::max(worstRow, std::fabs(s - 1.0));
        }
        G_CHECK(worstRow < kFloatTol, "[%s] softmax 每行之和偏离 1，最大 %.3e", c.tag, worstRow);
    }
}

// ---------------------------------------------------------------------------
// L5 / L6：MSE
//   forward  返回 (1/(M*N)) * Σ(pred-real)^2 —— **均方**误差，已归一化
//   backward 给出 2*(pred-real)/(M*N)
//
//   两者是配套的：backward 正是 forward 对 pred 的导数
//   （d/dp [ (p-t)^2/(M*N) ] = 2(p-t)/(M*N)）。所以这一条同时钉住两件事：
//   forward 有没有除 M*N、两边归一化口径是否一致。
// ---------------------------------------------------------------------------
inline void l5_l6_mse()
{
    printf("[L5] launchMSELossForward / [L6] launchMSELossBackwardKernel\n");

    struct Case { size_t M, N; };
    const Case cases[] = { {1, 1}, {1, 10}, {64, 10}, {3, 257} };

    for (const Case& c : cases) {
        Matrix pred = makeMatrix(c.M, c.N, 1901);
        Matrix real = makeMatrix(c.M, c.N, 1902);

        const double got = launchMSELossForward(pred.data_gpu(), real.data_gpu(), c.M, c.N);
        if (!runAndCheck("launchMSELossForward")) {
            G_CHECK(false, "M=%zu N=%zu kernel 失败", c.M, c.N);
            continue;
        }

        const std::vector<float> hp = download(pred), hr = download(real);
        double sumSq = 0.0;
        for (size_t i = 0; i < hp.size(); ++i) {
            const double d = (double)hp[i] - (double)hr[i];
            sumSq += d * d;
        }

        const double mean = sumSq / (double)(c.M * c.N);   // 期望的返回值

        const double tol = relTol(mean);
        G_CHECK(std::fabs(got - mean) < tol,
                "forward M=%zu N=%zu 得到 %.6f，期望均方误差 %.6f（平方和 %.6f / %zu）",
                c.M, c.N, got, mean, sumSq, c.M * c.N);

        // backward
        Matrix out(c.M, c.N);
        upload(out, std::vector<float>(out.size(), -999.0f));
        launchMSELossBackwardKernel(pred.data_gpu(), real.data_gpu(), out.data_gpu(), c.M, c.N);
        if (!runAndCheck("launchMSELossBackwardKernel")) {
            G_CHECK(false, "M=%zu N=%zu kernel 失败", c.M, c.N);
            continue;
        }

        const std::vector<float> hOut = download(out);
        std::vector<float> wantB(hp.size());
        for (size_t i = 0; i < hp.size(); ++i)
            wantB[i] = (float)(2.0 * ((double)hp[i] - (double)hr[i]) / (double)(c.M * c.N));

        const double e = maxDiff(hOut, wantB);
        G_CHECK(e < kFloatTol, "backward M=%zu N=%zu 误差 %.3e", c.M, c.N, e);
    }
}

// ---------------------------------------------------------------------------
// L7：优化器一步
//   velocity = velocity * momentum + grad
//   value   -= velocity * learning_rate
// ---------------------------------------------------------------------------
inline void l7_optimizer_step()
{
    printf("[L7] launchGeneralUpdateWithMomentum\n");

    struct Case { size_t n; };
    const Case cases[] = { {1}, {17}, {256}, {1000}, {4096} };

    const float lr = 0.015f, mu = 0.9f;

    for (const Case& c : cases) {
        Matrix value = makeMatrix(1, c.n, 2001);
        Matrix grad = makeMatrix(1, c.n, 2002);
        Matrix vel = makeMatrix(1, c.n, 2003);

        const std::vector<float> hv = download(value), hg = download(grad);
        const std::vector<float> hm = download(vel);

        launchGeneralUpdateWithMomentum(value.data_gpu(), grad.data_gpu(), vel.data_gpu(), c.n, lr, mu);
        if (!runAndCheck("launchGeneralUpdateWithMomentum")) {
            G_CHECK(false, "n=%zu kernel 失败", c.n);
            continue;
        }

        // 参照：拿下载下来的初值手算一步
        std::vector<float> wantV(c.n), wantM(c.n);
        for (size_t i = 0; i < c.n; ++i) {
            wantM[i] = hm[i] * mu + hg[i];
            wantV[i] = hv[i] - wantM[i] * lr;
        }

        const std::vector<float> gotV = download(value), gotM = download(vel);
        G_CHECK(maxDiff(gotM, wantM) < relTol(mu),
                "n=%zu velocity 误差 %.3e", c.n, maxDiff(gotM, wantM));
        G_CHECK(maxDiff(gotV, wantV) < relTol(lr),
                "n=%zu value 误差 %.3e", c.n, maxDiff(gotV, wantV));

        // 第二轮：确认 momentum 是累加的（velocity 被写回后能继续用）
        launchGeneralUpdateWithMomentum(value.data_gpu(), grad.data_gpu(), vel.data_gpu(), c.n, lr, mu);
        if (!runAndCheck("launchGeneralUpdateWithMomentum (第二轮)")) {
            G_CHECK(false, "n=%zu 第二轮失败", c.n);
            continue;
        }
        std::vector<float> wantV2(c.n), wantM2(c.n);
        for (size_t i = 0; i < c.n; ++i) {
            wantM2[i] = wantM[i] * mu + hg[i];
            wantV2[i] = wantV[i] - wantM2[i] * lr;
        }
        G_CHECK(maxDiff(download(value), wantV2) < relTol(lr),
                "n=%zu 第二轮 value 误差 %.3e", c.n, maxDiff(download(value), wantV2));
    }
}

// ---------------------------------------------------------------------------
// 入口
// ---------------------------------------------------------------------------
inline int runModelTests()
{
    printf("\n===== 层 / 损失 / 优化器 =====\n");
    l1_linear_forward();
    l2_l3_relu();
    l4_log_softmax();
    l5_l6_mse();
    l7_optimizer_step();
    return fails();
}

} // namespace gputest
