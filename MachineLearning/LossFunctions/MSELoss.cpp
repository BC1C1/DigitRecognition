#include "MSELoss.h"

double MSELoss::forward(const Matrix& pred, const Matrix& real)
{
    auto M = pred.rows();
    auto N = pred.cols();
    assert(real.rows() == M);
    assert(real.cols() == N);
    pred_cache = pred; 
    real_cache = real; // 因为backward对它也是只读，所以没关系
    reshape(M, N);
    return launchMSELossForward(pred.data_gpu(), real.data_gpu(), M, N);
}

Matrix MSELoss::backward()
{
    size_t M = pred_cache.rows();
    size_t N = pred_cache.cols();
    launchMSELossBackwardKernel(pred_cache.data_gpu(), real_cache.data_gpu(), backwardCache.data_gpu(), M, N);
    return backwardCache;
}

void MSELoss::reshape(size_t M, size_t N)
{
    // 其实N不会变
    if (backwardCache.rows() != M || backwardCache.cols() != N) {
        backwardCache = Matrix(M, N);
    }
}
