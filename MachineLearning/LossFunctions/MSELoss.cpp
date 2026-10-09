#include "MSELoss.h"

float MSELoss::forward(const Matrix& pred, const Matrix& real)
{
    pred_cache = pred;
    real_cache = real;
    auto M = pred.rows();
    auto N = pred.cols();
    float sum = 0;
    for (size_t i = 0; i < M; i++) {
        for (size_t j = 0; j < N; j++) {
            auto d = pred(i, j) - real(i, j);
            sum += d * d;
        }
    }
    return sum / (pred.rows() * pred.cols());
}

Matrix MSELoss::backward()
{
    size_t M = pred_cache.rows();
    size_t N = pred_cache.cols();
    auto temp = (pred_cache - real_cache);
    temp = temp.foreach_do([N, M](float x)->float { return 2 * x / (M * N); });
    return temp;
}
