#include "CrossEntropyLoss.h"

float CrossEntropyLoss::forward(const Matrix& pred, const Matrix& real)
{
    // pred is value
    size_t M = pred.rows(); // M point
    size_t N = pred.cols(); // N class
    float loss = 0;
    auto p = matrixLogSoftmaxWithCache(pred);
    softmaxCache = p.second;
    oneHotCache = real;
    auto temp = p.first.elementwiseMul(real);
    for (size_t i = 0; i < M; i++) {
        for (size_t j = 0; j < N; j++) {
            loss += temp(i, j);
        }
    }
    return -loss / M;
}

Matrix CrossEntropyLoss::backward()
{
    auto d = softmaxCache - oneHotCache;
    size_t rows = softmaxCache.rows();
    d = d.foreach_do([rows](float x) { return x / rows; });
    return d;
}
