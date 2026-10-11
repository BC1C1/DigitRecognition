#include "CrossEntropyLoss.h"

double CrossEntropyLoss::forward(const Matrix& pred, const Matrix& real)
{
    // pred is value
    size_t M = pred.rows(); // M point
    size_t N = pred.cols(); // N class
    reshape(M, N);
    oneHotCache = real; // 不需要reshape
    launchLogSoftmax(pred.data_gpu(), softmaxCache.data_gpu(), logsoftmaxCache.data_gpu(), M, N);
    launchElementwiseMulInPlace(logsoftmaxCache.data_gpu(), real.data_gpu(), real.size());
    double sum = launchSumKernel(logsoftmaxCache.data_gpu(), logsoftmaxCache.size()); 
    return -sum / M;
}

Matrix CrossEntropyLoss::backward()
{
    size_t rows = softmaxCache.rows();
    launchSubInPlace(softmaxCache.data_gpu(), oneHotCache.data_gpu(), softmaxCache.size());
    launchScaleMulInPlace(softmaxCache.data_gpu(), softmaxCache.size(), 1.0 / rows);
    return softmaxCache;
}

void CrossEntropyLoss::reshape(size_t M, size_t N)
{
    if (logsoftmaxCache.rows() != M || logsoftmaxCache.cols() != N) {
        logsoftmaxCache = Matrix(M, N);
        softmaxCache = Matrix(M, N);
   }
}
