#pragma once

#include "Layer.h"

class MaxPool2D : public Layer
{
public:
    MaxPool2D(size_t H, size_t W, size_t channel, size_t ksize, size_t stride) :
        H(H), W(W), channel(channel), ksize(ksize), stride(stride), maxIndex(nullptr)
    {
        OH = (H - ksize) / stride + 1;
        OW = (W - ksize) / stride + 1;
    }
    ~MaxPool2D() {
        if (maxIndex) {
            cudaFreeAsync_throw(maxIndex); maxIndex = nullptr;
        }
    }

    virtual Matrix forward(const Matrix& x) override {
        reshape(x.rows());
        launchForward(x);
        return maxCache;
    }

    virtual Matrix backward(const Matrix& grad) override {
        launchBackward(grad);
        return dXCache;
    }

    virtual std::vector<ParamPtr> getParams() { return {}; }
private:
    void launchForward(const Matrix& x) {
        launchMaxPool2DForward(x.data_gpu(), maxCache.data_gpu(), x.rows(), channel, H, W, ksize, stride, maxIndex);
    }
    void launchBackward(const Matrix& grad) {
        launchFillInplace(dXCache.data_gpu(), dXCache.size(), 0.0);
        launchMaxPool2DBackward(grad.data_gpu(), dXCache.data_gpu(), grad.rows(), channel, H, W, ksize, stride, maxIndex);
    }
    void reshape(size_t N) {
        if (maxCache.rows() != N) {
            maxCache = Matrix(N, channel * OH * OW);
            dXCache = Matrix(N, channel * H * W);
            if (maxIndex) {
                cudaFreeAsync_throw(maxIndex); maxIndex = nullptr;
            }
            cudaMallocAsync_throw(&maxIndex, sizeof(size_t) * N * channel * OH * OW);
        }
    }
private:
    Matrix maxCache;
    Matrix dXCache;
    size_t* maxIndex;
    size_t H, W, channel, ksize, stride;
    size_t OH, OW;
};
// nullptr 是说此时这个指针无效，现在cpu压力并不大，所以做一些额外操作来满足我掌控语义的想法是合理的