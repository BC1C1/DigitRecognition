#pragma once

#include "Layer.h"

class MaxPool2D : public Layer
{
public:
	MaxPool2D(size_t H, size_t W, size_t channel, size_t ksize, size_t stride) :
		H(H), W(W), channel(channel), ksize(ksize), stride(stride) {
		OH = (H - ksize) / stride + 1;
		OW = (W - ksize) / stride + 1;
	}

	virtual Matrix forward(const Matrix& x) override {
		// x is (N, (C H W)), so
		size_t N = x.rows();
		assert(channel * H * W == x.cols());
		if (maxCache.rows() != N)
			maxCache = Matrix(N, channel * OH * OW);
		maxIndex.clear();
		for (size_t n = 0; n < N; n++) {
			for (size_t c = 0; c < channel; c++) {
				size_t cHW = c * H * W;
				size_t cOHOW = c * OH * OW;
				for (size_t oh = 0; oh < OH; oh++) {
					for (size_t ow = 0; ow < OW; ow++) {
						// find the range
						size_t head_x = ow * stride;
						size_t head_y = oh * stride;
						double m_value = x(n, cHW + head_y * W + head_x);
						size_t idx = cHW + head_y * W + head_x;
						for (size_t h = 0; h < ksize; h++) {
							for (size_t w = 0; w < ksize; w++) {
								size_t cIdx = cHW + (head_y + h) * W + head_x + w;
								double cur = x(n, cIdx);
								if (cur > m_value) {
									m_value = cur;
									idx = cIdx;
								}
							}
						}
						maxCache(n, cOHOW + oh * OW + ow) = m_value;
						maxIndex.push_back(idx);
					}
				}
			}
		}
		return maxCache;
	}
	virtual Matrix backward(const Matrix& grad) override {
		// grad is (N, (C OH OW))
		size_t N = grad.rows();
		assert(channel * H * W == grad.cols());
		Matrix ret(N, channel * H * W);
		size_t k = 0; 
		for (size_t n = 0; n < N; n++) {
			for (size_t c = 0; c < channel; c++) {
				size_t cOHOW = c * OH * OW;
				for (size_t oh = 0; oh < OH; oh++) {
					size_t ohOW = oh * OW;
					for (size_t ow = 0; ow < OW; ow++) {
						double value = grad(n, cOHOW + ohOW + ow);
						ret(n, maxIndex[k]) += value;
						k++;
					}
				}
			}
		}
		return ret;
	}
	virtual void update(double learning_rate) override {
	}
private:
	Matrix maxCache;
	std::vector<size_t> maxIndex;
	size_t H, W, channel, ksize, stride;
	size_t OH, OW;
};

