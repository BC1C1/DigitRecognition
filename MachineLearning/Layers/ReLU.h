#pragma once

#include "Layer.h"

class ReLU : public Layer
{
public:
	virtual Matrix forward(const Matrix& x) override {
		reshape(x.rows(), x.cols());
		launchForward(x);
		return forwardCache;
	}
	virtual Matrix backward(const Matrix& gard) override {
		launchBackward(gard);
		return backwardCache;
	}
	virtual std::vector<ParamPtr> getParams() { return {}; }
private:
	void launchForward(const Matrix& x) {
		launchReLUForward(x.data_gpu(), forwardCache.data_gpu(), x.rows(), x.cols());
	}
	void launchBackward(const Matrix& grad) {
		launchReLUBackward(grad.data_gpu(), forwardCache.data_gpu(), backwardCache.data_gpu(),
			grad.rows(), grad.cols());
	}
	void reshape(size_t N, size_t D) {
		if (forwardCache.rows() != N || forwardCache.cols() != D) {
			forwardCache = Matrix(N, D);
			backwardCache = Matrix(N, D);
		}
	}
private:
	Matrix forwardCache;
	Matrix backwardCache;
};

