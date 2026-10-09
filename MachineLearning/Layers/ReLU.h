#pragma once

#include "Layer.h"

class ReLU : public Layer
{
public:
	virtual Matrix forward(const Matrix& x) override {
		if (xCache.rows() != x.rows() || xCache.cols() != x.cols())
			xCache = Matrix(x.rows(), x.cols());
		for (size_t i = 0; i < x.rows(); i++) {
			for (size_t j = 0; j < x.cols(); j++) {
				float value = x(i, j);
				xCache(i, j) = value > 0 ? value : 0;
			}
		}
		return xCache;
	}
	virtual Matrix backward(const Matrix& gard) override {
		// xCache 中为0的认为是前面被掩盖了
		Matrix ret(gard.rows(), gard.cols());
		for (size_t i = 0; i < xCache.rows(); i++) {
			for (size_t j = 0; j < xCache.cols(); j++) {
				ret(i, j) = xCache(i, j) == 0 ? 0 : gard(i, j);
			}
		}
		return ret;
	}
	//virtual void update(double learning_rate) override {

	//}
	virtual std::vector<ParamPtr> getParams() { return {}; }
private:
	Matrix xCache;
};

