#pragma once

#include "Layer.h"

class ReLU : public Layer
{
public:
	virtual Matrix forward(const Matrix& x) override {
		xCache = x.foreach_do([](double x) -> double {
			return x > 0 ? x : 0;
		});
		return xCache;
	}
	virtual Matrix backward(const Matrix& gard) override {
		// xCache 中为0的认为是前面被掩盖了
		auto f = [](double x) { return x == 0 ? 0 : 1; };
		return Matrix::elementwiseMul(xCache.foreach_do(f), gard);
	}
	virtual void update(double learning_rate) {

	}
private:
	Matrix xCache;
};

