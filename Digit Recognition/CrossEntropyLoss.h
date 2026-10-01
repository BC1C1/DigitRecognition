#pragma once

#include "LossFunction.h"
#include "Utils.h"
#include "vector"

// with softmax
class CrossEntropyLoss : public LossFunction
{
public:
	virtual double forward(const Matrix& pred, const Matrix& real) override;
	virtual Matrix backward() override;

	static std::pair<Matrix, Matrix> matrixLogSoftmaxWithCache(const Matrix& inputs) {
		size_t N = inputs.cols();
		if (N == 0) return { Matrix(), Matrix() };
		size_t M = inputs.rows();
		Matrix ret1(M, N);
		Matrix ret2(M, N);
		for (size_t l = 0; l < inputs.rows(); l++) {
			// want log(e^(i - M) / sum(e^(j - M)) = i - M - log(sum(e^(j - M)))
			// want e^(i - M) / sum(e^(j - M))
			// M
			double maxVal = inputs(l, 0);
			for (size_t i = 0; i < N; i++) {
				maxVal = std::max(maxVal, inputs(l, i));
			}
			// sum(e^(j - M))
			double e_s = 0;
			for (size_t i = 0; i < N; i++) {
				e_s += std::exp(inputs(l, i) - maxVal);
			}
			auto log_e_s = std::log(e_s);
			for (size_t i = 0; i < N; i++) {
				ret1(l, i) = (inputs(l, i) - maxVal - log_e_s);
				ret2(l, i) = std::exp(inputs(l, i) - maxVal) / e_s;
			}
		}
		return { ret1, ret2 };
	}
private:
	Matrix softmaxCache;
	Matrix oneHotCache;
};

