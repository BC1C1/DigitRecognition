#pragma once

#include "Layer.h"

class Linear : public Layer
{
public:
	Linear(size_t input_dim, size_t output_dim);

	virtual Matrix forward(const Matrix& x) override;
	virtual Matrix backward(const Matrix& grad) override;
	//virtual void update(double learning_rate) override;
	virtual std::vector<ParamPtr> getParams() {
		return std::vector<ParamPtr>{ { &W, &dw, &vW }, { &b, &db, &vb } };
	}

private:
	Matrix W;
	//Matrix W_T;
	Matrix b;
	Matrix cache;
	Matrix forwardPartCache;    // forward 的输出缓冲（复用，避免每批分配）
	Matrix backwardRetCache;    // backward 的返回值缓冲（复用）

	Matrix dw;
	Matrix db;

	Matrix vW;
	Matrix vb;

	//Matrix gard_T_cache;
};

