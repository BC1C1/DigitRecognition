#pragma once

#include "Layer.h"

class Linear : public Layer
{
public:
	Linear(size_t input_dim, size_t output_dim);

	virtual Matrix forward(const Matrix& x) override;
	virtual Matrix backward(const Matrix& grad) override;
	virtual void update(double learning_rate) override;

private:
	Matrix W;
	//Matrix W_T;
	Matrix b;
	Matrix cache;

	Matrix dw;
	Matrix db;

	//Matrix gard_T_cache;
};

