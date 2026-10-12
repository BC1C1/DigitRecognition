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
		return std::vector<ParamPtr>{ { &W, &dw, &mW, &vW }, { &b, &db, &mb, &vb } };
	}
private:
	void launchForward(const Matrix& x);
	void launchBackward(const Matrix& grad);
	void reshape(size_t N);

private:
	Matrix W;
	Matrix b;
	Matrix cache;				// x
	Matrix forwardCache;		// forward 
	Matrix backwardRetCache;    // backward 

	Matrix dw;
	Matrix db;

	Matrix mW;
	Matrix mb;
	Matrix vW;
	Matrix vb;

	//Matrix gard_T_cache;
	size_t I, O;
};

