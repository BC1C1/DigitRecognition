#pragma once

#include "TypeDefine.h"
#include <vector>
#include "cuda_bridge.h"

struct ParamPtr {
	Matrix* w;
	Matrix* g;
	Matrix* m;
	Matrix* v;
};

class Layer
{
public:
	virtual Matrix forward(const Matrix& x) = 0;
	virtual Matrix backward(const Matrix& grad) = 0;
	//virtual void update(double learning_rate) = 0;
	virtual std::vector<ParamPtr> getParams() = 0;
};

