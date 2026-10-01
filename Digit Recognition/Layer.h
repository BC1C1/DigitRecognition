#pragma once

#include "TypeDefine.h"

class Layer
{
public:
	virtual Matrix forward(const Matrix& x) = 0;
	virtual Matrix backward(const Matrix& grad) = 0;
	virtual void update(double learning_rate) = 0;
};

