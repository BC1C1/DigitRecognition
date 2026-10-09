#pragma once

#include "Layer.h"

class LossFunction
{
public:
	virtual float forward(const Matrix& pred, const Matrix& real) = 0;
	virtual Matrix backward() = 0;
};

