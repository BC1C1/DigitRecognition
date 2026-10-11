#pragma once

#include "Layer.h"
#include "cuda_bridge.h"

class LossFunction
{
public:
	virtual double forward(const Matrix& pred, const Matrix& real) = 0;
	virtual Matrix backward() = 0;
};

