#pragma once

#include "LossFunction.h"

class MSELoss : public LossFunction
{
public:
	virtual double forward(const Matrix& pred, const Matrix& real) override;
	virtual Matrix backward() override;

private:
	Matrix pred_cache;
	Matrix real_cache;
};

