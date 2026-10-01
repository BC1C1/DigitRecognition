#pragma once

#include <vector>
#include <memory>

#include "SuperParam.h"

#include "SGD.h"
#include "Layer.h"

class Optimizer : public SGD
{
public:
	Optimizer() : learning_rate(::learning_rate) {}

	virtual void step() override;

	virtual void addLayer(std::shared_ptr<Layer> layer) override;

private:
	double learning_rate;
	std::vector<std::shared_ptr<Layer>> layers;
};

