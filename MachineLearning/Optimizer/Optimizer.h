#pragma once

#include <vector>
#include <memory>

#include "SuperParam.h"

#include "SGD.h"
#include "Layer.h"

class Optimizer : public SGD
{
public:
	Optimizer() : learning_rate(cfg::learning_rate), mu(cfg::momentum) {}

	virtual void step() override;

	virtual void addLayer(std::shared_ptr<Layer> layer) override;

	virtual void setLearningRate(float newLearningRate) override { learning_rate = newLearningRate; }

	virtual float getLR() const override { return learning_rate; }
private:
	double learning_rate;
	float mu;
	std::vector<std::shared_ptr<Layer>> layers;
};

