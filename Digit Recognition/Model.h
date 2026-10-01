#pragma once

#include <memory>

#include "Layer.h"
#include "LossFunction.h"
#include "SGD.h"


class Model
{
public:
	Model() : op(nullptr), lossFunction(nullptr) {}
	void setOptimizer(std::shared_ptr<SGD> o) { this->op = o; }
	void setLossFunction(std::shared_ptr<LossFunction> loss) { this->lossFunction = loss; }
	void addLayer(std::shared_ptr<Layer> layer) { 
		assert(op != nullptr && "optimizer cannot be nullptr");
		this->layers.push_back(layer);
		op->addLayer(layer); 
	}

	Matrix forward(const Matrix& input);
	void backward(const Matrix& gard);
	Matrix predict(const Matrix& input);

	double train_step(const Matrix& input, const Matrix& real);

	// for test
	Matrix net_forward(const Matrix& input);
	double loss_forward(const Matrix& net_out, const Matrix& real);
	Matrix loss_backward();
	void net_backward(const Matrix& gard);
	void opt_step();

private:
	std::vector<std::shared_ptr<Layer>> layers;
	std::shared_ptr<SGD> op;
	std::shared_ptr<LossFunction> lossFunction;

};

