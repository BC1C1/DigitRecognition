#pragma once

#include <memory>
#include <qjsonobject.h>
#include <qjsonarray.h>
#include <array>

#include "Layer.h"
#include "Linear.h"
#include "Conv2D.h"
#include "MaxPool2D.h"
#include "ReLU.h"
#include "LossFunction.h"
#include "MSELoss.h"
#include "CrossEntropyLoss.h"
#include "SGD.h"
#include "Optimizer.h"
#include "SuperParam.h"

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

	void setLearningRate(double newLearningRate) { op->setLearningRate(newLearningRate); }
	double getLearningRate() const { return op->getLR(); }

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

public:
	static std::shared_ptr<Model> loadFromJson(QJsonObject obj) {
		auto model = std::make_shared<Model>();
		// loss
		QString lossType = obj["loss"].toString();
		auto lossObj = lossFromQString(lossType);
		assert(lossObj);
		model->setLossFunction(lossObj);
		// opt
		QString opt = obj["optimizer"].toString();
		auto optObj = optimizerFromQString(opt);
		assert(optObj);
		model->setOptimizer(optObj);
		// layers
		QJsonArray ls = obj["layers"].toArray();
		for (const auto& l : ls) {
			QString layerType = l.toObject()["type"].toString();
			QJsonArray layerParams = l.toObject()["params"].toArray();
			auto layerobj = layerFromNameWithParams(layerType, layerParams);
			assert(layerobj);
			model->addLayer(layerobj);
		}
		return model;
	}

	// tojson 可以先留空，后续需要了再补, 现在加载可以硬编码在SuperParam.h里

private:
	using LayerPointer = std::shared_ptr<Layer>;

	static std::shared_ptr<LossFunction> lossFromQString(QString lossName) {
		if (lossName == "MSE") return std::make_shared<MSELoss>();
		if (lossName == "CrossEntropy") return std::make_shared<CrossEntropyLoss>();
		return nullptr;
	}

	static std::shared_ptr<SGD> optimizerFromQString(QString optName) {
		if (optName == "SGD") return std::make_shared<Optimizer>();
		return nullptr;
	}

	static LayerPointer layerFromNameWithParams(QString layerName, QJsonArray params) {
		static size_t curH = cfg::H;
		static size_t curW = cfg::W;
		if (layerName == "linear") {
			assert(params.size() == 2);
			int in = params[0].toInt();
			int out = params[1].toInt();
			return linear(in, out);
		}
		if (layerName == "relu") {
			assert(params.size() == 0);
			return relu();
		}
		if (layerName == "conv2d") {
			assert(params.size() == 5);
			size_t inputdim = (size_t)(params[0].toObject()["inputdim"].toInt());
			size_t outputdim = (size_t)(params[1].toObject()["outputdim"].toInt());
			size_t ksize = (size_t)(params[2].toObject()["ksize"].toInt());
			size_t stride = (size_t)(params[3].toObject()["stride"].toInt());
			size_t padding = (size_t)(params[4].toObject()["padding"].toInt());
			auto ret = conv2d(curH, curW, inputdim, outputdim, ksize, stride, padding);
			curH = (curH + 2 * padding - ksize) / stride + 1;
			curW = (curW + 2 * padding - ksize) / stride + 1;
			return ret;
		}
		if (layerName == "maxpool2d") {
			assert(params.size() == 3);
			size_t channel = (size_t)(params[0].toObject()["channel"].toInt());
			size_t ksize = (size_t)(params[1].toObject()["ksize"].toInt());
			size_t stride = (size_t)(params[2].toObject()["stride"].toInt());
			auto ret = maxpool2d(curH, curW, channel, ksize, stride);
			curH = (curH - ksize) / stride + 1;
			curW = (curW - ksize) / stride + 1;
			return ret;
		}
		return nullptr;
	}

	static LayerPointer linear(size_t in, size_t out) {
		return std::make_shared<Linear>(in, out);
	}

	static LayerPointer relu() {
		return std::make_shared<ReLU>();
	}

	static LayerPointer conv2d(size_t curH, size_t curW, size_t inputdim, size_t outputdim,
		size_t ksize, size_t stride, size_t padding) {
		return std::make_shared<Conv2D>(curH, curW, inputdim, outputdim, ksize, stride, padding);
	}

	static LayerPointer maxpool2d(size_t curH, size_t curW, size_t channel, size_t ksize, size_t stride) {
		return std::make_shared<MaxPool2D>(curH, curW, channel, ksize, stride);
	}

private:
	std::vector<std::shared_ptr<Layer>> layers;
	std::shared_ptr<SGD> op;
	std::shared_ptr<LossFunction> lossFunction;

};

