#pragma once
#include <iostream>

#include <memory>
#include <qjsonobject.h>
#include <qjsonarray.h>
#include <qhash.h>
#include <qvector.h>
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
		QString lossType = obj[cfg::lossNameString].toString();
		auto lossObj = lossFromQString(lossType);
		assert(lossObj);
		model->setLossFunction(lossObj);
		// opt
		QString opt = obj[cfg::optimizerNameString].toString();
		auto optObj = optimizerFromQString(opt);
		assert(optObj);
		model->setOptimizer(optObj);
		// layers
		QJsonArray ls = obj[cfg::layersNameString].toArray();
		QVector<size_t> dim_list;
		for (const auto& l : ls) {
			QString layerType = l.toObject()[cfg::typeNameString].toString();
			QJsonObject layerParams = l.toObject()[cfg::paramsNameString].toObject();
			auto layerobj = layerFromNameWithParams(layerType, layerParams, dim_list);
			assert(layerobj);
			model->addLayer(layerobj);
		}
        std::cout << "model is ready!" << std::endl;
		return model;
	}

	// tojson 可以先留空，后续需要了再补, 现在加载可以硬编码在SuperParam.h里

private:
	using LayerPointer = std::shared_ptr<Layer>;

	static std::shared_ptr<LossFunction> lossFromQString(QString lossName) {
		if (lossName == cfg::MSELossNameString) return std::make_shared<MSELoss>();
		if (lossName == cfg::crossEntropyLossNameString) return std::make_shared<CrossEntropyLoss>();
		return nullptr;
	}

	static std::shared_ptr<SGD> optimizerFromQString(QString optName) {
		if (optName == cfg::AdamOptimizerNameString) return std::make_shared<Optimizer>();
		return nullptr;
	}

    static LayerPointer layerFromNameWithParams(QString layerName, QJsonObject params,
        QVector<size_t>& dimlist) {
        const size_t curRank = dimlist.size();

        // ---------- linear ----------
        if (layerName == cfg::linearNameString) {
            const size_t layerRank = cfg::layerDimMap.value(layerName);  // 1
            size_t outputDim = params[cfg::outputDimNameString].toInt();
            size_t inputDim;

            if (curRank < layerRank) {
                inputDim = params[cfg::inputDimNameString].toInt();
            }
            else {
                size_t newDim = 1;
                for (auto d : dimlist) newDim *= d;
                inputDim = newDim;
                if (params.contains(cfg::inputDimNameString)) {
                    size_t checkSize = params[cfg::inputDimNameString].toInt();
                    if (checkSize != newDim) {
                        std::cout << "[linear] inputdim mismatch: json=" << checkSize
                            << ", derived=" << newDim
                            << ", using derived value" << std::endl;
                    }
                }
            }

            dimlist.clear();
            dimlist.push_back(outputDim); 
            return linear(inputDim, outputDim);
        }

        // ---------- relu ----------
        if (layerName == cfg::reluNameString) {
            assert(params.isEmpty());
            assert(curRank > 0);
            return relu();
        }

        // ---------- conv2d ----------
        if (layerName == cfg::conv2dNameString) {
            const size_t layerRank = cfg::layerDimMap.value(layerName);  // 3
            size_t outputdim = params[cfg::outputDimNameString].toInt();
            size_t ksize = params[cfg::kernelsizeNameString].toInt();
            size_t stride = params[cfg::strideNameString].toInt();
            size_t padding = params[cfg::paddingNameString].toInt();

            size_t inputdim, inputH, inputW;
            if (curRank < layerRank) {
                inputdim = params[cfg::inputDimNameString].toInt();
                inputH = params[cfg::inputHNameString].toInt();
                inputW = params[cfg::inputWNameString].toInt();
            }
            else {
                inputdim = dimlist[0];
                inputH = dimlist[1];
                inputW = dimlist[2];

                // 可选校验
                if (params.contains(cfg::inputDimNameString)) {
                    size_t check = params[cfg::inputDimNameString].toInt();
                    if (check != inputdim) {
                        std::cout << "[conv2d] inputdim mismatch: json=" << check
                            << ", derived=" << inputdim
                            << ", using derived value" << std::endl;
                    }
                }
            }

            dimlist.clear();
            dimlist.push_back(outputdim);
            dimlist.push_back((inputH + 2 * padding - ksize) / stride + 1);
            dimlist.push_back((inputW + 2 * padding - ksize) / stride + 1);
            //std::cout << "conv2D inputH: " << inputH << std::endl;
            //std::cout << "conv2D inputW: " << inputW << std::endl;
            //std::cout << "conv2D inputdim: " << inputdim << std::endl;
            return conv2d(inputH, inputW, inputdim, outputdim, ksize, stride, padding);
        }

        // ---------- maxpool2d ----------
        if (layerName == cfg::maxpool2dNameString) {
            const size_t layerRank = cfg::layerDimMap.value(layerName);  // 3
            size_t ksize = params[cfg::kernelsizeNameString].toInt();
            size_t stride = params[cfg::strideNameString].toInt();

            size_t channel, inputH, inputW;
            if (curRank < layerRank) {
                channel = params[cfg::channelNameString].toInt();
                inputH = params[cfg::inputHNameString].toInt();
                inputW = params[cfg::inputWNameString].toInt();
            }
            else {
                channel = dimlist[0];
                inputH = dimlist[1];
                inputW = dimlist[2];

                if (params.contains(cfg::channelNameString)) {
                    size_t check = params[cfg::channelNameString].toInt();
                    if (check != channel) {
                        std::cout << "[maxpool2d] channel mismatch: json=" << check
                            << ", derived=" << channel
                            << ", using derived value" << std::endl;
                    }
                }
            }

            dimlist.clear();
            dimlist.push_back(channel);
            dimlist.push_back((inputH - ksize) / stride + 1);
            dimlist.push_back((inputW - ksize) / stride + 1);

            return maxpool2d(inputH, inputW, channel, ksize, stride);
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

