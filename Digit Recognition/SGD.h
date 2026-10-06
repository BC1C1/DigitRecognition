#pragma once
#include <memory>
#include "Layer.h"
class SGD
{
public:
 	virtual void step() = 0;
    virtual void addLayer(std::shared_ptr<Layer> layer) = 0;
    virtual void setLearningRate(double learningrate) = 0;
    virtual double getLR() const = 0;
};

