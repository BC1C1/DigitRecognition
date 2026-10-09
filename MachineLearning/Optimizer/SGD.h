#pragma once
#include <memory>
#include "Layer.h"
class SGD
{
public:
 	virtual void step() = 0;
    virtual void addLayer(std::shared_ptr<Layer> layer) = 0;
    virtual void setLearningRate(float learningrate) = 0;
    virtual float getLR() const = 0;
};

