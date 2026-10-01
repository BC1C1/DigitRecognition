#include "Optimizer.h"

void Optimizer::step()
{
	for (const auto& l : layers) {
		l->update(learning_rate);
	}
}

void Optimizer::addLayer(std::shared_ptr<Layer> layer)
{
	layers.push_back(layer);
}
