#include "Optimizer.h"

void Optimizer::step()
{
	for (const auto& l : layers) {
		auto params = l->getParams();
		// 通用
		for (const auto& ps : params) {
			auto& value = *ps.value;
			auto& grad = *ps.grad;
			auto& velocity = *ps.momentum;
			assert(value.rows() == grad.rows());
			assert(velocity.rows() == grad.rows());
			assert(value.cols() == grad.cols());
			assert(velocity.cols() == grad.cols());
			// faster
			launchGeneralUpdateWithMomentum(value.data_gpu(), grad.data_gpu(), velocity.data_gpu(),
				value.size(), learning_rate, mu);
		}
	}
}

void Optimizer::addLayer(std::shared_ptr<Layer> layer)
{
	layers.push_back(layer);
}
