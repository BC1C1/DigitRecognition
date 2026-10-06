#include "Optimizer.h"

void Optimizer::step()
{
	//for (const auto& l : layers) {
	//	l->update(learning_rate);
	//}
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
			// velocity = velocity * mu + grad;
			// value = value - velocity * learning_rate;
			//for (size_t i = 0; i < velocity.rows(); i++) {
			//	for (size_t j = 0; j < velocity.cols(); j++) {
			//		velocity(i, j) = velocity(i, j) * mu + grad(i, j);
			//		value(i, j) -= velocity(i, j) * learning_rate;
			//	}
			//}
			// faster
			auto vptr = value.raw_data();
			auto gptr = grad.raw_data();
			auto mptr = velocity.raw_data();
			for (size_t i = 0; i < velocity.size(); i++) {
				mptr[i] = mptr[i] * mu + gptr[i];
				vptr[i] -= mptr[i] * learning_rate;
			}
		}
	}
}

void Optimizer::addLayer(std::shared_ptr<Layer> layer)
{
	layers.push_back(layer);
}
