#include "Optimizer.h"

void Optimizer::step()
{
	t++;
	float inv_bc1 = 1.0f / (1.0f - powf(b1, (float)t));
	float inv_bc2 = 1.0f / (1.0f - powf(b2, (float)t));
	for (const auto& l : layers) {
		auto params = l->getParams();
		// 通用
		for (const auto& ps : params) {
			auto& w = *ps.w;
			auto& g = *ps.g;
			auto& m = *ps.m;
			auto& v = *ps.v;
			assert(w.rows() == g.rows());
			assert(w.rows() == m.rows());
			assert(w.rows() == v.rows());
			assert(w.cols() == g.cols());
			assert(w.cols() == m.cols());
			assert(w.cols() == v.cols());
			launchAdamUpdateKernel(w.data_gpu(), g.data_gpu(), m.data_gpu(), v.data_gpu(), w.size(),
				learning_rate, b1, b2, epsilon, t, inv_bc1, inv_bc2);
		}
	}
}

void Optimizer::addLayer(std::shared_ptr<Layer> layer)
{
	layers.push_back(layer);
}
