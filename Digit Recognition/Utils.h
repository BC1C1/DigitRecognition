#pragma once

#include "QRandomGenerator"
#include <cmath>

#include "SuperParam.h"

inline double randn() {
	static thread_local QRandomGenerator gen(cfg::magicSeed);
	double u1 = gen.generateDouble();
	double u2 = gen.generateDouble();
	if (u1 < 1e-10) u1 = 1e-10;
	return std::sqrt(-2.0 * std::log(u1)) * std::cos(2.0 * cfg::M_PI * u2);
}