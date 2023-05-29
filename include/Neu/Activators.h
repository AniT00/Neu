#pragma once
#include <cmath>

namespace Activators {
	using Activator = float(*)(float);
	//typedef float((*Activator))(float);

	inline static float sigmoid(float x);

	inline static float sigmoidDer(float x);

	inline static float tanhDer(float x);
}