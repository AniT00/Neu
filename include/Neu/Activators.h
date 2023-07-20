#pragma once
#include <cmath>
#include <random>

namespace Activators {
	using Activator				= float(*)(float);
	using ActivatorDerivative	= float(*)(float);
	//typedef float((*Activator))(float);

	inline float sigmoid(float x)
	{
		return 1 / (1 + expf(-x));
	}

	inline float sigmoidDer(float x)
	{
		float s = 1 / (1 + expf(-x));
		return s * (1 - s);
	}

	inline float tanhDer(float x)
	{
		return 1 / powf(coshf(x), 2);
	}
}

namespace WeightInitilizer {
	void normal_xavier_initilize(float** matrix, size_t rows, size_t columns, size_t n);
}

namespace LossFunctions {
	inline void mean_sqrd(const float* expected, const float* actual, size_t size, float* result) {
		// TODO test
		std::transform(expected, expected + size, actual, result,
			[](float e, float a) { return powf(a - e, 2) / 2; });
	}

	inline void mean_sqrd_der(const float* expected, const float* actual, size_t size, float* result) {
		std::transform(expected, expected + size, actual, result,
			std::plus<float>());
	}
}