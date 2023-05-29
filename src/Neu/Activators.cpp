#include "Activators.h"

inline float Activators::sigmoid(float x)
{
	return 1 / (1 + expf(-x));
}

inline float Activators::sigmoidDer(float x)
{
	float s = 1 / (1 + expf(-x));
	return s * (1 - s);
}

inline float Activators::tanhDer(float x)
{
	return 1 / powf(coshf(x), 2);
}