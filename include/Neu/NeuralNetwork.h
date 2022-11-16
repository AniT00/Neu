#pragma once
#include <math.h>
//#include "Matrix.h"

class NeuralNetwork
{
private:
	typedef float((*Activator))(float);

	inline static float sigmoid(float x)
	{
		return 1 / (1 + expf(-x));
	}

	inline static float sigmoidDer(float x)
	{
		return 1 / (1 + expf(-x)) * (1 - 1 / (1 + expf(-x)));
	}

	inline static float sigmoidDerOptimized(float sigm_x)
	{
		return sigm_x * (1 - sigm_x);
	}

	inline static float tanhDer(float x)
	{
		return 1 / powf(coshf(x), 2);
	}

	inline static float tanhDerOptimized(float tanh_x)
	{
		return 1 - powf(tanh_x, 2);
	}

	float** _outputs;
	float** _sums;
	float*** _weights;
	float** _bias;
	float** _deltas;

	Activator* _activator;
	Activator* _activatorDer;
	float** _activatorDerArg = nullptr;

	float _learningRate = 0.005f;

	size_t _layerNum;
	size_t* _sizes;

	/// <summary>
	/// store index of biggest output of last predict()</see> after answer() call
	/// </summary>
	int _lastAnswer = 0;

public:
	NeuralNetwork(size_t* sizes, size_t layerNum);

	void predict(float* input);

	void train(float* input, float* output);

	float loss(float* output) const;

	/// <summary>
	/// return index of biggest output. Caches value
	/// </summary>
	/// <returns></returns>
	unsigned int answer();

	float* const output() const;

	void setLearningRate(float value);

	void useOptimizedDerivative(bool value);

	~NeuralNetwork();
};