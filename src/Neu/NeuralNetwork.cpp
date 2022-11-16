#include "NeuralNetwork.h"
#include <random>

NeuralNetwork::NeuralNetwork(size_t* sizes, size_t layerNum)
{
	_layerNum = layerNum;
	_sizes = new size_t[_layerNum];
	for (int i = 0; i < _layerNum; i++)
	{
		_sizes[i] = sizes[i];
	}

	_activator = new Activator[_layerNum - 1];
	_activatorDer = new Activator[_layerNum - 1];
	for (int i = 0; i < _layerNum - 2; i++)
	{
		_activator[i] = tanhf;
		_activatorDer[i] = tanhDerOptimized;
	}
	_activator[_layerNum - 2] = tanhf;
	_activatorDer[_layerNum - 2] = tanhDerOptimized;

	_outputs = new float* [_layerNum];
	_weights = new float** [_layerNum - 1];
	_sums = new float* [_layerNum - 1];
	_bias = new float* [_layerNum - 1];
	_deltas = new float* [_layerNum - 1];

	_activatorDerArg = _outputs + 1;

	for (int i = 0; i < _layerNum - 1; i++)
	{
		_outputs[i] = new float[_sizes[i]]{ 0 };
		_sums[i] = new float[_sizes[i + 1]];
		_deltas[i] = new float[_sizes[i + 1]];
		_bias[i] = new float[_sizes[i + 1]]{0};

		_weights[i] = new float*[_sizes[i]];
		for (int j = 0; j < _sizes[i]; j++)
		{
			_weights[i][j] = new float[_sizes[i + 1]];
			for (int k = 0; k < _sizes[i + 1]; k++)
			{
				_weights[i][j][k] = (float)(rand() % 20000 - 10000) / 10000;
			}
		}
	}
	_outputs[_layerNum - 1] = new float[_sizes[_layerNum - 1]]{ 0 };
}

void NeuralNetwork::predict(float* input)
{
	_lastAnswer = -1;
	
	for (int i = 0; i < _sizes[0]; i++)
	{
		_outputs[0][i] = input[i];
	}

	for (int i = 0; i < _layerNum - 1; i++)
	{
		for (int j = 0; j < _sizes[i + 1]; j++)
		{
			_sums[i][j] = _bias[i][j];

			for (int k = 0; k < _sizes[i]; k++)
			{
				_sums[i][j] += _outputs[i][k] * _weights[i][k][j];
			}
			
			_outputs[i + 1][j] = _activator[i](_sums[i][j]);
		}
	}
}

void NeuralNetwork::train(float* input, float* output)
{
	predict(input);

	for (int i = 0; i < _sizes[_layerNum - 1]; i++)
	{
		_deltas[_layerNum - 2][i] = (_outputs[_layerNum - 1][i] - output[i]);
	}

	for (size_t i = _layerNum - 2; i > 0; i--)
	{
		for (size_t j = 0; j < _sizes[i]; j++)
		{
			_deltas[i - 1][j] = 0;
			for (size_t k = 0; k < _sizes[i + 1]; k++)
			{
				_deltas[i - 1][j] += _deltas[i][k] * _weights[i][j][k] * _activatorDer[i](_activatorDerArg[i][k]);
			}
		}
	}

	for (size_t i = 0; i < _layerNum - 1; i++)
	{
		for (size_t j = 0; j < _sizes[i + 1]; j++)
		{
			float weightDelta = _deltas[i][j] * _activatorDer[i](_activatorDerArg[i][j]);
			for (size_t k = 0; k < _sizes[i]; k++)
			{
				_weights[i][k][j] -= weightDelta * _outputs[i][k] * _learningRate;
			}
		}

		for (size_t j = 0; j < _sizes[i + 1]; j++)
		{
			_bias[i][j] -= _deltas[i][j] * _learningRate;
		}
	}
}

float NeuralNetwork::loss(float* output) const
{
	float losses = 0;
	for (int i = 0; i < _sizes[_layerNum - 1]; i++)
	{
		losses += powf(_outputs[_layerNum - 1][i] - output[i], 2) / 2;
	}
    return losses;
}

unsigned int NeuralNetwork::answer()
{
	if (_lastAnswer > -1)
	{
		return _lastAnswer;
	}

	float max = _outputs[_layerNum - 1][0];
	unsigned int ind = 0;
	for (int k = 1; k < 3; k++)
	{
		if (_outputs[_layerNum - 1][k] > max)
		{
			max = _outputs[_layerNum - 1][k];
			ind = k;
		}
	}
	_lastAnswer = ind;
	return ind;
}

float* const NeuralNetwork::output() const
{
    return _outputs[_layerNum - 1];
}

void NeuralNetwork::useOptimizedDerivative(bool value)
{
	_activatorDerArg = value ? _outputs + 1 : _sums;
}

NeuralNetwork::~NeuralNetwork()
{
	for (int i = 0; i < _layerNum - 1; i++)
	{
		for (int j = 0; j < _sizes[i]; j++)
		{
			delete[] _weights[i][j];
		}

		delete[] _outputs[i];
		delete[] _sums[i];
		delete[] _deltas[i];
		delete[] _bias[i];
		delete[] _weights[i];
	}
	delete[] _outputs[_layerNum - 1];
	
	delete[] _outputs;
	delete[] _sums;
	delete[] _deltas;
	delete[] _bias;
	delete[] _weights;
	delete[] _sizes;
}

void NeuralNetwork::setLearningRate(float value)
{
	if (value > 0.f)
	{
		_learningRate = value;
	}
	else 
	{
		throw std::exception();
	}
}


