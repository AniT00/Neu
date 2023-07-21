#pragma once
#include "Activators.h"

#include <initializer_list>
#include <assert.h>
#include <iostream>
#include <iomanip>

class Layer {
public:
	Layer(const float* input, size_t inputSize, size_t layerSize, float* learningRate) {
		m_input = input;
		m_inputSize = inputSize;

		m_layerSize = layerSize;
		m_weightedInput = new float[m_layerSize] { 0 };
		m_output = new float[m_layerSize] { 0 };

		m_learningRate = learningRate;

		m_delta = new float[m_inputSize];
		m_bias = new float[m_layerSize] { 0 };
		m_activator = Activators::sigmoid;
		m_activatorDer = Activators::sigmoidDer;

		m_weights = new float* [m_layerSize];
		float* weights = new float[m_inputSize * m_layerSize] { 0 };
		for (int i = 0; i < m_layerSize; i++) {
			m_weights[i] = &weights[i * m_inputSize];
		}

		WeightInitilizer::normal_xavier_initilize(m_weights, m_layerSize, m_inputSize, m_inputSize);
	}

	float* calculate() {
		for (size_t i = 0; i < m_layerSize; i++)
		{
			m_weightedInput[i] = m_bias[i];

			for (size_t j = 0; j < m_inputSize; j++)
			{
				m_weightedInput[i] += m_input[j] * m_weights[i][j];
			}

			m_output[i] = m_activator(m_weightedInput[i]);
		}
		return m_output;
	}

	void backpropagate(const float* delta) {
		for (size_t i = 0; i < m_layerSize; i++) {
			float weightDelta = delta[i] * m_activatorDer(m_weightedInput[i]);
			for (size_t j = 0; j < m_inputSize; j++)
			{
				m_weights[i][j] -= weightDelta * m_input[j] * *m_learningRate;
			}
			m_bias[i] -= weightDelta * *m_learningRate;
		}
	}

	const float* calculateDelta(const float* delta)
	{
		for (size_t i = 0; i < m_inputSize; i++)
		{
			m_delta[i] = 0;
			for (size_t j = 0; j < m_layerSize; j++)
			{
				m_delta[i] += delta[j] * m_weights[j][i] * m_activatorDer(m_weightedInput[i]);
			}
		}
		return m_delta;
	}

	const float** getWeights() {
		return (const float** )m_weights;
	}

	const float* getOutput() const {
		return m_output;
	}

	size_t size() const {
		return m_layerSize;
	}

	size_t inputSize() const {
		return m_inputSize;
	}

	~Layer()
	{
		delete[] *m_weights;
		delete[] m_weights;
		delete[] m_weightedInput;
		delete[] m_output;
		delete[] m_bias;
	}
	
	friend std::ostream& operator<<(std::ostream& os, const Layer& l) {
		for (size_t i = 0; i < l.m_layerSize; i++)
		{
			for (size_t j = 0; j < l.m_inputSize; j++)
			{
				os << std::setw(10) << std::setprecision(2) << l.m_weights[i][j];
			}
			os << '\n';
		}
		return os;
	}

private:
	float** m_weights;
	float* m_weightedInput;
	const float* m_input;
	float* m_output;
	float* m_delta;
	size_t m_inputSize;
	size_t m_layerSize;
	float* m_bias;
	const float* m_learningRate;
	Activators::Activator m_activator;
	Activators::ActivatorDerivative m_activatorDer;
};