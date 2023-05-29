#pragma once
#include "Activators.h"

#include <initializer_list>

class Layer {
public:
	Layer(size_t inputSize, size_t layerSize) {
		m_inputSize = inputSize;
		m_layerSize = layerSize;
		m_nodes = new float[layerSize];
		if (inputSize > 0) {
			m_weights = new float* [inputSize];
			float* weights = new float[inputSize * layerSize];
			for (int i = 0; i < inputSize; i++) {
				m_weights[i] = &weights[i * layerSize];
			}
		}
		else {
			m_weights = nullptr;
		}
	}

	float* calculate(float* input) {
		for (int j = 0; j < m_layerSize; j++)
		{
			m_sums[j] = m_bias[j];

			for (int k = 0; k < m_inputSize; k++)
			{
				m_sums[j] += m_nodes[k] * m_weights[k][j];
			}

			m_nodes[j] = m_activator(m_sums[j]);
		}
		return m_nodes;
	}

	const float* getOutput() const {
		return m_nodes;
	}

	void setOutput(std::initializer_list<float> output);

private:
	float** m_weights;
	float* m_nodes;
	size_t m_inputSize;
	size_t m_layerSize;
	float* m_sums;
	float* m_bias;
	Activators::Activator m_activator;
};