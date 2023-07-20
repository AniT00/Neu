#include "NeuralNetwork.h"
#include <random>
#include <assert.h>
#include <numeric>


NeuralNetwork::NeuralNetwork(std::initializer_list<size_t> layers) {
	m_layerNum = layers.size() - 1;
	m_layers = new Layer * [m_layerNum];
	auto it = layers.begin();
	m_inputSize = *it;
	m_input = new float[m_inputSize];
	const float* layerInput = m_input;
	for (int i = 0; i < m_layerNum; i++, it++) {
		m_layers[i] = new Layer(layerInput, *it, *(it + 1), &m_learningRate);
		layerInput = m_layers[i]->getOutput();
	}
	m_outputSize = *it;
	m_batch_error = new float[m_outputSize];
	m_error_delta = new float[m_outputSize];
	m_batch_error_delta = new float[m_outputSize];
	m_output = m_layers[m_layerNum - 1]->getOutput();
	m_batch_output = new float [m_outputSize];
	m_expected_batch_output = new float[m_outputSize] {0};

}

const float* NeuralNetwork::predict(float* input)
{
	_lastAnswer = -1;

	for (size_t i = 0; i < m_inputSize; i++) {
		m_input[i] = input[i];
	}
	for (int i = 0; i < m_layerNum; i++) {
		m_layers[i]->calculate();
	}
	return m_output;
}

void NeuralNetwork::train()
{
	assert(m_sample != nullptr && m_expected != nullptr);

	if (log) m_logger << "Training started.\n";
	for (size_t i = 0; i < m_epochs; i++)
	{
		if (log) m_logger << "\tEpoch: " << i << '\n';
		for (size_t j = 0; j < m_sampleSize / m_batchSize; j++)
		{

			if (log) m_logger << "[Batch: " << j << "]\n";
			const float* output;
			std::fill(m_expected_batch_output, m_expected_batch_output + m_outputSize, 0);
			std::fill(m_batch_output, m_batch_output + m_batchSize, 0);
			for (int k = 0; k < m_batchSize; k++) {

				output = predict(m_sample + (j * m_batchSize + k) * m_inputSize);

				const float* expected = m_expected + j * m_batchSize + k;
				std::transform(expected, expected + m_outputSize,
					m_expected_batch_output,
					m_expected_batch_output, std::plus<float>());

				 for (int i = 0; i < m_outputSize; i++) {
					 m_batch_output[i] += output[i];
				 }
				if (log) {
					m_logger << "Input: ";
					std::for_each(m_input, m_input + m_inputSize, [this](float v) { m_logger << v << ' '; });
					m_logger <<'\n';
					m_logger << "Output: ";
					std::for_each(m_output, m_output+ m_outputSize, [this](float v) { m_logger << v << ' '; });
					m_logger << '\n';
					//m_logger << "Accuracy: " << total_loss(m_expected + (j), m_batch_output) << '\n';
					m_logger << '\n';
				}
			}
			calculateBatchError();
			float loss = std::accumulate(m_batch_error, m_batch_error + m_outputSize, 0);
			if (loss < m_target_loss) {
				m_logger << "Successfully reduced losses to target: " << m_target_loss << "\n";
				break;
			}
			backpropagate(j);

		}
	}

	/*for (int i = 0; i < _sizes[_layerNum - 1]; i++)
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
	}*/
}

void NeuralNetwork::backpropagate(size_t batch_N)
{
	//calculateError(m_expected);
	const float* delta = calculateBatchErrorDelta(m_expected + batch_N * m_batchSize);
	/*for (int i = 0; i < m_outputSize; i++) {
		std::cout << std::setw(10) << std::setprecision(2) << delta[i];
	}
	std::cout << '\n';*/
	for (size_t i = m_layerNum - 1; i != std::numeric_limits<size_t>::max(); i--) {
		const float* n_delta = m_layers[i]->calculateDelta(delta);
		m_layers[i]->backpropagate(delta);
		delta = n_delta;
		/*for (int j = 0; j < m_layers[i]->inputSize(); j++) {
			std::cout << std::setw(10) << std::setprecision(2) << delta[j];
		}
		std::cout << '\n';*/
	}
}

float* NeuralNetwork::calculateBatchError()
{
	m_cost_f(m_expected_batch_output, m_output, m_outputSize, m_batch_error);
	/*for (size_t i = 0; i < m_outputSize; i++) {
		m_batch_error[i] = m_cost_f(m_expected_batch_output[i], m_output[i]);
	}*/
	return m_batch_error;
}

float* NeuralNetwork::calculateBatchErrorDelta(const float* expected)
{
	// TODO
	std::fill(m_batch_error_delta, m_batch_error_delta + m_outputSize, 0);
	for (size_t i = 0; i < m_batchSize; i++)
	{
		m_cost_der_f(expected, m_batch_output, m_outputSize, m_error_delta);
		std::transform(m_error_delta, m_error_delta + m_outputSize, m_batch_error_delta, std::plus<float>());
		/*for (size_t j = 0; j < m_outputSize; j++) {
			m_batch_error_delta[j] += LossFunctions::mean_sqrd_der(expected[i * m_outputSize + j], m_batch_output[j]);
		}*/
	}
	return m_batch_error_delta;
}

float NeuralNetwork::total_loss(const float* expected, const float* actual) const
{
	switch (m_answerType)
	{
	case NeuralNetwork::INDEX:
		return lossIndexAnswer(expected, actual);
	case NeuralNetwork::ARRAY:
		return lossArrayAnswer(expected, actual);
	default:
		return 0;
		break;
	}
}

float NeuralNetwork::lossArrayAnswer(const float* expected, const float* actual) const {
	float losses = 0.f;
	for (size_t i = 0; i < m_layers[m_layerNum - 1]->size(); i++)
	{
		losses += LossFunctions::mean_sqrd(m_expected[i] * m_batchSize, actual[i]);
	}
	return losses;
}

float NeuralNetwork::lossIndexAnswer(const float* expected, const float* actual) const {
	float losses = 0.f;
	for (size_t i = 0; i < m_layers[m_layerNum - 1]->size(); i++)
	{
		losses += LossFunctions::mean_sqrd((*m_expected == i) * m_batchSize, actual[i]);
	}
	return losses;
}

void NeuralNetwork::logOutput() {
	std::cout << "Output:\n";
	std::for_each(m_output, m_output + m_outputSize, [this](float v) { m_logger << v << ' '; });
	std::cout << '\n';
}

//unsigned int NeuralNetwork::answer()
//{
//	if (_lastAnswer > -1)
//	{
//		return _lastAnswer;
//	}
//
//	float max = _outputs[_layerNum - 1][0];
//	unsigned int ind = 0;
//	for (int k = 1; k < 3; k++)
//	{
//		if (_outputs[_layerNum - 1][k] > max)
//		{
//			max = _outputs[_layerNum - 1][k];
//			ind = k;
//		}
//	}
//	_lastAnswer = ind;
//	return ind;
//}

const float* NeuralNetwork::getOutput() const
{
    return m_output;
}

NeuralNetwork& NeuralNetwork::setLearningRate(float value)
{
	assert(value > 0.f);
	m_learningRate = value;
	return *this;
}

NeuralNetwork& NeuralNetwork::setInputSample(float* sample, size_t size) {
	m_sample = sample;
	m_sampleSize = size;
	return *this;
}

NeuralNetwork& NeuralNetwork::setAnswerType(Answer type) {
	m_answerType = type;
	return *this;
}

NeuralNetwork& NeuralNetwork::setAnswers(float* answers) {
	m_expected = answers;
	return* this;
}

NeuralNetwork& NeuralNetwork::setBatchSize(size_t size) {
	m_batchSize = size;
	return *this;
}

NeuralNetwork& NeuralNetwork::setTargetBatchLoss(float loss)
{
	m_target_loss = loss;
	return *this;
}

NeuralNetwork& NeuralNetwork::setEpochs(size_t count) {
	m_epochs = count;
	return *this;
}

NeuralNetwork& NeuralNetwork::setLogger(std::ostream* stream) {
	m_logger = Logger(stream);
	log = stream != nullptr;
	return *this;
}

NeuralNetwork& NeuralNetwork::logOutput(bool log) {
	m_logOutput = log;
	return *this;
}

NeuralNetwork::~NeuralNetwork()
{
	for (int i = 0; i < m_layerNum; i++) {
		delete m_layers[i];
	}
	delete[] m_layers;
}