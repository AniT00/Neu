#pragma once
#include "Layer.h"
#include "Logger.h"

#include <cmath>
#include <initializer_list>
#include <functional>
#include <iostream>
//#include "Matrix.h"

class NeuralNetwork
{
public:
	using cost_func = void(*)(const float*, const float*, size_t, float*);

	NeuralNetwork(std::initializer_list<size_t> layers);

	const float* predict(float* input);

	void train();


	/// <summary>
	/// return index of biggest output. Caches value
	/// </summary>
	/// <returns></returns>
	//unsigned int answer();

	const float* getOutput() const;

	NeuralNetwork& setLearningRate(float value);

	NeuralNetwork& setInputSample(float* sample, size_t size);

	NeuralNetwork& setAnswers(float* answers);

	NeuralNetwork& setBatchSize(size_t size);

	NeuralNetwork& setTargetBatchLoss(float loss);

	NeuralNetwork& setEpochs(size_t count);

	NeuralNetwork& setLogger(std::ostream* stream);

	NeuralNetwork& logOutput(bool log);

	~NeuralNetwork();

private:
	void backpropagate(size_t batch_N);

	float* calculateBatchError();

	float* calculateBatchErrorDelta(const float* expected);

	void logOutput();

	float* m_input;
	size_t m_inputSize;

	size_t m_outputSize;
	const float* m_output;

	Layer** m_layers;
	size_t m_layerNum;

	float* m_batch_error;
	float* m_error_delta;
	float* m_batch_error_delta;
	float* m_batch_output;
	float* m_expected_batch_output;
	float m_target_loss;

	cost_func m_cost_f;
	cost_func m_cost_der_f;
	

	float m_learningRate = 0.005f;
	float* m_sample = nullptr;
	size_t m_sampleSize = 0;
	size_t m_batchSize = 1;
	size_t m_epochs = 1;

	float* m_expected = nullptr;

	size_t _layerNum;
	size_t* _sizes;

	Logger m_logger;
	bool log = false;
	bool m_logOutput = false;

	/// <summary>
	/// store index of biggest output of last predict()</see> after answer() call
	/// </summary>
	int _lastAnswer = 0;
};