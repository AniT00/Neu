#pragma once
#include "Layer.h"
#include "Logger.h"

#include <cmath>
#include <functional>
#include <initializer_list>
#include <iostream>

class NeuralNetwork
{
public:
  // Primal purpose is to define behavior of loss function.
  enum class NetworkType
  {
    // Default network state.
    DEFAULT,
    // Use only if network has one output to classify into classes. Expected
    // answer array is array of target class 0 or 1 for each sample.
    BINARY_CLASSIFIER,
    // Use to reduce answer array size. Network assumes the the answer array
    // element with index i contain index of output neuron that should be
    // activated for sample with index i.
    CLASSIFIER
  };

  NeuralNetwork();

  NeuralNetwork(std::initializer_list<size_t> layers);

  const float* predict(float* input);

  void train();

  /// <summary>
  /// return index of biggest output. Caches value
  /// </summary>
  /// <returns></returns>
  // unsigned int answer();

  const float* getOutput() const;

  NeuralNetwork& setType(NetworkType type);

  NeuralNetwork& setLearningRate(float value);

  NeuralNetwork& setActivator(const Activator& activator);

  NeuralNetwork& setLossFunction(const LossFunction& loss);

  NeuralNetwork& setInputSample(float* sample, size_t size);

  NeuralNetwork& setAnswers(float* answers);

  NeuralNetwork& setBatchSize(size_t size);

  NeuralNetwork& setTargetBatchLoss(float loss);

  float getTargetBatchLoss() const;

  float getLoss() const;

  float getAccuracy() const;

  NeuralNetwork& setEpochs(size_t count);

  NeuralNetwork& setLogger(std::ostream* stream);

  NeuralNetwork& logPrediction(bool log);

  ~NeuralNetwork();

private:
  void backpropagate(size_t batch_N);

  float* calculateBatchError();

  float* calculateBatchErrorDelta();

  void addOutputToBatch(const float* output);

  void addExpectedToBatch(const float* expected);

  void logPrediction();

  NetworkType m_networkType = NetworkType::DEFAULT;

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
  float m_loss;
  float m_correct = 0;
  float m_accuracy = 0;

  Activator m_activatorFunc;
  LossFunction m_lossFunc;

  float m_learningRate = 0.005f;
  float* m_sample = nullptr;
  size_t m_sampleSize = 0;
  size_t m_batchSize = 1;
  size_t m_epochs = 1;

  float* m_target;
  float* m_answers = nullptr;
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