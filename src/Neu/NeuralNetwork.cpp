#include "Neu/NeuralNetwork.h"
#include "clblast.h"

#include <assert.h>
#include <numeric>
#include <random>

NeuralNetwork::NeuralNetwork(std::initializer_list<size_t> layers)
  : m_costFunc(LossFunction::MeanSqrt)
  , m_activatorFunc(Activator::Sigmoid)
{
  m_layerNum = layers.size() - 1;
  m_layers = new Layer*[m_layerNum];
  auto it = layers.begin();
  m_inputSize = *it;
  m_input = new float[m_inputSize];
  const float* layerInput = m_input;
  for (int i = 0; i < m_layerNum; i++, it++) {
    m_layers[i] = new Layer(layerInput, *it, *(it + 1), &m_learningRate);
    m_layers[i]->setActivator(&m_activatorFunc);
    layerInput = m_layers[i]->getOutput();
  }

  m_outputSize = *it;
  m_batch_error = new float[m_outputSize];
  m_error_delta = new float[m_outputSize];
  m_batch_error_delta = new float[m_outputSize];
  m_output = m_layers[m_layerNum - 1]->getOutput();
  m_batch_output = new float[m_outputSize];
  m_expected_batch_output = new float[m_outputSize]{ 0 };
  m_target_loss = 0;
}

const float*
NeuralNetwork::predict(float* input)
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

void
NeuralNetwork::train()
{
  assert(m_sample != nullptr && m_expected != nullptr);

  if (log)
    m_logger << "Training started.\n";
  for (size_t i = 0; i < m_epochs; i++) {
    if (log)
      m_logger << "\tEpoch: " << i << '\n';
    for (size_t j = 0; j < m_sampleSize / m_batchSize; j++) {

      if (log)
        m_logger << "[Batch: " << j << "]\n";
      const float* output;
      std::fill(
        m_expected_batch_output, m_expected_batch_output + m_outputSize, 0);
      std::fill(m_batch_output, m_batch_output + m_batchSize, 0);
      for (int k = 0; k < m_batchSize; k++) {

        output = predict(m_sample + (j * m_batchSize + k) * m_inputSize);

        const float* expected = m_expected + j * m_batchSize + k;
        std::transform(expected,
                       expected + m_outputSize,
                       m_expected_batch_output,
                       m_expected_batch_output,
                       std::plus<float>());

        for (int i = 0; i < m_outputSize; i++) {
          m_batch_output[i] += output[i];
        }
        if (log) {
          m_logger << "Input: ";
          std::for_each(m_input, m_input + m_inputSize, [this](float v) {
            m_logger << v << ' ';
          });
          m_logger << '\n';
          m_logger << "Output: ";
          std::for_each(m_output, m_output + m_outputSize, [this](float v) {
            m_logger << v << ' ';
          });
          m_logger << '\n';
          // m_logger << "Accuracy: " << total_loss(m_expected + (j),
          // m_batch_output) << '\n';
          m_logger << '\n';
        }
      }
      calculateBatchError();
      float loss =
        std::accumulate(m_batch_error, m_batch_error + m_outputSize, 0.f);
      if (loss < m_target_loss) {
        m_logger << "Successfully reduced losses to target: " << m_target_loss
                 << "\n";
        return;
      }
      backpropagate(j);
    }
  }
}

void
NeuralNetwork::backpropagate(size_t batch_N)
{
  const float* delta =
    calculateBatchErrorDelta(m_expected + batch_N * m_batchSize);
  for (size_t i = m_layerNum - 1; i != std::numeric_limits<size_t>::max();
       i--) {
    const float* n_delta = m_layers[i]->calculateDelta(delta);
    m_layers[i]->backpropagate(delta);
    delta = n_delta;
  }
}

float*
NeuralNetwork::calculateBatchError()
{
  m_costFunc(m_expected_batch_output, m_output, m_outputSize, m_batch_error);
  return m_batch_error;
}

float*
NeuralNetwork::calculateBatchErrorDelta(const float* expected)
{
  // TODO
  std::fill(m_batch_error_delta, m_batch_error_delta + m_outputSize, 0);
  for (size_t i = 0; i < m_batchSize; i++) {
    m_costFunc.der(expected, m_batch_output, m_outputSize, m_error_delta);
    std::transform(m_error_delta,
                   m_error_delta + m_outputSize,
                   m_batch_error_delta,
                   m_batch_error_delta,
                   std::plus<float>());
  }
  return m_batch_error_delta;
}

void
NeuralNetwork::logOutput()
{
  std::cout << "Output:\n";
  std::for_each(m_output, m_output + m_outputSize, [this](float v) {
    m_logger << v << ' ';
  });
  std::cout << '\n';
}

const float*
NeuralNetwork::getOutput() const
{
  return m_output;
}

NeuralNetwork&
NeuralNetwork::setLearningRate(float value)
{
  assert(value > 0.f);
  m_learningRate = value;
  return *this;
}

NeuralNetwork&
NeuralNetwork::setActivator(const Activator& activator)
{
  m_activatorFunc = activator;
  return *this;
}

NeuralNetwork&
NeuralNetwork::setLossFunction(const LossFunction& loss)
{

  return *this;
}

NeuralNetwork&
NeuralNetwork::setInputSample(float* sample, size_t size)
{
  m_sample = sample;
  m_sampleSize = size;
  return *this;
}

NeuralNetwork&
NeuralNetwork::setAnswers(float* answers)
{
  m_expected = answers;
  return *this;
}

NeuralNetwork&
NeuralNetwork::setBatchSize(size_t size)
{
  m_batchSize = size;
  return *this;
}

NeuralNetwork&
NeuralNetwork::setTargetBatchLoss(float loss)
{
  m_target_loss = loss;
  return *this;
}

NeuralNetwork&
NeuralNetwork::setEpochs(size_t count)
{
  m_epochs = count;
  return *this;
}

NeuralNetwork&
NeuralNetwork::setLogger(std::ostream* stream)
{
  m_logger = Logger(stream);
  log = stream != nullptr;
  return *this;
}

NeuralNetwork&
NeuralNetwork::logOutput(bool log)
{
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