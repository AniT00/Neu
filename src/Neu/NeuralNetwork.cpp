#include "Neu/NeuralNetwork.h"
#include "clblast.h"

#include <assert.h>
#include <numeric>
#include <random>

NeuralNetwork::NeuralNetwork()
{}

NeuralNetwork::NeuralNetwork(std::initializer_list<size_t> layers)
  : m_lossFunc(LossFunction::MeanSqrt)
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
  m_target = new float[m_outputSize];
  m_batch_error = new float[m_outputSize];
  m_error_delta = new float[m_outputSize];
  m_batch_error_delta = new float[m_outputSize];
  m_output = m_layers[m_layerNum - 1]->getOutput();
  m_batch_output = new float[m_outputSize];
  m_expected_batch_output = new float[m_outputSize]{ 0 };
  m_expected = new float[m_outputSize];
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
  assert(m_sample != nullptr && m_answers != nullptr);

  m_correct = 0;

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
      std::fill(m_batch_output, m_batch_output + m_outputSize, 0);
      for (int k = 0; k < m_batchSize; k++) {

        output = predict(m_sample + (j * m_batchSize + k) * m_inputSize);

        std::fill(m_expected, m_expected + m_outputSize, 0);
        m_expected[(int)m_answers[j * m_batchSize + k]] = 1;
        // const float* expected = m_answers + j * m_batchSize + k;
        addExpectedToBatch(m_expected);
        addOutputToBatch(output);

        auto o = (std::max_element(output, output + m_outputSize) - output);
        m_correct += 1 * (o == (int)m_answers[j * m_batchSize + k]);

        if (log) {
          logPrediction();
        }
      }
      calculateBatchError();
      m_loss =
        std::accumulate(m_batch_error, m_batch_error + m_outputSize, 0.f);
      if (m_loss < m_target_loss) {
        m_logger << "Successfully reduced losses to target: " << m_target_loss
                 << "\n";
        m_accuracy = m_correct / (m_sampleSize / m_batchSize);
        return;
      }
      backpropagate(j);
    }
  }
  m_accuracy = m_correct / (m_sampleSize / m_batchSize);
}

void
NeuralNetwork::backpropagate(size_t batch_N)
{
  std::fill(m_expected, m_expected + m_outputSize, 0);
  m_expected[(int)m_answers[batch_N * m_batchSize]] = 1;
  const float* delta = calculateBatchErrorDelta();
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
  m_lossFunc(m_expected_batch_output, m_output, m_outputSize, m_batch_error);
  return m_batch_error;
}

float*
NeuralNetwork::calculateBatchErrorDelta()
{
  m_lossFunc.der(
    m_expected_batch_output, m_batch_output, m_outputSize, m_batch_error_delta);
  return m_batch_error_delta;
  /*using enum NeuralNetwork::NetworkType;
  switch (m_networkType) {
    case NetworkType::DEFAULT:
      std::fill(m_batch_error_delta, m_batch_error_delta + m_outputSize, 0);
      for (size_t i = 0; i < m_batchSize; i++) {
        m_lossFunc.der(
          m_expected_batch_output, m_batch_output, m_outputSize, m_error_delta);
        std::transform(m_error_delta,
                       m_error_delta + m_outputSize,
                       m_batch_error_delta,
                       m_batch_error_delta,
                       std::plus<float>());
      }
      break;
    case BINARY_CLASSIFIER:
      m_batch_error_delta[0] = 0;
      for (size_t i = 0; i < m_batchSize; i++) {
        m_lossFunc.der(expected, m_batch_output, m_outputSize, m_error_delta);

        m_batch_error_delta[0] += m_error_delta[0];
      }
      break;
    case CLASSIFIER:
      std::fill(m_batch_error_delta, m_batch_error_delta + m_outputSize, 0);
      for (size_t i = 0; i < m_batchSize; i++) {
        m_lossFunc.der(expected, m_batch_output, m_outputSize, m_error_delta);

        m_batch_error_delta[0] += m_error_delta[0];
      }
      break;
    default:
      break;
  }*/
}

void
NeuralNetwork::addOutputToBatch(const float* output)
{
  std::transform(output,
                 output + m_outputSize,
                 m_batch_output,
                 m_batch_output,
                 std::plus<float>());
}

void
NeuralNetwork::addExpectedToBatch(const float* expected)
{
  std::transform(expected,
                 expected + m_outputSize,
                 m_expected_batch_output,
                 m_expected_batch_output,
                 std::plus<float>());
  /*using enum NeuralNetwork::NetworkType;
  switch (m_networkType) {
    case NetworkType::DEFAULT:
      std::transform(expected,
                     expected + m_outputSize,
                     m_target_batch_output,
                     m_target_batch_output,
                     std::plus<float>());
      break;
    case BINARY_CLASSIFIER:
    case CLASSIFIER:
      m_target_batch_output[(int)*expected]++;
      break;
    default:
      break;
  }*/
}

void
NeuralNetwork::logPrediction()
{
  m_logger << "Input: ";
  std::for_each(
    m_input, m_input + m_inputSize, [this](float v) { m_logger << v << ' '; });
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

const float*
NeuralNetwork::getOutput() const
{
  return m_output;
}

NeuralNetwork&
NeuralNetwork::setType(NetworkType type)
{
  m_networkType = type;
  return *this;
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

// TODO
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
  m_answers = answers;
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

float
NeuralNetwork::getTargetBatchLoss() const
{
  return m_target_loss;
}

float
NeuralNetwork::getLoss() const
{
  return m_loss;
}

float
NeuralNetwork::getAccuracy() const
{
  return m_accuracy;
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
NeuralNetwork::logPrediction(bool log)
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