#include "Neu/Layer.h"

Layer::Layer(const float* input,
             size_t inputSize,
             size_t layerSize,
             float* learningRate)
  : m_activator(&Activator::Sigmoid)
{
  m_input = input;
  m_inputSize = inputSize;

  m_layerSize = layerSize;
  m_weightedInput = new float[m_layerSize]{ 0 };
  m_output = new float[m_layerSize]{ 0 };
  m_derivative = new float[m_layerSize]{ 0 };

  m_learningRate = learningRate;

  m_delta = new float[m_inputSize];
  m_bias = new float[m_layerSize]{ 0 };

  m_weights = new float*[m_layerSize];
  float* weights = new float[m_inputSize * m_layerSize]{ 0 };
  for (int i = 0; i < m_layerSize; i++) {
    m_weights[i] = &weights[i * m_inputSize];
  }

  weight_initilizer::normal_xavier_initilize(
    m_weights, m_layerSize, m_inputSize, m_inputSize);
}

float*
Layer::calculate()
{
  // clblast::Gemv<float>(clblast::Layout::kRowMajor, clblast::Transpose::kNo,
  // m_layerSize, m_inputSize, 1, m_matrix, 0, m_inputSize, *m_inp, 0, )
  std::copy(m_bias, m_bias + m_layerSize, m_weightedInput);
  for (size_t i = 0; i < m_layerSize; i++) {

    for (size_t j = 0; j < m_inputSize; j++) {
      m_weightedInput[i] += m_input[j] * m_weights[i][j];
    }
  }
  (*m_activator)(m_weightedInput, m_layerSize, m_output);
  return m_output;
}

void
Layer::backpropagate(const float* delta)
{
  if (!m_validDerivative) {
    m_activator->der(m_weightedInput, m_layerSize, m_derivative);
    std::transform(delta,
                   delta + m_layerSize,
                   m_derivative,
                   m_derivative,
                   std::multiplies());
  }
  m_validDerivative = false;
  // TODO check multip
  for (size_t i = 0; i < m_layerSize; i++) {
    for (size_t j = 0; j < m_inputSize; j++) {
      m_weights[i][j] -= m_derivative[i] * m_input[j] * *m_learningRate;
    }
    m_bias[i] -= m_derivative[i] * *m_learningRate;
  }
}

const float*
Layer::calculateDelta(const float* delta)
{
  std::fill(m_delta, m_delta + m_inputSize, 0);
  m_activator->der(m_weightedInput, m_layerSize, m_derivative);
  std::transform(
    delta, delta + m_layerSize, m_derivative, m_derivative, std::multiplies());
  for (size_t i = 0; i < m_inputSize; i++) {
    for (size_t j = 0; j < m_layerSize; j++) {
      m_delta[i] += m_weights[j][i] * m_derivative[j];
    }
  }
  m_validDerivative = true;
  return m_delta;
}

void
Layer::setActivator(const Activator* activator)
{
  m_activator = activator;
}

const float**
Layer::getWeights()
{
  return (const float**)m_weights;
}

const float*
Layer::getOutput() const
{
  return m_output;
}

size_t
Layer::size() const
{
  return m_layerSize;
}

size_t
Layer::inputSize() const
{
  return m_inputSize;
}

Layer::~Layer()
{
  delete[] * m_weights;
  delete[] m_weights;
  delete[] m_weightedInput;
  delete[] m_output;
  delete[] m_bias;
}

std::ostream&
operator<<(std::ostream& os, const Layer& l)
{
  for (size_t i = 0; i < l.m_layerSize; i++) {
    for (size_t j = 0; j < l.m_inputSize; j++) {
      os << std::setw(10) << std::setprecision(2) << l.m_weights[i][j];
    }
    os << '\n';
  }
  return os;
}
