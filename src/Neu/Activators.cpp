#include "Neu/Activators.h"

const ParametricActivator<float> ParametricActivator<float>::LeakRelu(activators::relu,
                                        activators::relu_der,
                                        0.01f);
const ParametricActivator<float> ParametricActivator<float>::Relu(activators::relu,
                                    activators::relu_der,
                                    0.f);

const Activator Activator::Sigmoid(activators::sigmoid,
                                   activators::sigmoid_der);

Activator::Activator(activator_f activator, activator_f activatorDerivative)
{
  m_activator = activator;
  m_activator_der = activatorDerivative;
}

void
Activator::der(float* input, size_t size, float* result) const
{
  m_activator_der(input, size, result);
}

void
Activator::operator()(float* input, size_t size, float* result) const
{
  m_activator(input, size, result);
}

LossFunction LossFunction::CrossEntropy(loss_functions::cross_entropy,
                                        loss_functions::cross_enrtropy_der);
LossFunction LossFunction::Softmax(loss_functions::softmax,
                                   loss_functions::softmax_der);
LossFunction LossFunction::MeanSqrt(loss_functions::mean_sqrd,
                                    loss_functions::mean_sqrd_der);

LossFunction::LossFunction(loss_f activator, loss_f activatorDerivative)
{
  m_activator = activator;
  m_activator_der = activatorDerivative;
}

void
LossFunction::der(const float* expected,
                  const float* actual,
                  size_t size,
                  float* result) const
{
  m_activator_der(expected, actual, size, result);
}

void
LossFunction::operator()(const float* expected,
                         const float* actual,
                         size_t size,
                         float* result) const
{
  m_activator(expected, actual, size, result);
}

void
weight_initilizer::normal_xavier_initilize(float** matrix,
                                           size_t rows,
                                           size_t columns,
                                           size_t n)
{
  static std::default_random_engine generator;
  std::normal_distribution<float> normal(0, sqrt(1.f / n));
  for (size_t i = 0; i < rows; i++) {
    for (size_t j = 0; j < columns; j++) {
      matrix[i][j] = normal(generator);
    }
  }
}
