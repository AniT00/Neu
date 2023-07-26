#pragma once
#include <cmath>
#include <random>

using activator_f = void (*)(const float* input, size_t size, float* result);

using loss_f = void (*)(const float* expected,
                        const float* actual,
                        size_t size,
                        float* result);

class Activator
{
public:
  Activator(activator_f activator, activator_f activatorDerivative);

  void der(float* input, size_t size, float* result) const;

  void operator()(float* input, size_t size, float* result) const;

  static const Activator Sigmoid;

private:
  activator_f m_activator, m_activatorDer;
};

namespace activators {

inline void
sigmoid(const float* input, size_t size, float* result)
{
  std::transform(
    input, input + size, result, [](float x) { return 1 / (1 + expf(-x)); });
}

inline void
sigmoidDer(const float* input, size_t size, float* result)
{
  std::transform(input, input + size, result, [](float x) {
    float s = 1 / (1 + expf(-x));
    return s * (1 - s);
  });
}

}

namespace weight_initilizer {
void
normal_xavier_initilize(float** matrix, size_t rows, size_t columns, size_t n);
}

class LossFunction
{
public:
  LossFunction(loss_f activator, loss_f activatorDerivative);

  void der(const float* expected,
           const float* actual,
           size_t size,
           float* result) const;

  void operator()(const float* expected,
                  const float* actual,
                  size_t size,
                  float* result) const;

  static LossFunction MeanSqrt;
  static LossFunction CrossEntropy;

private:
  loss_f m_activator, m_activatorDer;
};

namespace loss_functions {

inline void
cross_entropy(const float* expected,
              const float* actual,
              size_t size,
              float* result)
{
  std::transform(actual, actual + size, expected, result, [](float a, float e) {
    return -(e * log2f(a) + (1 - e) * log2f(1 - a));
  });
}

inline void
cross_enrtropy_der(const float* expected,
                   const float* actual,
                   size_t size,
                   float* result)
{
  std::transform(actual, actual + size, expected, result, [](float a, float e) {
    return (-e / a) + (1 - e) / (1 - a);
  });
}

inline void
mean_sqrd(const float* expected,
          const float* actual,
          size_t size,
          float* result)
{
  std::transform(actual, actual + size, expected, result, [](float a, float e) {
    return powf(a - e, 2) / 2;
  });
}

inline void
mean_sqrd_der(const float* expected,
              const float* actual,
              size_t size,
              float* result)
{
  std::transform(actual, actual + size, expected, result, std::minus<float>());
}

}