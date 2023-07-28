#pragma once
#include <cmath>
#include <functional>
#include <numeric>
#include <random>

using activator_f = std::function<void(const float*, size_t, float*)>;

using loss_f = void (*)(const float* expected,
                        const float* actual,
                        size_t size,
                        float* result);

class Activator;

class Activator
{
public:
  Activator(activator_f activator, activator_f activatorDerivative);

  const activator_f& getFunc() { return m_activator; }

  const activator_f& getDerFunc() { return m_activator_der; }

  void der(float* input, size_t size, float* result) const;

  void operator()(float* input, size_t size, float* result) const;

  static const Activator Sigmoid;

private:
  activator_f m_activator, m_activator_der;
};

template<typename... Ts>
class ParametricActivator : public Activator
{
public:
  using unwraped_act_f =
    std::function<void(Ts..., const float*, size_t, float*)>;

  ParametricActivator(unwraped_act_f activator,
                      unwraped_act_f activator_der,
                      Ts... params)
    : m_activator(std::bind_front(activator, params))
    , m_activator_der(std::bind_front(activator_der, params))
    , m_unwraped_function(activator)
    , m_unwraped_der_function(activator_der)
  {
  }

  void setParams(Ts... params)
  {
    m_activator = Activator(std::bind_front(m_unwraped_function, params),
                            std::bind_front(m_unwraped_der_function, params));
  }

  static const ParametricActivator<float> LeakRelu;
  static const ParametricActivator<float> Relu;

private:
  unwraped_act_f m_unwraped_function;
  unwraped_act_f m_unwraped_der_function;
};
namespace activators {
inline void
relu(float neg_mult, const float* input, size_t size, float* result)
{
  std::transform(
    input, input + size, result, [](float x) { return std::max(0.f, x); });
}

inline void
relu_der(float neg_mult, const float* input, size_t size, float* result)
{
  std::transform(
    input, input + size, result, [](float x) { return x > 0 ? 0 : 1; });
}

inline void
sigmoid(const float* input, size_t size, float* result)
{
  std::transform(
    input, input + size, result, [](float x) { return 1 / (1 + expf(-x)); });
}

inline void
sigmoid_der(const float* input, size_t size, float* result)
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

  static LossFunction Softmax;
  static LossFunction CrossEntropy;
  static LossFunction MeanSqrt;

private:
  loss_f m_activator, m_activator_der;
};

namespace loss_functions {

inline void
softmax(const float* expected, const float* actual, size_t size, float* result)
{
  // Write e^y for each output y to calculate sum.
  std::transform(
    actual, actual + size, result, [](float x) { return expf(x); });
  float sum = std::accumulate(result, result + size, 0);
  std::transform(
    actual, actual + size, expected, result, [sum](float a, float e) {
      return expf(a) / sum;
    });
}

inline void
softmax_der(const float* expected,
            const float* actual,
            size_t size,
            float* result)
{
  std::transform(actual, actual + size, expected, result, [](float a, float e) {
    return -(e * log2f(a) + (1 - e) * log2f(1 - a));
  });
}

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