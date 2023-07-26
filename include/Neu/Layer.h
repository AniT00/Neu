#pragma once
#include "Activators.h"
#include "clblast.h"

#include <cassert>
#include <initializer_list>
#include <iomanip>
#include <iostream>

class Layer
{
public:
  Layer(const float* input,
        size_t inputSize,
        size_t layerSize,
        float* learningRate);

  float* calculate();

  void backpropagate(const float* delta);

  const float* calculateDelta(const float* delta);

  void setActivator(const Activator* activator);

  const float** getWeights();

  const float* getOutput() const;

  size_t size() const;

  size_t inputSize() const;

  ~Layer();

  friend std::ostream& operator<<(std::ostream& os, const Layer& l);

private:
  float** m_weights;
  float* m_weightedInput;
  const float* m_input;
  float* m_output;
  float* m_derivative;
  float* m_delta;
  bool m_validDerivative = 0;
  size_t m_inputSize;
  size_t m_layerSize;
  float* m_bias;
  cl_mem m_matrix;
  const cl_mem* m_inp;
  const float* m_learningRate;
  const Activator* m_activator;
};