#include "CsvReader.h"
#include "NeuralNetwork.h"
#include <iostream>

constexpr std::initializer_list<size_t> configuration = { 4, 5, 5, 3 };

using aa = std::function<void(const float*, size_t, float*)>;

void
f(float x)
{
}

inline void
relu(float neg_mult, const float*, size_t, float*)
{
}

int
main()
{    
  CsvReader reader("samples/Iris.csv");
  auto record = reader.next();
  reader.setIdentifiers(record.value().get());
  size_t sampleSize = reader.getLineCount() - 1;
  size_t inputSize = record.value().get().getColumnCount() - 1;
  float* sample = new float[inputSize * sampleSize]{ 0 };
  float* answers = new float[sampleSize]{ 0 };
  for (size_t i = 0; i < sampleSize; i++) {
    CsvReader::Record record = reader.next().value();
    for (size_t j = 0; j < inputSize; j++) {
      sample[i + j] = atof(record.get(j + 1).c_str());
      if (record.get("Species").compare("Iris-setosa") == 0) {
        answers[i] = 0;
      } else if (record.get("Species").compare("Iris-versicolor") == 0) {
        answers[i] = 1;
      } else if (record.get("Species").compare("Iris-virginica") == 0) {
        answers[i] = 2;
      }
    }
  }

  NeuralNetwork network(configuration);
  network.setLogger(&std::cout)
    .setInputSample(sample, sampleSize)
    .setBatchSize(1)
    .setEpochs(200)
    .setLearningRate(1.f)
    .setType(NeuralNetwork::NetworkType::CLASSIFIER)
    .setAnswers(answers)
    .setActivator(ParametricActivator<>::Relu)
    .setLossFunction(LossFunction::CrossEntropy)
    .train();

  return 0;
}