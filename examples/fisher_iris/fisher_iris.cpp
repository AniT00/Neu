#include "CsvReader.h"
#include "NeuralNetwork.h"
#include <iostream>

constexpr std::initializer_list<size_t> configuration = { 4, 5, 5, 1 };

int
main()
{
  CsvReader reader("samples/Iris.csv");
  auto record = reader.next();
  size_t sampleSize = reader.getLineCount() - 1;
  size_t inputSize = record.value().get().getColumnCount() - 1;
  float* sample = new float[inputSize * sampleSize]{ 0 };
  float* answers = new float[sampleSize]{ 0 };
  for (size_t i = 0; i < sampleSize; i++) {
    CsvReader::Record record = reader.next().value();
    for (size_t j = 0; j < inputSize; j++) {
      sample[i+j] = atof(record.get(j+1).c_str());
    }
	}

  NeuralNetwork network(configuration);
  network.setLogger(&std::cout)
    .setInputSample(sample, sampleSize)
    .setBatchSize(1)
    .setEpochs(200)
    .setLearningRate(1.f)
    .setAnswers(answers)
    .setActivator(Activator::Sigmoid)
    .setLossFunction(LossFunction::CrossEntropy)
    .train();

  return 0;
}