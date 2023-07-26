#include "CsvReader.h"
#include "NeuralNetwork.h"
#include <iostream>

constexpr std::initializer_list<size_t> configuration = { 4, 5, 5, 1 };

int
main()
{
  CsvReader reader("samples/Iris.csv");
  auto line = reader.next();
  if (!line) {
    std::cerr << "Invalid file.\n";
    return 1;
  }
  reader.setIdentifiers(line.value());
  size_t sampleSize = reader.getLineCount();
  size_t inputSize = line.value().getColumnCount() - 1;
  float* sample = new float[inputSize * sampleSize]{ 0 };
  float* answers = new float[sampleSize]{ 0 };

  NeuralNetwork network(configuration);
  network.setLogger(&std::cout)
    .setInputSample(sample, sampleSize)
    .setBatchSize(1)
    .setEpochs(200)
    .setLearningRate(10.f)
    .setAnswers(answers)
    .setActivator(Activators::sigmoid);
    .train();

  return 0;
}