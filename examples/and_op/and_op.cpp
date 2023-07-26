#include <iostream>
#include "NeuralNetwork.h"

constexpr std::initializer_list<size_t> configuration = { 2, 1 };

int main()
{
	NeuralNetwork network(configuration);
	const size_t sampleSize = 4;
	float sample[2 * sampleSize]{
		0, 0,
		0, 1,
		1, 0,
		1, 1
	};
	float answers[sampleSize]{
		0,
		0,
		0,
		1
	};
	network.setLogger(&std::cout)
		.setInputSample(sample, sampleSize)
		.setBatchSize(1)
		.setEpochs(200)
		.setLearningRate(10.f)
		.setAnswers(answers)
		.train();

	return 0;
}
