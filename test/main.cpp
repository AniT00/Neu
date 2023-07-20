#include <iostream>
#include "CL/cl.h"
#include "NeuralNetwork.h"
#include "CsvReader.h"

using namespace std;

int main()
{
	NeuralNetwork network({ 2, 3, 1 });
	size_t sampleSize = 4;
	float sample[]{
		0, 0,
		0, 1,
		1, 0,
		1, 1
	};
	float answers[]{
		0,
		0,
		1,
		0.5
	};
	/*CsvReader reader("Iris.csv");
	size_t sampleSize = 150;
	float* sample = new float[sampleSize * 4];
	float* answers = new float[sampleSize];
	reader.next();
	for (int i = 0; i < sampleSize; i++) {
		reader.next();
		for (int j = 0; j < 4; j++) {
			sample[i * 4 + j] = stof(reader.get(j + 1));
		}

		std::string type = reader.get(5);
		if (type.compare("Iris-setosa") == 0) {
			answers[i] = 0;
		}
		else if (type.compare("Iris-versicolor") == 0) {
			answers[i] = 1;
		}
		else if (type.compare("Iris-virginica") == 0) {
			answers[i] = 2;
		}
	}*/
	network.setLogger(&std::cout)
		//.logOutput(true)
		.setInputSample(sample, sampleSize)
		.setBatchSize(1)
		.setEpochs(1000)
		.setLearningRate(0.4)
		.setAnswers(answers)
		.train();
	return 0;
}
