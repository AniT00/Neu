#include "Activators.h"

void WeightInitilizer::normal_xavier_initilize(float** matrix, size_t rows, size_t columns, size_t n) {
	static std::default_random_engine generator;
	std::normal_distribution<float> normal(0, sqrt(1.f / n));
	for (size_t i = 0; i < rows; i++)
	{
		for (size_t j = 0; j < columns; j++)
		{
			matrix[i][j] = normal(generator);
		}
	}
}