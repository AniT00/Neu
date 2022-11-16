#include <iostream>

#include "NeuralNetwork.h"

using namespace std;

int main()
{
	size_t sizes[3]{ 2, 3, 1 };
	NeuralNetwork network(sizes, 3);
	cout << "Hello CMake.!!!!!!" << endl;
	return 0;
}
