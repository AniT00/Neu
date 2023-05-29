#include <iostream>
#include "CL/cl.h"
#include "NeuralNetwork.h"

using namespace std;

int main()
{
	size_t sizes[3]{ 2, 3, 1 };
	NeuralNetwork network({ 1, 2, 3 });
	return 0;
}
