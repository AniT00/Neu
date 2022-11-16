#pragma once
#include <random>
#include <fstream>
#include <sstream>


class CsvReader
{
public:
	bool getline(float* destination, size_t length);

	void skipline();

	void open(const char* fileName);

	CsvReader(const char* fileName);

private:
	std::fstream _file;
	std::string line;
	std::stringstream lineStream;
	std::string cell;
};