#include "CsvReader.h"
#include <sstream>

bool CsvReader::getline(float* destination, size_t length)
{
	if (_file.eof()) {
		return false;
	}
	
	std::getline(_file, line);
	lineStream.str(line);
	lineStream.seekg(0);

	for (int i = 0; i < length; i++)
	{
		std::getline(lineStream, cell, ',');
		destination[i] = std::stof(cell);
	}

	return true;
}

void CsvReader::skipline()
{
	std::getline(_file, line);
}

void CsvReader::open(const char* fileName)
{
	_file.open(fileName);
}

CsvReader::CsvReader(const char* fileName)
	: _file(fileName, std::ios_base::in), lineStream(line)
{
}
