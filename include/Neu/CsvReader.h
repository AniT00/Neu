#pragma once
#include <random>
#include <fstream>
#include <sstream>
#include <map>


class CsvReader
{
public:
	CsvReader(const char* fileName);

	CsvReader(const char* fileName, std::initializer_list<std::string> identifiers);

	bool next();

	std::string get(const std::string&);

	std::string get(size_t id);

	void open(const char* fileName);

	void close();

private:

	std::fstream m_file;

	std::map<std::string, std::string*>* m_identifiers = nullptr;
	std::vector<std::string> m_values;
};