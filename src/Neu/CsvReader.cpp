#include "CsvReader.h"
#include <sstream>

CsvReader::CsvReader(const char* fileName)
	: m_file(fileName, std::ios_base::in)
{
	std::string line;
	std::getline(m_file, line);
	std::stringstream lineStream(line);
	std::string cell;

	while(std::getline(lineStream, cell, ','))
	{
		m_values.push_back(std::string());
	}
	m_file.seekg(0);
}

CsvReader::CsvReader(const char* fileName, std::initializer_list<std::string> identifiers) 
	: m_file(fileName, std::ios_base::in), m_values(identifiers.size()) {
	m_identifiers = new std::map<std::string, std::string*>();
	auto iv = m_values.begin();
	for (auto id = identifiers.begin(); id != identifiers.end(); id++, iv++) {
		m_identifiers->insert({ *id, &*iv });
	}
}

bool CsvReader::next() {
	std::string line;
	if (!std::getline(m_file, line)) {
		return false;
	}

	std::stringstream lineStream(line);
	std::string cell;

	for (auto& v : m_values)
	{
		std::getline(lineStream, cell, ',');
		v = cell;
	}

	return true;
}

std::string CsvReader::get(const std::string& id) {
	return *m_identifiers->at(id);
}

std::string CsvReader::get(size_t id) {
	return m_values[id];
}

void CsvReader::open(const char* fileName) {
	m_file.open(fileName);
}

void CsvReader::close() {
	m_file.close();
}
