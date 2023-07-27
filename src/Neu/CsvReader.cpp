#include "Neu/CsvReader.h"

#include <assert.h>

CsvReader::CsvReader(const char* fileName)
{
  open(fileName);
}

CsvReader::CsvReader(const char* fileName,
                     std::initializer_list<std::string> identifiers)
  : m_file(fileName, std::ios_base::in)
  , m_last_line(identifiers.size())
{
  setIdentifiers(identifiers);
}

void
CsvReader::setIdentifiers(std::initializer_list<std::string> identifiers)
{
  m_last_line.setIdentifiers(identifiers);
}

void
CsvReader::setIdentifiers(const Record& line)
{
  m_last_line.setIdentifiers(line);
}

std::optional<const std::reference_wrapper<CsvReader::Record>>
CsvReader::next()
{
  std::optional<std::string> buffer = readLine();
  if (!buffer) {
    return {};
  }
  m_last_line.setContent(buffer.value());

  return m_last_line;
}

void
CsvReader::goToLine(size_t index)
{
  m_file.seekg(0);
  for (size_t i = 0; i < index; i++) {
    readLine();
  }
}

void
CsvReader::open(const char* fileName)
{
  m_file.open(fileName);
  if (m_file.fail()) {
    throw std::runtime_error("No such file");
  }
  std::optional<std::string> line = readLine();
  if (!line) {
    throw std::runtime_error("Empty file");
  }
  m_last_line = Record(line.value());
  m_file.seekg(0);
}

void
CsvReader::close()
{
  m_file.close();
}

std::optional<std::string>
CsvReader::readLine()
{
  std::string buffer;
  if (!std::getline(m_file, buffer)) {
    return {};
  }
  return buffer;
}

CsvReader::Record::Record(size_t size)
{
  m_values.reserve(size);
}

CsvReader::Record::Record(const std::string& line, size_t size)
{
  if (size != 0) {
    m_values.reserve(size);
		this->setContent(line);
  } else {
    std::stringstream lineStream(line);
    std::string cell;

    while (std::getline(lineStream, cell, ',')) {
      m_values.push_back(cell);
    }
	}
}

CsvReader::Record::Record(Record&& obj)
{
  *this = std::move(obj);
}

void
CsvReader::Record::setContent(const std::string& line)
{
  std::stringstream lineStream(line);
  std::string cell;

  for (std::string& v : m_values) {
    std::getline(lineStream, cell, ',');
    v = cell;
  }
}

const std::vector<std::string>&
CsvReader::Record::getValues()
{
  return m_values;
}

size_t
CsvReader::Record::getColumnCount() const
{
  return m_values.size();
}

void
CsvReader::Record::setIdentifiers(std::initializer_list<std::string> identifiers)
{
  auto it_v = m_values.data();
  for (const auto& e : identifiers) {
    m_identifiers.insert({ e, it_v });
    it_v++;
  }
}

void
CsvReader::Record::setIdentifiers(const Record& line)
{
  m_identifiers.clear();
  auto iv = m_values.data();
  for (const std::string& e : line) {
    m_identifiers.insert({ e, iv });
  }
}

const std::string&
CsvReader::Record::get(const std::string& id)
{
  return *m_identifiers.at(id);
}

const std::string&
CsvReader::Record::get(size_t id)
{
  return m_values[id];
}

std::vector<std::string>::const_iterator
CsvReader::Record::begin() const
{
  return m_values.begin();
}

std::vector<std::string>::const_iterator
CsvReader::Record::end() const
{
  return m_values.end();
}

CsvReader::Record&
CsvReader::Record::operator=(Record&& obj)
{
  m_identifiers.swap(obj.m_identifiers);
  m_values.swap(obj.m_values);
  return *this;
}
