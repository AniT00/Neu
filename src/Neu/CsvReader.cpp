#include "Neu/CsvReader.h"

#include <assert.h>

CsvReader::CsvReader(const char* fileName)
  : m_file(fileName, std::ios_base::in)
{
  if (!next()) {
    throw std::runtime_error("Empty file");
  }
  m_file.seekg(0);
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
CsvReader::setIdentifiers(const Line& line)
{
  m_last_line.setIdentifiers(line);
}

std::optional<CsvReader::Line>
CsvReader::next()
{
  std::string buffer;
  if (!std::getline(m_file, buffer)) {
    return {};
  }
  m_last_line.setContent(buffer);

  return Line(m_last_line);
}

void
CsvReader::goToLine(size_t index)
{
  m_file.seekg(0);
  for (size_t i = 0; i < index; i++) {
    next();
  }
}

void
CsvReader::open(const char* fileName)
{
  m_file.open(fileName);
}

void
CsvReader::close()
{
  m_file.close();
}

CsvReader::Line::Line(size_t size)
{
  m_values.reserve(size);
  m_columnCount = size;
}

CsvReader::Line::Line(const std::string& line, size_t size)
{
  if (size != 0) {
    m_values.reserve(size);
  }
  this->setContent(line);
  m_columnCount = m_values.size();
}

CsvReader::Line::Line(Line&& obj)
{
  *this = std::move(obj);
}

void
CsvReader::Line::setContent(const std::string& line)
{
  std::stringstream lineStream(line);
  std::string cell;

  for (std::string& v : m_values) {
    std::getline(lineStream, cell, ',');
    v = cell;
  }
}

const std::vector<std::string>&
CsvReader::Line::getValues()
{
  return m_values;
}

size_t
CsvReader::Line::getColumnCount()
{
  return m_columnCount;
}

void
CsvReader::Line::setIdentifiers(std::initializer_list<std::string> identifiers)
{
  auto it_v = m_values.data();
  for (const auto& e : identifiers) {
    m_identifiers.insert({ e, it_v });
    it_v++;
  }
}

void
CsvReader::Line::setIdentifiers(const Line& line)
{
  m_identifiers.clear();
  auto iv = m_values.data();
  for (const std::string& e : line) {
    m_identifiers.insert({ e, iv });
  }
}

const std::string&
CsvReader::Line::get(const std::string& id)
{
  return *m_identifiers.at(id);
}

const std::string&
CsvReader::Line::get(size_t id)
{
  return m_values[id];
}

std::vector<std::string>::const_iterator
CsvReader::Line::begin() const
{
  return m_values.begin();
}

std::vector<std::string>::const_iterator
CsvReader::Line::end() const
{
  return m_values.end();
}

CsvReader::Line&
CsvReader::Line::operator=(Line&& obj)
{
  m_identifiers.swap(obj.m_identifiers);
  m_values.swap(obj.m_values);
  std::swap(m_columnCount, obj.m_columnCount);
  return *this;
}
