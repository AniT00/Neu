#include "Neu/CsvReader.h"

#include <assert.h>

CsvReader::CsvReader(const char* fileName)
{
  open(fileName);
}

CsvReader::CsvReader(const char* fileName,
                     std::initializer_list<std::string> identifiers)
  : m_file(fileName, std::ios_base::in)
  , m_last_record(identifiers.size())
{
  setIdentifiers(identifiers);
}

void
CsvReader::setIdentifiers(std::initializer_list<std::string> identifiers)
{
  m_last_record.setIdentifiers(identifiers);
}

void
CsvReader::setIdentifiers(const Record& line)
{
  m_last_record.setIdentifiers(line);
}

CsvReader::Record
CsvReader::getRecord(size_t index) const
{
  if (m_lineCount == -1) {
    getLineCount();
	}
  if (index > m_lineCount) {
    throw std::out_of_range("Index is out of range.");
	}
  std::streampos g = m_file.tellg();
  unsafeGoToRecord(index);
  std::optional<std::string> buffer = readLine();
  if (!buffer) {
    return {};
  }
  m_file.seekg(g);
  Record r(buffer.value(), m_recordIndex);
  return r;
}

std::optional<const std::reference_wrapper<CsvReader::Record>>
CsvReader::next()
{
  // Check if already storing this record.
  if (m_last_record.getIndex() != m_recordIndex - 1) {
    m_recordIndex++;
    std::optional<std::string> buffer = readLine();
    if (!buffer) {
      return {};
    }
    m_last_record.setContent(buffer.value(), m_recordIndex);
  }
  return m_last_record;
}

void
CsvReader::skip()
{
  m_file.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
}

const CsvReader::Record&
CsvReader::getLast() const
{
  return m_last_record;
}

void
CsvReader::goToRecord(size_t index)
{
  // TODO CsvReader::goToRecord: Maybe it can be optimized if we store every
  // record first char position in some container.
  unsafeGoToRecord(index);
}

void
CsvReader::open(const char* fileName)
{
  m_file.open(fileName);
  m_recordIndex = 0;
  m_lineCount = -1;
  if (m_file.fail()) {
    throw std::runtime_error("No such file");
  }
  std::optional<std::string> line = readLine();
  if (!line) {
    throw std::runtime_error("Empty file");
  }
  m_last_record = Record(line.value(), 0);
  m_file.seekg(0);
}

void
CsvReader::close()
{
  m_file.close();
}

size_t
CsvReader::getLineCount() const
{
  std::streampos g = m_file.tellg();
  m_file.seekg(0);
  if (m_lineCount == -1) {
    m_lineCount = std::count(std::istreambuf_iterator<char>(m_file),
                             std::istreambuf_iterator<char>(),
                             '\n');
  }
  m_file.seekg(g);
  return m_lineCount;
}

CsvReader::const_iterator
CsvReader::cbegin() const
{
  return const_iterator(*this);
}

CsvReader::const_iterator
CsvReader::cend() const
{
  return const_iterator(*this, getLineCount());
}

void
CsvReader::unsafeGoToRecord(size_t index) const
{
  if (index > m_recordIndex) {
    for (size_t i = m_recordIndex; i < index; i++) {
      unsafeSkip();
    }
  } else {
    m_file.seekg(0);
    for (size_t i = 0; i < index; i++) {
      unsafeSkip();
    }
  }
}

void
CsvReader::unsafeSkip() const
{
  m_file.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
}

std::optional<std::string>
CsvReader::readLine()
{
  return std::as_const(*this).readLine();
}

std::optional<std::string>
CsvReader::readLine() const
{
  std::string buffer;
  if (!std::getline(m_file, buffer)) {
    return {};
  }
  return buffer;
}

std::optional<std::string>
CsvReader::peekLine() const
{
  std::streampos pos = m_file.tellg();
  std::optional<std::string> line = readLine();
  m_file.seekg(pos);
  return line;
}

CsvReader::Record::Record(size_t size)
{
  m_values.reserve(size);
}

CsvReader::Record::Record(const std::string& line, size_t index, size_t size)
{
  if (size != 0) {
    m_values.reserve(size);
    this->setContent(line, index);
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
CsvReader::Record::setContent(const std::string& line, size_t index)
{
  std::stringstream lineStream(line);
  std::string cell;

  for (std::string& v : m_values) {
    std::getline(lineStream, cell, ',');
    v = cell;
  }
  m_index = index;
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

size_t
CsvReader::Record::getIndex() const
{
  return m_index;
}

void
CsvReader::Record::setIdentifiers(
  std::initializer_list<std::string> identifiers)
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
    iv++;
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

std::ostream&
operator<<(std::ostream& os, const CsvReader::Record& obj)
{
  for (const std::string& s : obj.m_values) {
    os << s << ",\t";
  }
  return os;
}

CsvReader::const_iterator::const_iterator()
  : m_reader(nullptr)
  , m_index(0)
{
}

CsvReader::const_iterator::const_iterator(const CsvReader& reader,
                                                 size_t index)
  : m_reader(&reader)
  , m_index(index)
{
}

CsvReader::const_iterator::const_reference
CsvReader::const_iterator::operator*() const
{
  m_record = m_reader->getRecord(m_index);
  return m_record;
}

CsvReader::const_iterator::const_pointer
CsvReader::const_iterator::operator->() const
{
  m_record = m_reader->getRecord(m_index);
  return &m_record;
}

CsvReader::const_iterator&
CsvReader::const_iterator::operator++()
{
  m_index++;
  return *this;
}

CsvReader::const_iterator
CsvReader::const_iterator::operator++(int)
{
  const_iterator temp(*this);
  ++(*this);
  return temp;
}

bool
operator==(const CsvReader::const_iterator& a,
           const CsvReader::const_iterator& b)
{
  return a.m_index == b.m_index;
}

bool
operator!=(const CsvReader::const_iterator& a,
           const CsvReader::const_iterator& b)
{
  return !(a == b);
}
