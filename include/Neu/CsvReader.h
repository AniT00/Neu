#pragma once
#include <fstream>
#include <map>
#include <optional>
#include <random>
#include <sstream>

class CsvReader
{
public:
  class Record
  {
  public:
    Record() = default;

    explicit Record(size_t size);

    explicit Record(const std::string& line, size_t size = 0);

    Record(const Record& obj) = default;

    Record(Record&& obj);

    void setContent(const std::string& line);

    const std::vector<std::string>& getValues();

    size_t getColumnCount() const;

    void setIdentifiers(std::initializer_list<std::string> identifiers);

    void setIdentifiers(const Record& line);

    const std::string& get(const std::string&);

    const std::string& get(size_t id);

    std::vector<std::string>::const_iterator begin() const;

    std::vector<std::string>::const_iterator end() const;

    Record& operator=(Record&& obj);

  private:
    std::map<std::string, const std::string*> m_identifiers;
    std::vector<std::string> m_values;
  };

  CsvReader(const char* fileName);

  CsvReader(const char* fileName,
            std::initializer_list<std::string> identifiers);

  void setIdentifiers(std::initializer_list<std::string> identifiers);

  void setIdentifiers(const Record& line);

  std::optional<const std::reference_wrapper<Record>> next();

  void goToLine(size_t index);

  void open(const char* fileName);

  void close();

  size_t getLineCount()
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

private:
  std::optional<std::string> readLine();

  std::ifstream m_file;

  Record m_last_line;

  size_t m_lineCount = -1;
};