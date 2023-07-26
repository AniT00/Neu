#pragma once
#include <fstream>
#include <map>
#include <optional>
#include <random>
#include <sstream>

class CsvReader
{
public:
  class Line
  {
  public:
    Line() = default;

    explicit Line(size_t size);

    explicit Line(const std::string& line, size_t size = 0);

    Line(const Line& obj) = default;

    Line(Line&& obj);

    void setContent(const std::string& line);

    const std::vector<std::string>& getValues();

    size_t getColumnCount();

    void setIdentifiers(std::initializer_list<std::string> identifiers);

    void setIdentifiers(const Line& line);

    const std::string& get(const std::string&);

    const std::string& get(size_t id);

    std::vector<std::string>::const_iterator begin() const;

    std::vector<std::string>::const_iterator end() const;

    Line& operator=(Line&& obj);

  private:
    std::map<std::string, const std::string*> m_identifiers;
    std::vector<std::string> m_values;
    size_t m_columnCount = 0;
  };

  CsvReader(const char* fileName);

  CsvReader(const char* fileName,
            std::initializer_list<std::string> identifiers);

  void setIdentifiers(std::initializer_list<std::string> identifiers);

  void setIdentifiers(const Line& line);

  std::optional<Line> next();

  void goToLine(size_t index);

  void open(const char* fileName);

  void close();

  size_t getLineCount()
  {
    if (m_lineCount == -1) {
      m_lineCount = std::count(std::istreambuf_iterator<char>(m_file),
                               std::istreambuf_iterator<char>(),
                               '\n');
    }
    return m_lineCount;
  }

private:
  std::fstream m_file;

  Line m_last_line;

  size_t m_lineCount = -1;
};