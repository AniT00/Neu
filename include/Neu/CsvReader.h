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

    explicit Record(const std::string& line, size_t index, size_t size = 0);

    Record(const Record& obj)
      : m_identifiers(obj.m_identifiers)
      , m_values(obj.m_values)
      , m_index(obj.m_index)
    {
    }

    Record(Record&& obj);

    void setContent(const std::string& line, size_t index);

    const std::vector<std::string>& getValues();

    size_t getColumnCount() const;

    size_t getIndex() const;

    void setIdentifiers(std::initializer_list<std::string> identifiers);

    void setIdentifiers(const Record& line);

    const std::string& get(const std::string&);

    const std::string& get(size_t id);

    std::vector<std::string>::const_iterator begin() const;

    std::vector<std::string>::const_iterator end() const;

    // Record& operator=(Record&& obj) = default;

    Record& operator=(Record&& obj);

    friend std::ostream& operator<<(std::ostream& os, const Record& obj);

  private:
    std::map<std::string, const std::string*> m_identifiers;
    std::vector<std::string> m_values;
    size_t m_index;
  };

  class const_iterator
  {
  public:
    using iterator_category = std::random_access_iterator_tag;
    using value_type = Record;
    using difference_type = size_t;
    using pointer = value_type*;
    using const_pointer = const value_type*;
    using reference = value_type&;
    using const_reference = const value_type&;

    const_iterator();

    const_iterator(const CsvReader& reader, size_t index = 0);

    const_iterator(const const_iterator& obj) = default;

    const_iterator(const_iterator&& obj) = default;

    const_reference operator*() const;
    const_pointer operator->() const;

    const_iterator& operator++();
    const_iterator operator++(int);

    friend bool operator==(const const_iterator& a, const const_iterator& b);
    friend bool operator!=(const const_iterator& a, const const_iterator& b);

    const_iterator& operator=(const_iterator& obj) = default;

    operator size_t() { return 0; }

  private:
    const CsvReader* m_reader;
    mutable Record m_record;
    difference_type m_index;
  };

  CsvReader(const char* fileName);

  CsvReader(const char* fileName,
            std::initializer_list<std::string> identifiers);

  void setIdentifiers(std::initializer_list<std::string> identifiers);

  void setIdentifiers(const Record& line);

  Record getRecord(size_t index) const;

  std::optional<const std::reference_wrapper<Record>> next();

  void skip();

  const Record& getLast() const;

  void goToRecord(size_t index);

  void open(const char* fileName);

  void close();

  size_t getLineCount() const;

  const_iterator cbegin() const;

  const_iterator cend() const;

private:
  /// <summary>
  /// Changes object state. Implemented to prevent code duplication in
  /// getRecord(size_t) const since goToRecord(size_t) is non-const
  /// </summary>
  void unsafeGoToRecord(size_t index) const;

  void unsafeSkip() const;

  std::optional<std::string> readLine();

  std::optional<std::string> readLine() const;

  std::optional<std::string> peekLine() const;

  mutable std::ifstream m_file;

  Record m_last_record;

  size_t m_recordIndex;
  mutable size_t m_lineCount;
};