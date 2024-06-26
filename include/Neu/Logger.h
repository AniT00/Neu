#pragma once
#include <ostream>


class Logger
{
public:
  Logger();

  Logger(std::ostream* stream);

  void log(const std::string& text);

  void logln(const std::string& text);

  template<typename T>
  Logger& operator<<(const T& obj)
  {
    *m_stream << obj;
    return *this;
  }

private:
  std::ostream* m_stream;
};