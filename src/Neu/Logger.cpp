#include "Logger.h"

Logger::Logger()
	: m_stream(nullptr) {

}

Logger::Logger(std::ostream* stream)
	: m_stream(stream) {
}

void Logger::log(const std::string& text) {
	*m_stream << text;
}

void Logger::logln(const std::string& text) {
	*m_stream << text << '\n';
}