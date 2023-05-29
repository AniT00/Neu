# CMakeList.txt: проект CMake для Neu; включите исходный код и определения,
# укажите здесь логику для конкретного проекта.
#

# Добавьте источник в исполняемый файл этого проекта.
#add_library (Neu "NeuralNetwork" )

set(INCROOT ${PROJECT_SOURCE_DIR}/include/Neu)
set(SRCROOT ${PROJECT_SOURCE_DIR}/src/Neu)

set(SRC
	${INCROOT}/NeuralNetwork.h
	${SRCROOT}/NeuralNetwork.cpp
	${INCROOT}/CsvReader.h
	${SRCROOT}/CsvReader.cpp
	${INCROOT}/Layer.h
	${SRCROOT}/Layer.cpp
	${INCROOT}/Activators.h
	${SRCROOT}/Activators.cpp
)
source_group("" FILES ${SRC})

add_library(Neu STATIC ${SRC})

# OpenCL sources
target_include_directories(Neu PUBLIC ${PROJECT_SOURCE_DIR}/deps/OpenCL/include)

target_include_directories(Neu PUBLIC ${INCROOT})

if (CMAKE_VERSION VERSION_GREATER 3.12)
  set_property(TARGET Neu PROPERTY CXX_STANDARD 20)
endif()
